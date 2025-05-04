import logging
import sys
import time
import random
from collections import defaultdict
from datetime import timedelta

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from tqdm import tqdm

TARGET_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
MAX_DATASET_SIZE = 50000

# Setup logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def load_and_filter_dataset():
    # Load the new dataset
    logger.info("Loading Villian7/Emotions_Data dataset...")
    dataset = load_dataset("Villian7/Emotions_Data")
    logger.info(f"Original dataset size: {len(dataset['train'])} examples")

    def filter_fn(example):
        return example['label_text'] in TARGET_EMOTIONS

    logger.info("Filtering dataset for target emotions...")
    filtered_dataset = dataset.filter(filter_fn)
    logger.info(f"Filtered dataset size: {len(filtered_dataset['train'])} examples")
    
    # Limit to 50k examples
    if len(filtered_dataset['train']) > MAX_DATASET_SIZE:
        logger.info(f"Limiting dataset to {MAX_DATASET_SIZE} examples...")
        
        emotions_dict = defaultdict(list)
        for i, example in enumerate(filtered_dataset['train']):
            emotions_dict[example['label_text']].append(i)
        
        examples_per_emotion = MAX_DATASET_SIZE // len(TARGET_EMOTIONS)
        remaining = MAX_DATASET_SIZE % len(TARGET_EMOTIONS)
        
        selected_indices = []
        for emotion in TARGET_EMOTIONS:
            # Get all indices for this emotion
            emotion_indices = emotions_dict[emotion]
            # Select either all indices or the target number, whichever is smaller
            num_to_select = min(len(emotion_indices), examples_per_emotion)
            # Randomly sample the required number of indices
            sampled_indices = random.sample(emotion_indices, num_to_select)
            selected_indices.extend(sampled_indices)
            
            # Add remaining examples from emotions that have more data
            if remaining > 0 and len(emotion_indices) > num_to_select:
                extra_indices = random.sample(
                    [idx for idx in emotion_indices if idx not in sampled_indices],
                    min(remaining, len(emotion_indices) - num_to_select)
                )
                selected_indices.extend(extra_indices)
                remaining -= len(extra_indices)
        
        filtered_dataset['train'] = filtered_dataset['train'].select(selected_indices)
        logger.info(f"Limited dataset size: {len(filtered_dataset['train'])} examples")
    
    if 'validation' not in filtered_dataset and 'test' not in filtered_dataset:
        logger.info("Splitting dataset into train, validation, and test sets...")
        splits = filtered_dataset["train"].train_test_split(test_size=0.2, seed=42)
        train_data = splits["train"]
        temp_splits = splits["test"].train_test_split(test_size=0.5, seed=42)
        val_data = temp_splits["train"]
        test_data = temp_splits["test"]
        
        filtered_dataset = {
            "train": train_data,
            "validation": val_data,
            "test": test_data
        }
    
    return filtered_dataset

def process_labels(dataset):
    # Process labels to match our target emotion classes
    def map_labels(example):
        # Map emotion string to numeric label
        label_id = TARGET_EMOTIONS.index(example['label_text'])
        return {'text': example['text'], 'label': label_id}

    logger.info("Processing labels...")
    processed_dataset = {}
    for split in dataset:
        processed_dataset[split] = dataset[split].map(
            map_labels, 
            remove_columns=['label_text'] + [col for col in dataset[split].column_names if col not in ['text', 'label_text']]
        )
    
    return processed_dataset

def analyze_label_distribution(dataset):
    # Analyze and log the label distribution
    logger.info("Analyzing label distribution...")
    label_distribution = defaultdict(int)
    
    for example in tqdm(dataset["train"], desc="Counting labels"):
        label_distribution[example["label"]] += 1

    logger.info("Label distribution:")
    for label_id, count in label_distribution.items():
        percent = (count / len(dataset["train"])) * 100
        logger.info(f"{TARGET_EMOTIONS[label_id]}: {count} examples ({percent:.2f}%)")
    
    return label_distribution

def initialize_model_and_tokenizer():
    # Initialize the model and tokenizer
    logger.info("Initializing model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(TARGET_EMOTIONS),
        ignore_mismatched_sizes=True
    )
    return model, tokenizer

def tokenize_dataset(dataset, tokenizer):
    # Tokenize the dataset
    logger.info("Tokenizing dataset...")
    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    tokenized_dataset = {}
    for split in dataset:
        tokenized_dataset[split] = dataset[split].map(
            tokenize_fn,
            batched=True,
            desc=f"Tokenizing {split}",
            num_proc=4
        )
    
    return tokenized_dataset

def calculate_class_weights(label_distribution):
    # Calculate class weights for imbalanced data
    logger.info("Calculating class weights...")
    total = sum(label_distribution.values())
    class_weights = [total/(len(TARGET_EMOTIONS)*count) for label_id, count in sorted(label_distribution.items())]
    
    logger.info("Class weights:")
    for i, weight in enumerate(class_weights):
        logger.info(f"{TARGET_EMOTIONS[i]}: {weight:.4f}")
    
    return torch.tensor(class_weights, dtype=torch.float32)

class WeightedTrainer(Trainer):
    # Custom trainer with weighted loss
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class TimingCallback(TrainerCallback):
    # Callback to track training time
    def __init__(self):
        self.epoch_start_time = None
        self.epoch_times = []
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        logger.info(f"Starting epoch {state.epoch+1}/{args.num_train_epochs}")
        
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        logger.info(f"Epoch {state.epoch+1} completed in {format_time(epoch_time)}")
        
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = args.num_train_epochs - state.epoch - 1
        estimated_time = avg_epoch_time * remaining_epochs
        logger.info(f"Estimated time remaining: {format_time(estimated_time)}")

def compute_metrics(eval_pred):
    # Compute evaluation metrics
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    
    report = classification_report(
        labels, 
        predictions, 
        target_names=TARGET_EMOTIONS,
        output_dict=True
    )
    
    results = {
        "accuracy": accuracy,
        "f1": f1
    }
    
    for label, metrics in report.items():
        if label in TARGET_EMOTIONS:
            results[f"f1_{label}"] = metrics['f1-score']
    
    return results

def setup_training_args():
    # Setup training arguments
    return TrainingArguments(
        output_dir="./emotion_model",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="steps",  # Updated from eval_strategy
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir='./logs',
        logging_strategy="steps",
        logging_steps=100,
        report_to="tensorboard",
    )

def train_model(model, tokenizer, tokenized_dataset, class_weights):
    # Train the model
    training_args = setup_training_args()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    timer_callback = TimingCallback()

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        class_weights=class_weights,
        callbacks=[timer_callback]
    )

    logger.info("Starting training...")
    start_time = time.time()
    train_result = trainer.train()
    logger.info(f"Training completed in {format_time(time.time() - start_time)}")
    
    return trainer, train_result

def evaluate_model(trainer, tokenized_dataset):
    # Evaluate the model on test set
    logger.info("Evaluating on test set...")
    test_start_time = time.time()
    test_results = trainer.evaluate(tokenized_dataset["test"])
    logger.info(f"Test evaluation completed in {format_time(time.time() - test_start_time)}")
    logger.info(f"Test results: {test_results}")
    return test_results

def save_model(model, tokenizer):
    # Save the trained model
    logger.info("Saving model...")
    save_start_time = time.time()
    model.save_pretrained("./emotion_model")
    tokenizer.save_pretrained("./emotion_model")
    logger.info(f"Model saved in {format_time(time.time() - save_start_time)}")

def main():
    # 1. Load and preprocess data
    raw_dataset = load_and_filter_dataset()
    processed_dataset = process_labels(raw_dataset)
    label_distribution = analyze_label_distribution(processed_dataset)
    
    # 2. Initialize model and tokenizer
    model, tokenizer = initialize_model_and_tokenizer()
    tokenized_dataset = tokenize_dataset(processed_dataset, tokenizer)
    class_weights = calculate_class_weights(label_distribution)
    
    # 3. Train the model
    trainer, train_result = train_model(model, tokenizer, tokenized_dataset, class_weights)
    
    # 4. Evaluate and save
    evaluate_model(trainer, tokenized_dataset)
    save_model(model, tokenizer)
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()