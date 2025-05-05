import logging
import sys
import time
from collections import defaultdict
from datetime import timedelta

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    EarlyStoppingCallback
)
from tqdm import tqdm

TARGET_EMOTIONS = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
MODEL_NAME = "bhadresh-savani/albert-base-v2-emotion"

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

def load_and_preprocess_dataset():
    logger.info("Loading mteb/emotion dataset...")
    dataset = load_dataset("mteb/emotion")
    
    def map_labels(example):
        label_id = TARGET_EMOTIONS.index(example['label_text'])
        return {'text': example['text'], 'label': label_id}
    
    logger.info("Processing labels...")
    processed_dataset = dataset.map(map_labels)
    return processed_dataset

def analyze_label_distribution(dataset):
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
    logger.info(f"Initializing {MODEL_NAME} model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(TARGET_EMOTIONS),
        ignore_mismatched_sizes=True
    )
    return model, tokenizer

def tokenize_dataset(dataset, tokenizer):
    logger.info("Tokenizing dataset...")
    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        desc="Tokenizing",
        num_proc=4
    )
    return tokenized_dataset

def calculate_class_weights(dataset):
    logger.info("Calculating class weights...")
    labels = [example["label"] for example in dataset["train"]]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    logger.info("Class weights:")
    for i, weight in enumerate(class_weights):
        logger.info(f"{TARGET_EMOTIONS[i]}: {weight:.4f}")
    
    return class_weights

class WeightedTrainer(Trainer):
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
        "f1_weighted": f1
    }
    
    for label, metrics in report.items():
        if label in TARGET_EMOTIONS:
            results[f"f1_{label}"] = metrics['f1-score']
    
    return results

def setup_training_args():
    return TrainingArguments(
        output_dir="./emotion_model",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        logging_dir='./logs',
        logging_steps=100,
        report_to="tensorboard",
        warmup_steps=100,
        gradient_accumulation_steps=2,
        fp16=True,
        gradient_checkpointing=False,
        optim="adamw_torch",
        lr_scheduler_type="linear",
    )

def train_model(model, tokenizer, tokenized_dataset, class_weights):
    training_args = setup_training_args()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    callbacks = [TimingCallback(), EarlyStoppingCallback(early_stopping_patience=2)]

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        class_weights=class_weights,
        callbacks=callbacks
    )

    logger.info("Starting training...")
    start_time = time.time()
    train_result = trainer.train()
    logger.info(f"Training completed in {format_time(time.time() - start_time)}")
    
    return trainer, train_result

def evaluate_model(trainer, tokenized_dataset):
    logger.info("Evaluating on test set...")
    test_start_time = time.time()
    test_results = trainer.evaluate(tokenized_dataset["test"])
    logger.info(f"Test evaluation completed in {format_time(time.time() - test_start_time)}")
    logger.info(f"Test results: {test_results}")
    return test_results

def save_model(model, tokenizer):
    logger.info("Saving model...")
    save_start_time = time.time()
    model.save_pretrained("./emotion_model")
    tokenizer.save_pretrained("./emotion_model")
    logger.info(f"Model saved in {format_time(time.time() - save_start_time)}")

def main():
    # 1. Load and preprocess data
    dataset = load_and_preprocess_dataset()
    label_distribution = analyze_label_distribution(dataset)
    
    # 2. Initialize model and tokenizer
    model, tokenizer = initialize_model_and_tokenizer()
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    class_weights = calculate_class_weights(dataset)
    
    # 3. Train the model
    trainer, train_result = train_model(model, tokenizer, tokenized_dataset, class_weights)
    
    # 4. Evaluate and save
    evaluate_model(trainer, tokenized_dataset)
    save_model(model, tokenizer)
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()