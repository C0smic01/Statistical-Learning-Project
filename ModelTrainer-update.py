from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback
)
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import defaultdict
import logging
import sys
from tqdm import tqdm
import time
from datetime import timedelta

# Set up logging to show training progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Function to format time
def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

logger.info("Loading GoEmotions dataset...")
# 1. Load and preprocess dataset
dataset = load_dataset("go_emotions")
logger.info(f"Dataset loaded with {len(dataset['train'])} training examples")

# Define the 7 target emotions we want to keep
target_emotions = {'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'}

# Function to filter examples that have at least one label in target_emotions
def filter_by_target_emotions(example):
    labels = [dataset['train'].features['labels'].feature.names[idx] for idx in example['labels']]
    return any(label in target_emotions for label in labels)

logger.info("Filtering dataset to keep only examples with target emotions...")
filtered_dataset = dataset.filter(filter_by_target_emotions)
logger.info(f"After filtering: {len(filtered_dataset['train'])} training examples remain")

# Now we can directly use the labels without mapping since we've already filtered
j_hartmann_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

# Function to convert labels (now simpler since we only have target emotions)
def map_labels(example):
    # Get all labels (we know they're all in our target set)
    labels = [dataset['train'].features['labels'].feature.names[idx] for idx in example['labels']]
    
    # If multiple labels, choose the first one (or implement more sophisticated logic)
    chosen_label = labels[0]
    
    label_id = j_hartmann_labels.index(chosen_label)
    
    return {'text': example['text'], 'label': label_id}

logger.info("Processing labels...")
mapped_dataset = filtered_dataset.map(map_labels, remove_columns=['labels', 'id'])

# Display label distribution
logger.info("Label distribution after filtering and mapping:")
label_distribution = defaultdict(int)
for example in tqdm(mapped_dataset["train"], desc="Counting labels"):
    label_distribution[example["label"]] += 1

for label_id, count in label_distribution.items():
    percent = (count / len(mapped_dataset["train"])) * 100
    logger.info(f"Label {j_hartmann_labels[label_id]}: {count} examples ({percent:.2f}%)")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Create a weighted loss function using the class weights
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# Create a training callback to track time per epoch
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
        logger.info(f"Estimated completion time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + estimated_time))}")

# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    
    report = classification_report(
        labels, 
        predictions, 
        target_names=j_hartmann_labels,
        output_dict=True
    )
    
    results = {
        "accuracy": accuracy,
        "f1": f1
    }
    
    for label, metrics in report.items():
        if label in j_hartmann_labels:
            results[f"f1_{label}"] = metrics['f1-score']
    
    return results

if __name__ == '__main__':
    # Map 28 GoEmotions labels -> 7 labels from j-hartmann model
    emotion_mapping = {
        'anger': 'anger', 'disgust': 'disgust', 'fear': 'fear', 'joy': 'joy',
        'neutral': 'neutral', 'sadness': 'sadness', 'surprise': 'surprise',
        'admiration': 'joy', 'amusement': 'joy', 'approval': 'joy',
        'caring': 'joy', 'desire': 'joy', 'excitement': 'joy',
        'gratitude': 'joy', 'love': 'joy', 'optimism': 'joy',
        'pride': 'joy', 'relief': 'joy', 'confusion': 'fear',
        'curiosity': 'surprise', 'embarrassment': 'fear',
        'grief': 'sadness', 'nervousness': 'fear',
        'realization': 'surprise', 'remorse': 'sadness',
        'annoyance': 'anger', 'disappointment': 'sadness',
        'disapproval': 'anger'
    }

    # Define the target labels for our model
    j_hartmann_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

    logger.info("Loading GoEmotions dataset...")
    # 1. Load and preprocess dataset
    dataset = load_dataset("go_emotions")
    logger.info(f"Dataset loaded with {len(dataset['train'])} training examples")

    logger.info("Mapping labels from GoEmotions to 7-class emotion schema...")
    # Apply label mapping
    try:
        mapped_dataset = dataset.map(map_labels, remove_columns=['labels', 'id'])
    except Exception as e:
        logger.error(f"Error during label mapping: {e}")
        raise

    # Display label distribution
    logger.info("Label distribution after mapping:")
    label_distribution = defaultdict(int)
    for example in tqdm(mapped_dataset["train"], desc="Counting labels"):
        label_distribution[example["label"]] += 1

    for label_id, count in label_distribution.items():
        percent = (count / len(mapped_dataset["train"])) * 100
        logger.info(f"Label {j_hartmann_labels[label_id]}: {count} examples ({percent:.2f}%)")

    # 2. Prepare model and tokenizer
    logger.info("Loading pre-trained model and tokenizer...")
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=7,
        ignore_mismatched_sizes=True
    )

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    try:
        tokenized_dataset = mapped_dataset.map(
            tokenize_function, 
            batched=True,
            desc="Tokenizing",
            num_proc=1  # Disable multiprocessing to avoid errors on Windows
        )
    except Exception as e:
        logger.error(f"Error during tokenizing: {e}")
        raise

    # 3. Calculate class weights to handle data imbalance
    total = sum(label_distribution.values())
    class_weights = [total/(len(j_hartmann_labels)*count) for label_id, count in sorted(label_distribution.items())]
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    logger.info("Class weights calculated to handle class imbalance:")
    for i, weight in enumerate(class_weights):
        logger.info(f"Class {j_hartmann_labels[i]}: {weight:.4f}")

    # 4. Set up training arguments
    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./emotion_model",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="steps",  # Use evaluation_strategy for compatibility
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

    # Use a data collator with padding for more efficient batching
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Add the timer callback to the trainer
    timer_callback = TimingCallback()

    # Create the trainer with our weighted loss
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        class_weights=weights_tensor,
        callbacks=[timer_callback]
    )

    # 5. Train the model
    logger.info("Starting model training...")
    start_time = time.time()
    total_steps = len(tokenized_dataset["train"]) // training_args.per_device_train_batch_size * training_args.num_train_epochs
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Estimated training time (based on previous runs): ~{format_time(total_steps * 0.1)} (may vary)")

    train_result = trainer.train()

    # Print training metrics and total time
    total_training_time = time.time() - start_time
    logger.info(f"Training completed in {format_time(total_training_time)}")
    logger.info(f"Training completed. Metrics: {train_result.metrics}")

    # 6. Evaluate on test set
    logger.info("Evaluating on test set...")
    test_start_time = time.time()
    test_results = trainer.evaluate(tokenized_dataset["test"])
    test_time = time.time() - test_start_time
    logger.info(f"Test evaluation completed in {format_time(test_time)}")
    logger.info(f"Test results: {test_results}")

    # 7. Save the model
    logger.info("Saving the fine-tuned model...")
    save_start_time = time.time()
    model.save_pretrained("./emotion_model")
    tokenizer.save_pretrained("./emotion_model")
    save_time = time.time() - save_start_time
    logger.info(f"Model saved to ./emotion_model in {format_time(save_time)}")

    # Print final summary
    logger.info(f"Complete pipeline time: {format_time(time.time() - start_time)}")
    logger.info("Training pipeline completed!")