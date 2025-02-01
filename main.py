from transformers import GPT2ForSequenceClassification, TrainingArguments, Trainer, GPT2Tokenizer
from datasets import load_dataset
import pandas as pd
import numpy as np
import evaluate

# Load and preprocess the dataset
dataset = load_dataset("mteb/tweet_sentiment_extraction")

def tokenize_function(examples):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Initialize the model
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3)

# Define the compute metrics function
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Create a small train and eval dataset for demonstration
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select([i for i in list(range(100))])
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select([i for i in list(range(100))])

# Initialize the fineTuner class and train the model
class fineTuner:
    def __init__(self, model, small_train_dataset, small_eval_dataset, compute_metrics):
        self.model = model
        self.small_train_dataset = small_train_dataset
        self.small_eval_dataset = small_eval_dataset
        self.compute_metrics = compute_metrics

        self.training_args = TrainingArguments(
            output_dir="test_trainer",
            #evaluation_strategy="epoch",
            per_device_train_batch_size=1,  # Reduce batch size here
            per_device_eval_batch_size=1,    # Optionally, reduce for evaluation as well
            gradient_accumulation_steps=4
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.small_train_dataset,
            eval_dataset=self.small_eval_dataset,
            compute_metrics=self.compute_metrics,
        )

    def train(self):
        self.trainer.train()

# Instantiate and train the model
finetuner = fineTuner(model, small_train_dataset, small_eval_dataset, compute_metrics)
finetuner.train()