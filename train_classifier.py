import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import os

# --- Configuration ---
BASE_MODEL = 'distilbert-base-uncased'
OUTPUT_DIR_BASE = './models'
CSV_FILES = {
    'intent': 'intent_examples.csv',
    'topic': 'topic_examples.csv'
}
TEXT_COLUMN = 'example'
LABEL_COLUMNS = {
    'intent': 'intent',
    'topic': 'topic'
}

def create_label_mappings(df, label_column):
    """Creates mappings from label names to integer IDs and back."""
    labels = df[label_column].unique().tolist()
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    return label2id, id2label

def preprocess_data(df, label_column, tokenizer, label2id):
    """Converts a pandas DataFrame to a Hugging Face Dataset and tokenizes it."""
    # Rename label column to 'labels' for Hugging Face compatibility
    df['labels'] = df[label_column].map(label2id)
    
    # Drop rows where label mapping resulted in NaN (if any)
    df = df.dropna(subset=['labels'])
    df['labels'] = df['labels'].astype(int)

    # Create Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        return tokenizer(examples[TEXT_COLUMN], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def train_classifier(task: str):
    """
    Fine-tunes a transformer model for the specified classification task (intent or topic).

    Args:
        task (str): The task to train for. Must be 'intent' or 'topic'.
    """
    if task not in CSV_FILES:
        print(f"[Error] Invalid task '{task}'. Choose from {list(CSV_FILES.keys())}")
        return

    print(f"--- Starting fine-tuning process for '{task}' classification ---")

    # 1. Load and Prepare Data
    csv_path = CSV_FILES[task]
    label_column = LABEL_COLUMNS[task]
    if not os.path.exists(csv_path):
        print(f"[Error] Data file not found: {csv_path}")
        return
        
    print(f"Loading data from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[TEXT_COLUMN, label_column])
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)


    label2id, id2label = create_label_mappings(df, label_column)
    num_labels = len(label2id)
    print(f"Found {num_labels} unique labels for '{task}'.")

    # 2. Initialize Tokenizer and Model
    print(f"Loading tokenizer and model for '{BASE_MODEL}'...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # 3. Create Datasets (Train/Validation Split)
    print("Creating train and validation datasets...")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label_column])
    
    train_dataset = preprocess_data(train_df, label_column, tokenizer, label2id)
    val_dataset = preprocess_data(val_df, label_column, tokenizer, label2id)

    # 4. Set up Trainer
    output_dir = os.path.join(OUTPUT_DIR_BASE, f'{task}_classifier')
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Define metrics
    from datasets import load_metric
    import numpy as np
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 5. Train the Model
    print(f"Starting training for '{task}' model...")
    trainer.train()
    print("Training complete.")

    # 6. Save the Model and Tokenizer
    print(f"Saving best model and tokenizer to '{output_dir}'...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"--- Fine-tuning for '{task}' complete. Model saved successfully. ---")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune a classifier for intent or topic detection.")
    parser.add_argument('task', type=str, choices=['intent', 'topic'], help="The classification task to run ('intent' or 'topic').")
    args = parser.parse_args()
    
    # Ensure models directory exists
    os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    
    try:
        train_classifier(args.task)
    except Exception as e:
        print(f"[Critical Error] An unexpected error occurred during training for task '{args.task}':")
        import traceback
        traceback.print_exc()
