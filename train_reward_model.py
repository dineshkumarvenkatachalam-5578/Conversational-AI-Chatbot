import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# --- Configuration ---
BASE_MODEL = 'distilbert-base-uncased'
FEEDBACK_LOG_FILE = 'logs/rl_training_data.csv'
OUTPUT_DIR = './models/reward_model'

def load_and_prepare_data(tokenizer):
    """Loads feedback data and prepares it for training a reward model."""
    if not os.path.exists(FEEDBACK_LOG_FILE):
        raise FileNotFoundError(f"Feedback log file not found: {FEEDBACK_LOG_FILE}")

    print(f"Loading feedback data from '{FEEDBACK_LOG_FILE}'...")
    df = pd.read_csv(FEEDBACK_LOG_FILE)

    # The reward model needs to predict a score based on the conversation state and the chosen action.
    # We'll format the input as: "State: [observation] Action: [action]"
    
    def create_input_text(row):
        try:
            # The observation is stored as a JSON string
            observation = json.loads(row['observation_json'])
            user_input = observation.get('user_input', '')
            # Simple representation of state for now
            return f"User Input: {user_input} | Chosen Reply: {row['action_taken']}"
        except (json.JSONDecodeError, TypeError):
            return None

    df['text'] = df.apply(create_input_text, axis=1)
    df = df.dropna(subset=['text'])
    
    # The label for a reward model is the continuous reward value.
    # We'll rename the 'reward' column to 'labels' for Hugging Face compatibility.
    df = df.rename(columns={'reward': 'labels'})
    
    print(f"Prepared {len(df)} samples for reward model training.")

    # Create Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def train_reward_model():
    """
    Fine-tunes a transformer model to act as a reward model.
    This model learns to predict how "good" a response is in a given context.
    """
    print("--- Starting Reward Model Training ---")

    # 1. Initialize Tokenizer and Model
    # For reward modeling, we are predicting a single continuous value (the reward),
    # so we set num_labels=1.
    print(f"Loading tokenizer and model for '{BASE_MODEL}'...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)

    # 2. Load and Process Data
    full_dataset = load_and_prepare_data(tokenizer)
    
    # Split dataset
    train_test_split_dataset = full_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split_dataset['train']
    eval_dataset = train_test_split_dataset['test']

    # 4. Set up Trainer
    # We use a lower learning rate and more epochs for regression tasks typically.
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-6,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mse", # Mean Squared Error for regression
    )

    # Define metrics for regression (MSE)
    from sklearn.metrics import mean_squared_error

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        mse = mean_squared_error(labels, logits)
        return {"mse": mse}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # 5. Train the Model
    print("Starting training for reward model...")
    trainer.train()
    print("Training complete.")

    # 6. Save the Model
    print(f"Saving best reward model to '{OUTPUT_DIR}'...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("--- Reward Model training complete. ---")

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        train_reward_model()
    except Exception as e:
        print(f"[Critical Error] An unexpected error occurred during reward model training:")
        import traceback
        traceback.print_exc()
