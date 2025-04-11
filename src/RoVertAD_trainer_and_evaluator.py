import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import logging
from pathlib import Path
import itertools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(dotenv_path=Path("..") / ".env")
FILE_PATH = os.getenv('FILE_PATH', 'vad_lexicon.csv')
TOKENIZER_NAME = os.getenv('roberta-base')

def log_info(message):
    logging.info(message)

def log_warning(message):
    logging.warning(message)

def log_error(message):
    logging.error(message)

def load_and_preprocess_data(file_path, tokenizer_name, max_len):
    log_info(f"Loading and preprocessing data from {file_path}")
    df = pd.read_csv(file_path)
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)

    class VADDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, item):
            text = str(self.texts[item])
            label = self.labels[item]

            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )

            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.float)
            }

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_dataset = VADDataset(train_df['word'].values, train_df['valence'].values, tokenizer, max_len)
    val_dataset = VADDataset(val_df['word'].values, val_df['valence'].values, tokenizer, max_len)
    test_dataset = VADDataset(test_df['word'].values, test_df['valence'].values, tokenizer, max_len)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    log_info("Data loaded and preprocessed.")
    return train_dataloader, val_dataloader, test_dataloader

def initialize_model_optimizer_scheduler(model_name, num_labels, learning_rate, total_steps, dropout_rate, warmup_steps):
    log_info("Initializing model, optimizer, and scheduler.")
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, hidden_dropout_prob=dropout_rate, attention_probs_dropout_prob=dropout_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    log_info("Model, optimizer, and scheduler initialized.")
    return model, optimizer, scheduler, device

def train_and_validate(model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs):
    log_info("Starting training and validation.")
    train_losses = []
    val_losses = []
    r2_scores = []
    mse_scores = []

    for epoch in range(epochs):
        log_info(f'Epoch {epoch + 1}/{epochs}')
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).unsqueeze(1)

            model.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        log_info(f'Training Loss: {avg_train_loss:.4f}')

        model.eval()
        total_val_loss = 0
        predictions = []
        actual_labels = []

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device).unsqueeze(1)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()
                predictions.extend(outputs.logits.cpu().numpy().flatten())
                actual_labels.extend(labels.cpu().numpy().flatten())

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        log_info(f'Validation Loss: {avg_val_loss:.4f}')

        mse = mean_squared_error(actual_labels, predictions)
        r2 = r2_score(actual_labels, predictions)
        r2_scores.append(r2)
        mse_scores.append(mse)

    return train_losses, val_losses, r2_scores, mse_scores

def hyperparam_exploration():
    max_lens = [64, 128, 256]
    batch_sizes = [16, 32, 64]
    epochs_list = [3, 5, 7]
    learning_rates = [2e-5, 1e-5, 5e-6]
    dropout_rates = [0.1, 0.2, 0.3]
    warmup_steps_list = [0, 100, 500]

    best_r2 = -float('inf')
    best_params = {}
    results = []

    for max_len, batch_size, epochs, learning_rate, dropout_rate, warmup_steps in itertools.product(max_lens, batch_sizes, epochs_list, learning_rates, dropout_rates, warmup_steps_list):
        log_info(f"Trying: MAX_LEN={max_len}, BATCH_SIZE={batch_size}, EPOCHS={epochs}, LEARNING_RATE={learning_rate}, DROPOUT_RATE={dropout_rate}, WARMUP_STEPS={warmup_steps}")
        try:
            train_dataloader, val_dataloader, _ = load_and_preprocess_data(FILE_PATH, TOKENIZER_NAME, max_len)
            model, optimizer, scheduler, device = initialize_model_optimizer_scheduler(TOKENIZER_NAME, 1, learning_rate, len(train_dataloader) * epochs, dropout_rate, warmup_steps)
            train_losses, val_losses, r2_scores, mse_scores = train_and_validate(model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs)
            avg_r2 = sum(r2_scores) / len(r2_scores)
            results.append({
                'MAX_LEN': max_len,
                'BATCH_SIZE': batch_size,
                'EPOCHS': epochs,
                'LEARNING_RATE': learning_rate,
                'DROPOUT_RATE': dropout_rate,
                'WARMUP_STEPS': warmup_steps,
                'AVG_R2': avg_r2
            })
            if avg_r2 > best_r2:
                best_r2 = avg_r2
                best_params = {
                    'MAX_LEN': max_len,
                    'BATCH_SIZE': batch_size,
                    'EPOCHS': epochs,
                    'LEARNING_RATE': learning_rate,
                    'DROPOUT_RATE': dropout_rate,
                    'WARMUP_STEPS': warmup_steps
                }
        except Exception as e:
            log_error(f"Error during hyperparameter exploration: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv('hyperparam_exploration_results.csv', index=False)

    log_info(f"Best R2: {best_r2}")
    log_info(f"Best Parameters: {best_params}")
    return best_params

def evaluate_model(model, test_dataloader, device):
    log_info("Evaluating model on test set.")
    model.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).unsqueeze(1)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions.extend(outputs.logits.cpu().numpy().flatten())
            actual_labels.extend(labels.cpu().numpy().flatten())

    mse = mean_squared_error(actual_labels, predictions)
    r2 = r2_score(actual_labels, predictions)

    log_info(f"Mean Squared Error: {mse:.4f}")
    log_info(f"R-squared: {r2:.4f}")
    return mse, r2

def plot_losses(train_losses, val_losses, r2_scores, mse_scores):
    log_info("Plotting training and validation losses.")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('train_val_loss.png')
    plt.show()

    results_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train Loss': train_losses,
        'Validation Loss': val_losses,
        'R2 Score': r2_scores,
        'MSE Score': mse_scores
    })
    results_df.to_csv('training_results.csv', index=False)

def upload_model_to_huggingface(model, repo_name, tokenizer):
    log_info(f"Uploading model and tokenizer to Hugging Face Hub: {repo_name}")
    try:
        model.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)
        log_info(f"Model and tokenizer uploaded successfully.")
    except Exception as e:
        log_error(f"Failed to upload to Hugging Face Hub: {e}")

if __name__ == "__main__":
    best_params = hyperparam_exploration()
    MAX_LEN = best_params['MAX_LEN']
    BATCH_SIZE = best_params['BATCH_SIZE']
    EPOCHS = best_params['EPOCHS']
    LEARNING_RATE = best_params['LEARNING_RATE']
    DROPOUT_RATE = best_params['DROPOUT_RATE']
    WARMUP_STEPS = best_params['WARMUP_STEPS']

    train_dataloader, val_dataloader, test_dataloader = load_and_preprocess_data(FILE_PATH, TOKENIZER_NAME, MAX_LEN)
    model, optimizer, scheduler, device = initialize_model_optimizer_scheduler(TOKENIZER_NAME, 1, LEARNING_RATE, len(train_dataloader) * EPOCHS, DROPOUT_RATE, WARMUP_STEPS)
    train_losses, val_losses, r2_scores, mse_scores = train_and_validate(model, train_dataloader, val_dataloader, optimizer, scheduler, device, EPOCHS)
    mse, r2 = evaluate_model(model, test_dataloader, device)
    plot_losses(train_losses, val_losses, r2_scores, mse_scores)
    torch.save(model.state_dict(), 'finetuned_roberta_valence.pth')

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    REPO_NAME = os.getenv('mtiessler/RoVertAD')
    if REPO_NAME:
        upload_model_to_huggingface(model, REPO_NAME, tokenizer)