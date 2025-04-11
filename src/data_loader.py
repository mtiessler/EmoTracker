import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from typing import Tuple
from utils import log_info, log_error

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


def load_and_preprocess_data(file_path: str, tokenizer_name: str, max_len: int, batch_size: int) -> Tuple[
    DataLoader, DataLoader, DataLoader]:
    log_info(f"Loading and preprocessing data from {file_path}")
    df = pd.read_csv(file_path)

    # Log initial size
    log_info(f"Initial dataset size: {len(df)} rows")

    df['Word'] = df['Word'].astype(str)

    # Filtering atypical words
    initial_count = len(df)
    removed_count = 0

    for i in df.index:
        word = df.loc[i, 'Word']
        try:
            if ((" " in word) or any(x.isupper() for x in word) or "'" in word or word == 'nan'):
                df.drop(i, inplace=True)
                removed_count += 1
        except Exception as e:
            log_error(f"Error processing word '{word}': {e}")
            df.drop(i, inplace=True)
            removed_count += 1

    # Log filtering results
    log_info(f"Removed {removed_count} rows during filtering")
    log_info(f"Final dataset size: {len(df)} rows")

    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)

    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Log split sizes
    log_info(f"Train size: {len(train_df)} rows")
    log_info(f"Validation size: {len(val_df)} rows")
    log_info(f"Test size: {len(test_df)} rows")

    # Prepare 3D labels
    train_labels = train_df[['V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']].values
    val_labels = val_df[['V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']].values
    test_labels = test_df[['V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']].values

    # Create datasets
    train_dataset = VADDataset(train_df['Word'].values, train_labels, tokenizer, max_len)
    val_dataset = VADDataset(val_df['Word'].values, val_labels, tokenizer, max_len)
    test_dataset = VADDataset(test_df['Word'].values, test_labels, tokenizer, max_len)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    log_info("Data loaded and preprocessed.")
    return train_dataloader, val_dataloader, test_dataloader