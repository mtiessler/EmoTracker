import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import Tuple, List, Dict
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from utils import log_info
from model import VADRobertaModel


def initialize_model_optimizer_scheduler(model_name: str,
                                         learning_rate: float,
                                         total_steps: int,
                                         dropout_rate: float,
                                         warmup_steps: int) -> Tuple[
    VADRobertaModel, AdamW, torch.optim.lr_scheduler.LambdaLR, torch.device]:
    log_info("Initializing model, optimizer, and scheduler.")
    model = VADRobertaModel(model_name, dropout_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    log_info("Model, optimizer, and scheduler initialized.")
    return model, optimizer, scheduler, device


def train_and_validate(model: VADRobertaModel,
                       train_dataloader: torch.utils.data.DataLoader,
                       val_dataloader: torch.utils.data.DataLoader,
                       optimizer: AdamW,
                       scheduler: torch.optim.lr_scheduler.LambdaLR,
                       device: torch.device,
                       epochs: int) -> Tuple[List[float], List[float], Dict[str, List[float]], Dict[str, List[float]]]:
    log_info("Starting training and validation.")
    train_losses = []
    val_losses = []
    r2_scores = {'valence': [], 'arousal': [], 'dominance': []}
    mse_scores = {'valence': [], 'arousal': [], 'dominance': []}

    for epoch in range(epochs):
        log_info(f'Epoch {epoch + 1}/{epochs}')
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        log_info(f'Training Loss: {avg_train_loss:.4f}')

        # Validation
        val_metrics = evaluate_model(model, val_dataloader, device)
        val_losses.append(val_metrics['loss'])

        for dim in ['valence', 'arousal', 'dominance']:
            r2_scores[dim].append(val_metrics['r2'][dim])
            mse_scores[dim].append(val_metrics['mse'][dim])

        log_info(f"Validation - Loss: {val_metrics['loss']:.4f}")
        log_info(f"Valence - MSE: {val_metrics['mse']['valence']:.4f}, R2: {val_metrics['r2']['valence']:.4f}")
        log_info(f"Arousal - MSE: {val_metrics['mse']['arousal']:.4f}, R2: {val_metrics['r2']['arousal']:.4f}")
        log_info(f"Dominance - MSE: {val_metrics['mse']['dominance']:.4f}, R2: {val_metrics['r2']['dominance']:.4f}")

    return train_losses, val_losses, r2_scores, mse_scores


def evaluate_model(model: VADRobertaModel,
                   dataloader: torch.utils.data.DataLoader,
                   device: torch.device) -> Dict:
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            total_loss += loss.item()

            predictions = outputs['logits'].cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics for each dimension
    metrics = {
        'loss': avg_loss,
        'mse': {
            'valence': mean_squared_error(all_labels[:, 0], all_predictions[:, 0]),
            'arousal': mean_squared_error(all_labels[:, 1], all_predictions[:, 1]),
            'dominance': mean_squared_error(all_labels[:, 2], all_predictions[:, 2])
        },
        'r2': {
            'valence': r2_score(all_labels[:, 0], all_predictions[:, 0]),
            'arousal': r2_score(all_labels[:, 1], all_predictions[:, 1]),
            'dominance': r2_score(all_labels[:, 2], all_predictions[:, 2])
        }
    }

    return metrics