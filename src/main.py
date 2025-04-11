import logging
import torch
import itertools
import pandas as pd
from typing import Dict

from config import Config
from data_loader import load_and_preprocess_data
from trainer import initialize_model_optimizer_scheduler, train_and_validate, evaluate_model
from visualization import plot_losses
from utils import smoke_test, log_info, log_error


def hyperparam_exploration() -> Dict:
    config = Config()
    best_r2 = -float('inf')
    best_params = {}
    results = []

    for max_len, batch_size, epochs, learning_rate, dropout_rate, warmup_steps in itertools.product(
            config.MAX_LENS, config.BATCH_SIZES, config.EPOCHS_LIST,
            config.LEARNING_RATES, config.DROPOUT_RATES, config.WARMUP_STEPS_LIST
    ):
        log_info(f"Trying: MAX_LEN={max_len}, BATCH_SIZE={batch_size}, EPOCHS={epochs}, "
                 f"LEARNING_RATE={learning_rate}, DROPOUT_RATE={dropout_rate}, WARMUP_STEPS={warmup_steps}")
        try:
            train_dataloader, val_dataloader, _ = load_and_preprocess_data(
                config.FILE_PATH, config.TOKENIZER_NAME, max_len, batch_size
            )
            model, optimizer, scheduler, device = initialize_model_optimizer_scheduler(
                config.TOKENIZER_NAME,
                learning_rate,
                len(train_dataloader) * epochs,
                dropout_rate,
                warmup_steps
            )
            train_losses, val_losses, r2_scores, _ = train_and_validate(
                model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs
            )

            # Use average valence R2 as the metric for selection
            avg_r2 = sum(r2_scores['valence']) / len(r2_scores['valence'])
            results.append({
                'MAX_LEN': max_len,
                'BATCH_SIZE': batch_size,
                'EPOCHS': epochs,
                'LEARNING_RATE': learning_rate,
                'DROPOUT_RATE': dropout_rate,
                'WARMUP_STEPS': warmup_steps,
                'VALENCE_R2': avg_r2
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
    log_info(f"Best Valence R2: {best_r2}")
    log_info(f"Best Parameters: {best_params}")
    return best_params


def main():
    # First run smoke test
    if not smoke_test():
        logging.warning("Smoke test failed. Check your setup before proceeding.")
        return

    # Hyperparameter exploration
    best_params = hyperparam_exploration()

    # Update config with best parameters
    config = Config()
    for param, value in best_params.items():
        setattr(config, param, value)

    # Load data with best parameters
    train_dataloader, val_dataloader, test_dataloader = load_and_preprocess_data(
        config.FILE_PATH, config.TOKENIZER_NAME, config.MAX_LEN, config.BATCH_SIZE
    )

    # Initialize model with best parameters
    model, optimizer, scheduler, device = initialize_model_optimizer_scheduler(
        config.TOKENIZER_NAME,
        config.LEARNING_RATE,
        len(train_dataloader) * config.EPOCHS,
        config.DROPOUT_RATE,
        config.WARMUP_STEPS
    )

    # Train and validate
    train_losses, val_losses, r2_scores, mse_scores = train_and_validate(
        model, train_dataloader, val_dataloader, optimizer, scheduler, device, config.EPOCHS
    )

    # Evaluate on test set
    test_metrics = evaluate_model(model, test_dataloader, device)
    log_info(f"Test Results - Loss: {test_metrics['loss']:.4f}")
    for dim in ['valence', 'arousal', 'dominance']:
        log_info(f"{dim.capitalize()} - MSE: {test_metrics['mse'][dim]:.4f}, R2: {test_metrics['r2'][dim]:.4f}")

    # Plot results
    plot_losses(train_losses, val_losses, r2_scores, mse_scores)

    # Save model
    torch.save(model.state_dict(), 'finetuned_roberta_vad.pth')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()