import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pickle
import logging
import math
import os
import json

from model import EnhancedLSTMForecast
from dataset import VADTimeSeriesDataset
from preprocessing import create_sequences_pytorch_with_momentum_features
from config import (
    MODEL_ASSETS_DIR, PYTORCH_MODEL_FILENAME, SCALERS_INPUT_PICKLE_FILENAME,
    SCALERS_OUTPUT_PICKLE_FILENAME, BACKEND_CONFIG_FILENAME, TIME_STEP_YEARS
)


def create_scalers_and_scale_data(X_train, y_train_diff, X_test=None, y_test_diff=None):
    num_input_features = X_train.shape[2]
    num_output_features = y_train_diff.shape[1]

    X_train_reshaped_for_scaling = X_train.reshape(-1, num_input_features)

    input_scalers = [StandardScaler() for _ in range(num_input_features)]
    X_train_scaled_reshaped = np.zeros_like(X_train_reshaped_for_scaling)
    for i in range(num_input_features):
        input_scalers[i].fit(X_train_reshaped_for_scaling[:, i:i + 1])
        X_train_scaled_reshaped[:, i:i + 1] = input_scalers[i].transform(X_train_reshaped_for_scaling[:, i:i + 1])
    X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape)

    output_scalers = [StandardScaler() for _ in range(num_output_features)]
    y_train_diff_scaled = np.zeros_like(y_train_diff)
    for i in range(num_output_features):
        output_scalers[i].fit(y_train_diff[:, i:i + 1])
        y_train_diff_scaled[:, i:i + 1] = output_scalers[i].transform(y_train_diff[:, i:i + 1])

    X_test_scaled, y_test_diff_scaled = None, None
    if X_test is not None and X_test.size > 0 and y_test_diff is not None and y_test_diff.size > 0:
        X_test_reshaped_for_scaling = X_test.reshape(-1, num_input_features)
        X_test_scaled_reshaped = np.zeros_like(X_test_reshaped_for_scaling)
        for i in range(num_input_features):
            X_test_scaled_reshaped[:, i:i + 1] = input_scalers[i].transform(X_test_reshaped_for_scaling[:, i:i + 1])
        X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape)

        y_test_diff_scaled = np.zeros_like(y_test_diff)
        for i in range(num_output_features):
            y_test_diff_scaled[:, i:i + 1] = output_scalers[i].transform(y_test_diff[:, i:i + 1])

    return X_train_scaled, y_train_diff_scaled, X_test_scaled, y_test_diff_scaled, input_scalers, output_scalers


def create_data_loaders(X_train_scaled, y_train_diff_scaled, X_test_scaled, y_test_diff_scaled, batch_size):
    train_dataset = VADTimeSeriesDataset(X_train_scaled, y_train_diff_scaled)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = None
    if X_test_scaled is not None and X_test_scaled.size > 0:
        test_dataset = VADTimeSeriesDataset(X_test_scaled, y_test_diff_scaled)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def adjust_hidden_size_for_attention(hidden_size):
    if hidden_size % 8 != 0:
        actual_hidden_size = (hidden_size // 8 + (1 if hidden_size % 8 != 0 else 0)) * 8
        logging.warning(
            f"Adjusting hidden_size from {hidden_size} to {actual_hidden_size} for MultiheadAttention compatibility.")
        return actual_hidden_size
    return hidden_size


def train_model(model, train_loader, test_loader, num_epochs, learning_rate, device):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        log_message = f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.8f}"

        if test_loader and len(test_loader) > 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X_val, batch_y_val in test_loader:
                    batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                    outputs_val = model(batch_X_val)
                    loss_val = criterion(outputs_val, batch_y_val)
                    val_loss += loss_val.item()

            avg_val_loss = val_loss / len(test_loader)
            log_message += f", Validation Loss: {avg_val_loss:.8f}"
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(MODEL_ASSETS_DIR, f"best_{PYTORCH_MODEL_FILENAME}"))
                logging.info(f"Best model saved at epoch {epoch + 1} with val_loss: {best_val_loss:.8f}")
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break
        else:
            scheduler.step(avg_epoch_loss)

        logging.info(log_message)

    if os.path.exists(os.path.join(MODEL_ASSETS_DIR, f"best_{PYTORCH_MODEL_FILENAME}")):
        logging.info(f"Loading best model weights for final evaluation.")
        model.load_state_dict(torch.load(os.path.join(MODEL_ASSETS_DIR, f"best_{PYTORCH_MODEL_FILENAME}")))


def evaluate_model(model, test_loader, y_test_actual, last_actuals_for_test_reconstruction,
                   words_test, years_X_test, output_scalers, device, num_examples_to_show):
    if test_loader is None or y_test_actual.size == 0:
        print("\nNo test data for evaluation.")
        return

    model.eval()
    all_preds_diff_scaled = []
    with torch.no_grad():
        for batch_X_test, _ in test_loader:
            batch_X_test = batch_X_test.to(device)
            outputs_test_scaled = model(batch_X_test)
            all_preds_diff_scaled.append(outputs_test_scaled.cpu().numpy())

    if not all_preds_diff_scaled:
        return

    predictions_diff_scaled_np = np.concatenate(all_preds_diff_scaled)
    predictions_diff_unscaled = np.zeros_like(predictions_diff_scaled_np)
    num_output_features = len(output_scalers)

    if predictions_diff_scaled_np.size > 0:
        for i in range(num_output_features):
            predictions_diff_unscaled[:, i] = output_scalers[i].inverse_transform(
                predictions_diff_scaled_np[:, i].reshape(-1, 1)).flatten()

    if last_actuals_for_test_reconstruction.shape[0] == predictions_diff_unscaled.shape[0]:
        predictions_reconstructed_actual_values = last_actuals_for_test_reconstruction + predictions_diff_unscaled
    else:
        logging.warning("Shape mismatch in reconstruction. Skipping reconstruction.")
        return

    print_evaluation_results(predictions_reconstructed_actual_values, y_test_actual,
                             words_test, years_X_test, num_examples_to_show)


def print_evaluation_results(predictions_reconstructed_actual_values, y_test_actual,
                             words_test, years_X_test, num_examples_to_show):
    print(f"\n--- Enhanced Model Evaluation Summary (PyTorch with Advanced Momentum Features) ---")

    if predictions_reconstructed_actual_values.size > 0 and y_test_actual.size == predictions_reconstructed_actual_values.size:
        overall_mae_unscaled = mean_absolute_error(y_test_actual, predictions_reconstructed_actual_values)
        overall_rmse_unscaled = math.sqrt(mean_squared_error(y_test_actual, predictions_reconstructed_actual_values))
        print(f"Test MAE (on reconstructed unscaled data, overall): {overall_mae_unscaled:.8f}")
        print(f"Test RMSE (on reconstructed unscaled data, overall): {overall_rmse_unscaled:.8f}")

        for i, dim_label in enumerate(['V', 'A', 'D']):
            dim_mae = mean_absolute_error(y_test_actual[:, i], predictions_reconstructed_actual_values[:, i])
            dim_rmse = math.sqrt(mean_squared_error(y_test_actual[:, i], predictions_reconstructed_actual_values[:, i]))
            print(f"  {dim_label} - MAE (unscaled): {dim_mae:.8f}, RMSE (unscaled): {dim_rmse:.8f}")

        print(f"\n--- Examples of Predicted vs. Actual VAD values (Test Set, Reconstructed Unscaled) ---")
        num_to_display = min(num_examples_to_show, len(predictions_reconstructed_actual_values))
        if num_to_display == 0:
            print("No test examples to display.")
        for i in range(num_to_display):
            pred_v, pred_a, pred_d = predictions_reconstructed_actual_values[i]
            actual_v, actual_a, actual_d = y_test_actual[i]
            current_word = words_test[i]
            year_of_prediction = years_X_test[i] + TIME_STEP_YEARS
            print(f"Example {i + 1}: Word='{current_word}', Predicted Year={year_of_prediction}")
            print(f"  Predicted: V={pred_v:.4f}, A={pred_a:.4f}, D={pred_d:.4f}")
            print(f"  Actual:    V={actual_v:.4f}, A={actual_a:.4f}, D={actual_d:.4f}")
            print("-" * 20)


def save_model_and_assets(model, input_scalers, output_scalers, lookback_window,
                          num_input_features, num_output_features, dropout):
    os.makedirs(MODEL_ASSETS_DIR, exist_ok=True)
    model_save_path = os.path.join(MODEL_ASSETS_DIR, PYTORCH_MODEL_FILENAME)
    scalers_input_save_path = os.path.join(MODEL_ASSETS_DIR, SCALERS_INPUT_PICKLE_FILENAME)
    scalers_output_save_path = os.path.join(MODEL_ASSETS_DIR, SCALERS_OUTPUT_PICKLE_FILENAME)

    if os.path.exists(os.path.join(MODEL_ASSETS_DIR, f"best_{PYTORCH_MODEL_FILENAME}")):
        os.rename(os.path.join(MODEL_ASSETS_DIR, f"best_{PYTORCH_MODEL_FILENAME}"), model_save_path)
        logging.info(f"Best PyTorch model state_dict saved as: {model_save_path}")
    else:
        torch.save(model.state_dict(), model_save_path)
        logging.info(f"PyTorch model state_dict saved (last epoch) to: {model_save_path}")

    with open(scalers_input_save_path, 'wb') as f:
        pickle.dump(input_scalers, f)
    logging.info(f"Input Scalers (for {num_input_features} features) saved to: {scalers_input_save_path}")

    with open(scalers_output_save_path, 'wb') as f:
        pickle.dump(output_scalers, f)
    logging.info(f"Output Scalers (for {num_output_features} features) saved to: {scalers_output_save_path}")

    backend_config = {
        'lookback_window': lookback_window,
        'num_input_features': num_input_features,
        'num_output_features': num_output_features,
        'model_type': 'pytorch_enhanced_momentum',
        'model_params': {
            'input_size': num_input_features,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'output_size': num_output_features,
            'dropout_prob': dropout
        },
        'input_scalers_params': [{'mean': float(s.mean_[0]), 'scale': float(s.scale_[0])} for s in input_scalers],
        'output_scalers_params': [{'mean': float(s.mean_[0]), 'scale': float(s.scale_[0])} for s in output_scalers]
    }

    with open(BACKEND_CONFIG_FILENAME, 'w') as f:
        json.dump(backend_config, f, indent=4)
    logging.info(f"Backend model config saved to: {BACKEND_CONFIG_FILENAME}")


def train_and_evaluate_pytorch_lstm(ml_ready_data_path, lookback_param, forecast_horizon_param,
                                    train_until_year_param, hidden_size=128, num_lstm_layers=2,
                                    dropout=0.1, learning_rate=0.001, num_epochs=50,
                                    batch_size=64, num_examples_to_show=5):
    data_results = create_sequences_pytorch_with_momentum_features(
        ml_ready_data_path, lookback_param, forecast_horizon_param, train_until_year_param
    )

    (X_train, y_train_diff, X_test, y_test_diff, y_test_actual,
     last_actuals_for_test_reconstruction, words_test, years_X_test,
     lookback_window, num_input_features) = data_results

    num_output_features = 3

    if X_train.size == 0:
        logging.error("Training data is empty. Cannot train model.")
        return None, None, None

    X_train_scaled, y_train_diff_scaled, X_test_scaled, y_test_diff_scaled, input_scalers, output_scalers = create_scalers_and_scale_data(
        X_train, y_train_diff, X_test, y_test_diff
    )

    train_loader, test_loader = create_data_loaders(
        X_train_scaled, y_train_diff_scaled, X_test_scaled, y_test_diff_scaled, batch_size
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    actual_hidden_size = adjust_hidden_size_for_attention(hidden_size)

    model = EnhancedLSTMForecast(num_input_features, actual_hidden_size, num_lstm_layers,
                                 num_output_features, dropout_prob=dropout).to(device)

    print("Enhanced PyTorch Model Architecture:")
    print(model)

    train_model(model, train_loader, test_loader, num_epochs, learning_rate, device)

    evaluate_model(model, test_loader, y_test_actual, last_actuals_for_test_reconstruction,
                   words_test, years_X_test, output_scalers, device, num_examples_to_show)

    save_model_and_assets(model, input_scalers, output_scalers, lookback_window,
                          num_input_features, num_output_features, dropout)

    return model, input_scalers, output_scalers
