import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import logging
import math
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PYTORCH_MODEL_FILENAME = 'lstm_vad_model_pytorch.pth'
SCALERS_PICKLE_FILENAME = 'vad_scalers_pytorch.pkl'
MODEL_ASSETS_DIR = os.path.join('..','..','model_assets_pytorch')
BACKEND_CONFIG_FILENAME = os.path.join(MODEL_ASSETS_DIR, 'backend_model_config_pytorch.json')


def create_sequences_pytorch(ml_ready_data_path, lookback_window, forecast_horizon, train_until_year=None):
    try:
        with open(ml_ready_data_path, 'r', encoding='utf-8') as f:
            processed_data_unified = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading JSON data: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([]), 0
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading {ml_ready_data_path}: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([]), 0

    all_X, all_y, all_years_for_X = [], [], []
    global_timeline = []
    for word_data in processed_data_unified.values():
        if "temporal_vad" in word_data and "x" in word_data["temporal_vad"] and word_data["temporal_vad"]["x"]:
            global_timeline = word_data["temporal_vad"]["x"]
            break
    if not global_timeline:
        logging.error("Could not determine a global timeline.")
        return np.array([]), np.array([]), np.array([]), np.array([]), lookback_window

    for word, data in processed_data_unified.items():
        temp_vad = data.get("temporal_vad", {})
        if not all(k in temp_vad for k in ('x', 'v', 'a', 'd')):
            logging.debug(f"Word '{word}' missing essential keys. Skipping.")
            continue

        x_list = temp_vad['x']
        v_list = [np.nan if x is None else float(x) for x in temp_vad['v']]
        a_list = [np.nan if x is None else float(x) for x in temp_vad['a']]
        d_list = [np.nan if x is None else float(x) for x in temp_vad['d']]

        if not (len(x_list) == len(v_list) == len(a_list) == len(d_list)):
            logging.debug(f"Word '{word}': Mismatch in lengths. Skipping.")
            continue
        if not x_list:
            logging.debug(f"Word '{word}': x_list is empty. Skipping.")
            continue

        vad_series = np.array(list(zip(v_list, a_list, d_list)), dtype=float)
        if vad_series.shape[0] < lookback_window + forecast_horizon:
            logging.debug(f"Word '{word}': Not enough data. Skipping.")
            continue

        for i in range(len(vad_series) - lookback_window - forecast_horizon + 1):
            input_seq = vad_series[i: i + lookback_window]
            target = vad_series[i + lookback_window]

            if np.isnan(input_seq).any() or np.isnan(target).any():
                continue

            all_X.append(input_seq)
            all_y.append(target)
            year_index = i + lookback_window - 1
            year = x_list[year_index] if year_index < len(x_list) else -1
            all_years_for_X.append(year)

    if not all_X:
        logging.info("No valid sequences generated.")
        return np.array([]), np.array([]), np.array([]), np.array([]), lookback_window

    X_data = np.array(all_X, dtype=np.float32)
    y_data = np.array(all_y, dtype=np.float32)
    years_of_X_data = np.array(all_years_for_X)

    valid_indices = np.where(years_of_X_data != -1)[0]
    X_data, y_data, years_of_X_data = X_data[valid_indices], y_data[valid_indices], years_of_X_data[valid_indices]

    if X_data.size == 0:
        logging.info("No valid sequences after filtering.")
        return np.array([]), np.array([]), np.array([]), np.array([]), lookback_window

    if train_until_year is not None and years_of_X_data.size > 0:
        train_idx = np.where(years_of_X_data <= train_until_year)[0]
        test_idx = np.where(years_of_X_data > train_until_year)[0]
        X_train, y_train = X_data[train_idx], y_data[train_idx]
        X_test, y_test = X_data[test_idx], y_data[test_idx]
    else:
        X_train, y_train = X_data, y_data
        X_test, y_test = np.array([]), np.array([])

    return X_train, y_train, X_test, y_test, lookback_window


class VADTimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.0):
        super(LSTMForecast, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout_prob if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out


def train_and_evaluate_pytorch_lstm(
        ml_ready_data_path, lookback_param, forecast_horizon_param, train_until_year_param,
        num_features=3, hidden_size=50, num_lstm_layers=1, dropout=0.0,
        learning_rate=0.001, num_epochs=20, batch_size=32, num_examples_to_show=5
):
    X_train, y_train, X_test, y_test, lookback_window = create_sequences_pytorch(
        ml_ready_data_path, lookback_param, forecast_horizon_param, train_until_year_param
    )

    if X_train.size == 0:
        logging.error("Training data is empty. Cannot train model.")
        return None, None

    X_train_reshaped_for_scaling = X_train.reshape(-1, num_features)

    scalers = [StandardScaler() for _ in range(num_features)]
    X_train_scaled_reshaped = np.zeros_like(X_train_reshaped_for_scaling)

    for i in range(num_features):
        scalers[i].fit(X_train_reshaped_for_scaling[:, i:i + 1])
        X_train_scaled_reshaped[:, i:i + 1] = scalers[i].transform(X_train_reshaped_for_scaling[:, i:i + 1])

    X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape)

    y_train_scaled = np.zeros_like(y_train)
    for i in range(num_features):
        y_train_scaled[:, i:i + 1] = scalers[i].transform(y_train[:, i:i + 1])

    train_dataset = VADTimeSeriesDataset(X_train_scaled, y_train_scaled)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = None
    y_test_scaled_np_for_mse = None  # For calculating scaled MSE later

    if X_test.size > 0 and y_test.size > 0:
        X_test_reshaped_for_scaling = X_test.reshape(-1, num_features)
        X_test_scaled_reshaped = np.zeros_like(X_test_reshaped_for_scaling)
        for i in range(num_features):
            X_test_scaled_reshaped[:, i:i + 1] = scalers[i].transform(X_test_reshaped_for_scaling[:, i:i + 1])
        X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape)

        y_test_scaled = np.zeros_like(y_test)
        for i in range(num_features):
            y_test_scaled[:, i:i + 1] = scalers[i].transform(y_test[:, i:i + 1])

        y_test_scaled_np_for_mse = y_test_scaled  # Store for later MSE calculation
        test_dataset = VADTimeSeriesDataset(X_test_scaled, y_test_scaled)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    model = LSTMForecast(num_features, hidden_size, num_lstm_layers, num_features, dropout_prob=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("PyTorch Model Architecture:")
    print(model)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)

        log_message = f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.8f}"

        if test_loader:
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
        logging.info(log_message)

    if test_loader:
        model.eval()
        all_preds_scaled = []
        with torch.no_grad():
            for batch_X_test, _ in test_loader:  # No need for batch_y_test here, already have y_test_scaled_np_for_mse
                batch_X_test = batch_X_test.to(device)
                outputs_test_scaled = model(batch_X_test)
                all_preds_scaled.append(outputs_test_scaled.cpu().numpy())

        predictions_scaled_np = np.concatenate(all_preds_scaled)

        predictions_unscaled = np.zeros_like(predictions_scaled_np)

        for i in range(num_features):
            predictions_unscaled[:, i] = scalers[i].inverse_transform(
                predictions_scaled_np[:, i].reshape(-1, 1)).flatten()

        print(f"\n--- Model Evaluation Summary (PyTorch) ---")
        if y_test_scaled_np_for_mse is not None:
            test_mse_scaled = mean_squared_error(y_test_scaled_np_for_mse, predictions_scaled_np)
            print(f"Test MSE (on scaled data): {test_mse_scaled:.8f}")

        overall_mae_unscaled = mean_absolute_error(y_test, predictions_unscaled)
        overall_rmse_unscaled = math.sqrt(mean_squared_error(y_test, predictions_unscaled))
        print(f"Test MAE (on unscaled data, overall): {overall_mae_unscaled:.8f}")
        print(f"Test RMSE (on unscaled data, overall): {overall_rmse_unscaled:.8f}")

        for i, dim_label in enumerate(['V', 'A', 'D']):
            dim_mae = mean_absolute_error(y_test[:, i], predictions_unscaled[:, i])
            dim_rmse = math.sqrt(mean_squared_error(y_test[:, i], predictions_unscaled[:, i]))
            print(f"  {dim_label} - MAE (unscaled): {dim_mae:.8f}, RMSE (unscaled): {dim_rmse:.8f}")

        print(f"\n--- Examples of Predicted vs. Actual VAD values (Test Set, Unscaled) ---")
        num_to_display = min(num_examples_to_show, len(predictions_unscaled))
        if num_to_display == 0:
            print("No test examples to display.")
        for i in range(num_to_display):
            pred_v, pred_a, pred_d = predictions_unscaled[i]
            actual_v, actual_a, actual_d = y_test[i]
            print(f"Example {i + 1}:")
            print(f"  Predicted: V={pred_v:.4f}, A={pred_a:.4f}, D={pred_d:.4f}")
            print(f"  Actual:    V={actual_v:.4f}, A={actual_a:.4f}, D={actual_d:.4f}")
            print("-" * 20)
    else:
        print("\nNo test data for evaluation.")

    os.makedirs(MODEL_ASSETS_DIR, exist_ok=True)
    model_save_path = os.path.join(MODEL_ASSETS_DIR, PYTORCH_MODEL_FILENAME)
    scalers_save_path = os.path.join(MODEL_ASSETS_DIR, SCALERS_PICKLE_FILENAME)

    torch.save(model.state_dict(), model_save_path)
    logging.info(f"PyTorch model state_dict saved to: {model_save_path}")

    with open(scalers_save_path, 'wb') as f:
        pickle.dump(scalers, f)
    logging.info(f"Scalers saved to: {scalers_save_path}")

    backend_config = {
        'lookback_window': lookback_window,
        'num_features': num_features,
        'model_type': 'pytorch',
        'model_params': {  # Parameters needed to re-instantiate the model in backend
            'input_size': num_features,
            'hidden_size': hidden_size,
            'num_layers': num_lstm_layers,
            'output_size': num_features,
            'dropout_prob': dropout
        },
        'scalers_params': [{'mean': float(s.mean_[0]), 'scale': float(s.scale_[0])} for s in scalers]
    }

    with open(BACKEND_CONFIG_FILENAME, 'w') as f:
        json.dump(backend_config, f, indent=4)
    logging.info(f"Backend model config saved to: {BACKEND_CONFIG_FILENAME}")

    return model, scalers


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ml_ready_data_file = os.path.normpath(
        os.path.join(script_dir, '..', '..', 'data', 'Generated_VAD_Dataset', 'ml_ready_temporal_vad_data.json'))

    LOOKBACK_PARAM = 4
    FORECAST_HORIZON_PARAM = 1
    TRAIN_UNTIL_YEAR_PARAM = 1980
    NUM_EPOCHS = 1
    NUM_EXAMPLES_TO_SHOW = 5
    HIDDEN_SIZE_PARAM = 50
    NUM_LSTM_LAYERS_PARAM = 1
    DROPOUT_PARAM = 0.0  # Set to > 0 if num_lstm_layers > 1
    LEARNING_RATE_PARAM = 0.001

    if not os.path.exists(ml_ready_data_file):
        logging.error(f"Data file not found: {ml_ready_data_file}")
    else:
        trained_model, saved_scalers = train_and_evaluate_pytorch_lstm(
            ml_ready_data_path=ml_ready_data_file,
            lookback_param=LOOKBACK_PARAM,
            forecast_horizon_param=FORECAST_HORIZON_PARAM,
            train_until_year_param=TRAIN_UNTIL_YEAR_PARAM,
            hidden_size=HIDDEN_SIZE_PARAM,
            num_lstm_layers=NUM_LSTM_LAYERS_PARAM,
            dropout=DROPOUT_PARAM,
            learning_rate=LEARNING_RATE_PARAM,
            num_epochs=NUM_EPOCHS,
            num_examples_to_show=NUM_EXAMPLES_TO_SHOW
        )
        if trained_model:
            print("LSTM Training and asset saving complete.")
        else:
            print("LSTM Training failed.")