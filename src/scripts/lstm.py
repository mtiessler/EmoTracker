import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import logging
import math
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

KERAS_MODEL_FILENAME = 'lstm_vad_model.keras'
SCALERS_PICKLE_FILENAME = 'vad_scalers.pkl'
MODEL_ASSETS_DIR = os.path.join('..', '..', 'data', 'model_assets')


def create_sequences_and_clean(ml_ready_data_path, lookback_window, forecast_horizon, train_until_year=None):
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
        if "temporal_vad" in word_data and \
                "x" in word_data["temporal_vad"] and \
                word_data["temporal_vad"]["x"]:
            global_timeline = word_data["temporal_vad"]["x"]
            break

    if not global_timeline:
        logging.error("Could not determine a global timeline from the data. Aborting sequence creation.")
        return np.array([]), np.array([]), np.array([]), np.array([]), lookback_window

    for word, data in processed_data_unified.items():
        temp_vad = data.get("temporal_vad", {})
        if not all(k in temp_vad for k in ('x', 'v', 'a', 'd')):
            logging.debug(f"Word '{word}' missing essential temporal_vad keys. Skipping.")
            continue

        x_list = temp_vad['x']
        v_list = [np.nan if x is None else float(x) for x in temp_vad['v']]
        a_list = [np.nan if x is None else float(x) for x in temp_vad['a']]
        d_list = [np.nan if x is None else float(x) for x in temp_vad['d']]

        if not (len(x_list) == len(v_list) == len(a_list) == len(d_list)):
            logging.debug(f"Word '{word}': Mismatch in lengths of x,v,a,d lists. Skipping.")
            continue

        if not x_list:
            logging.debug(f"Word '{word}': x_list is empty. Skipping.")
            continue

        vad_series = np.array(list(zip(v_list, a_list, d_list)), dtype=float)
        if vad_series.shape[0] < lookback_window + forecast_horizon:
            logging.debug(
                f"Word '{word}': Not enough data for lookback+horizon. Has {vad_series.shape[0]}, needs {lookback_window + forecast_horizon}. Skipping.")
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

    X_data = np.array(all_X)
    y_data = np.array(all_y)
    years_of_X_data = np.array(all_years_for_X)

    valid_indices = np.where(years_of_X_data != -1)[0]
    X_data, y_data, years_of_X_data = X_data[valid_indices], y_data[valid_indices], years_of_X_data[valid_indices]

    if X_data.size == 0:
        logging.info("No valid sequences after filtering placeholder years.")
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


def train_and_evaluate_lstm(ml_ready_data_path, lookback_param, forecast_horizon_param, train_until_year_param,
                            num_features=3, num_examples_to_show=5, epochs=1):
    X_train, y_train, X_test, y_test, lookback_window = create_sequences_and_clean(
        ml_ready_data_path, lookback_param, forecast_horizon_param, train_until_year_param)

    if X_train.size == 0:
        logging.error("Training data is empty. Cannot train model.")
        return None, None

    X_train_reshaped = X_train.reshape(-1, num_features)
    scalers = [StandardScaler() for _ in range(num_features)]
    X_train_scaled_reshaped = np.zeros_like(X_train_reshaped)

    for i in range(num_features):
        scalers[i].fit(X_train_reshaped[:, i:i + 1])
        X_train_scaled_reshaped[:, i:i + 1] = scalers[i].transform(X_train_reshaped[:, i:i + 1])

    X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape)

    y_train_scaled = np.zeros_like(y_train)
    for i in range(num_features):
        y_train_scaled[:, i:i + 1] = scalers[i].transform(y_train[:, i:i + 1])

    X_test_scaled, y_test_scaled = np.array([]), np.array([])
    if X_test.size > 0:
        X_test_reshaped = X_test.reshape(-1, num_features)
        X_test_scaled_reshaped = np.zeros_like(X_test_reshaped)
        for i in range(num_features):
            X_test_scaled_reshaped[:, i:i + 1] = scalers[i].transform(X_test_reshaped[:, i:i + 1])
        X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape)

        if y_test.size > 0:
            y_test_scaled = np.zeros_like(y_test)
            for i in range(num_features):
                y_test_scaled[:, i:i + 1] = scalers[i].transform(y_test[:, i:i + 1])
        else:
            y_test_scaled = np.array([]).reshape(0, num_features)

    model = Sequential([
        Input(shape=(lookback_window, num_features)),
        LSTM(50, activation='tanh'),
        Dense(num_features)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        shuffle=False,
        verbose=1
    )

    if X_test_scaled.size > 0 and y_test_scaled.size > 0:
        loss = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
        print(f"Test MSE (scaled): {loss:.6f}")

        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = np.zeros_like(y_pred_scaled)
        for i in range(num_features):
            y_pred[:, i] = scalers[i].inverse_transform(y_pred_scaled[:, i].reshape(-1, 1)).flatten()

        mae = mean_absolute_error(y_test, y_pred)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Test MAE (unscaled): {mae:.6f}, Test RMSE (unscaled): {rmse:.6f}")

        for i, label in enumerate(['V', 'A', 'D']):
            dim_mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            dim_rmse = math.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
            print(f"  {label}: MAE={dim_mae:.6f}, RMSE={dim_rmse:.6f}")

        print("\n--- Sample Predictions (Test Set, Unscaled) ---")
        for i in range(min(num_examples_to_show, len(y_pred))):
            print(
                f"Example {i + 1}:\n  Predicted: V={y_pred[i][0]:.4f}, A={y_pred[i][1]:.4f}, D={y_pred[i][2]:.4f}\n  "
                f"Actual:    V={y_test[i][0]:.4f}, A={y_test[i][1]:.4f}, D={y_test[i][2]:.4f}\n")
    else:
        print("No test data available for evaluation.")

    os.makedirs(MODEL_ASSETS_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_ASSETS_DIR, KERAS_MODEL_FILENAME)
    scalers_path = os.path.join(MODEL_ASSETS_DIR, SCALERS_PICKLE_FILENAME)

    model.save(model_path)
    print(f"Keras model saved to: {model_path}")

    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"Scalers saved to: {scalers_path}")

    backend_config = {
        'lookback_window': lookback_window,
        'num_features': num_features
    }
    backend_config_path = os.path.join(MODEL_ASSETS_DIR, 'backend_model_config.json')
    with open(backend_config_path, 'w') as f:
        json.dump(backend_config, f, indent=4)
    print(f"Backend model config saved to: {backend_config_path}")

    return model, scalers


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ml_ready_data_path = os.path.join(script_dir, '..', '..', 'data', 'Generated_VAD_Dataset',
                                      'ml_ready_temporal_vad_data.json')
    ml_ready_data_path = os.path.normpath(ml_ready_data_path)

    lookback = 4
    forecast_horizon = 1
    train_until_year = 1980
    num_epochs = 1

    if not os.path.exists(ml_ready_data_path):
        logging.error(f"Data file not found: {ml_ready_data_path}")
        return

    model, _ = train_and_evaluate_lstm(
        ml_ready_data_path=ml_ready_data_path,
        lookback_param=lookback,
        forecast_horizon_param=forecast_horizon,
        train_until_year_param=train_until_year,
        epochs=num_epochs
    )

    if model:
        print("LSTM Training and asset saving complete.")
    else:
        print("LSTM Training failed.")


if __name__ == "__main__":
    main()