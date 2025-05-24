import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import logging
import math
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_sequences_and_clean(ml_ready_data_path, lookback_window, forecast_horizon, train_until_year=None):
    try:
        with open(ml_ready_data_path, 'r', encoding='utf-8') as f:
            processed_data_unified = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {ml_ready_data_path}")
        return np.array([]), np.array([]), np.array([]), np.array([]), 0
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {ml_ready_data_path}")
        return np.array([]), np.array([]), np.array([]), np.array([]), 0
    except Exception as e:
        print(f"An unexpected error occurred while reading {ml_ready_data_path}: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([]), 0

    all_X, all_y, all_years_for_X = [], [], []

    global_timeline = []
    for word_key_for_timeline in processed_data_unified:
        if "temporal_vad" in processed_data_unified[word_key_for_timeline] and \
                "x" in processed_data_unified[word_key_for_timeline]["temporal_vad"]:
            global_timeline = processed_data_unified[word_key_for_timeline]["temporal_vad"]["x"]
            if global_timeline:
                break
    if not global_timeline:
        logging.error("Could not determine a global timeline from the data. Aborting sequence creation.")
        return np.array([]), np.array([]), np.array([]), np.array([]), lookback_window

    for word, data in processed_data_unified.items():
        if 'temporal_vad' not in data:
            logging.debug(f"Word '{word}' missing 'temporal_vad' key. Skipping.")
            continue

        temp_vad_data = data['temporal_vad']

        if not ('x' in temp_vad_data and 'v' in temp_vad_data and \
                'a' in temp_vad_data and 'd' in temp_vad_data):
            logging.debug(f"Word '{word}' 'temporal_vad' missing x,v,a, or d keys. Skipping.")
            continue

        x_list = temp_vad_data['x']
        v_list = temp_vad_data['v']
        a_list = temp_vad_data['a']
        d_list = temp_vad_data['d']

        if not (len(x_list) == len(v_list) == len(a_list) == len(d_list)):
            logging.debug(f"Word '{word}': Mismatch in lengths of x,v,a,d lists. Skipping.")
            continue

        if not x_list:  # If x_list is empty, skip
            continue

        v_list = [np.nan if x is None else x for x in v_list]
        a_list = [np.nan if x is None else x for x in a_list]
        d_list = [np.nan if x is None else x for x in d_list]

        vad_series = np.array(list(zip(v_list, a_list, d_list)), dtype=float)

        if vad_series.ndim == 1 and vad_series.size == 0:
            continue
        if vad_series.shape[0] < lookback_window + forecast_horizon:
            continue

        for i in range(len(vad_series) - lookback_window - forecast_horizon + 1):
            input_seq_slice = vad_series[i: i + lookback_window]
            target_val_slice = vad_series[i + lookback_window: i + lookback_window + forecast_horizon]

            if np.isnan(input_seq_slice).any():
                continue

            target_val = target_val_slice[0]
            if np.isnan(target_val).any():
                continue

            all_X.append(input_seq_slice)
            all_y.append(target_val)

            current_x_timeline = x_list
            year_index = i + lookback_window - 1
            if year_index < len(current_x_timeline):
                all_years_for_X.append(current_x_timeline[year_index])
            else:
                logging.warning(
                    f"Index out of bounds for year in word {word} for sequence ending at data "
                    f"index {year_index} (timeline length {len(current_x_timeline)}). Appending placeholder year -1.")
                all_years_for_X.append(-1)

    if not all_X:
        return np.array([]), np.array([]), np.array([]), np.array([]), lookback_window

    X_data = np.array(all_X)
    y_data = np.array(all_y)
    years_of_X_data = np.array(all_years_for_X)

    valid_year_indices = np.where(years_of_X_data != -1)[0]
    X_data = X_data[valid_year_indices]
    y_data = y_data[valid_year_indices]
    years_of_X_data = years_of_X_data[valid_year_indices]

    if X_data.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), lookback_window

    if train_until_year is not None and years_of_X_data.size > 0:
        train_indices = np.where(years_of_X_data <= train_until_year)[0]
        test_indices = np.where(years_of_X_data > train_until_year)[0]

        X_train, y_train = X_data[train_indices], y_data[train_indices]
        X_test, y_test = X_data[test_indices], y_data[test_indices]
    else:
        X_train, y_train = X_data, y_data
        X_test, y_test = np.array([]), np.array([])

    return X_train, y_train, X_test, y_test, lookback_window


def train_and_evaluate_lstm(ml_ready_data_path, lookback_param, forecast_horizon_param, train_until_year_param,
                            num_features=3, num_examples_to_show=5):
    X_train, y_train, X_test, y_test, lookback_window_from_seq = create_sequences_and_clean(
        ml_ready_data_path,
        lookback_param,
        forecast_horizon_param,
        train_until_year_param
    )

    if X_train.size == 0 or y_train.size == 0:
        print("Error: Training data is empty after sequence creation.")
        return None, None

    X_train_reshaped = X_train.reshape(-1, num_features)
    scalers = [StandardScaler() for _ in range(num_features)]
    X_train_scaled_reshaped = np.zeros_like(X_train_reshaped)

    for i in range(num_features):
        scalers[i].fit(X_train_reshaped[:, i:i + 1])
        X_train_scaled_reshaped[:, i:i + 1] = scalers[i].transform(X_train_reshaped[:, i:i + 1])
    X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape)

    # Scale y_train (targets)
    y_train_scaled = np.zeros_like(y_train)
    for i in range(num_features):
        y_train_scaled[:, i:i + 1] = scalers[i].transform(y_train[:, i:i + 1])

    X_test_scaled = np.array([])
    y_test_scaled = np.array([])
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

    model = Sequential([
        Input(shape=(lookback_window_from_seq, num_features)),
        LSTM(50, activation='tanh', return_sequences=False),
        Dense(num_features)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
        shuffle=False
    )

    if X_test_scaled.size > 0 and y_test_scaled.size > 0:
        test_loss_scaled = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
        print(f"\n--- Model Evaluation Summary ---")
        print(f"Test MSE (on scaled data): {test_loss_scaled:.8f}")

        predictions_scaled = model.predict(X_test_scaled)

        predictions_unscaled = np.zeros_like(predictions_scaled)
        for i in range(num_features):
            predictions_unscaled[:, i] = scalers[i].inverse_transform(predictions_scaled[:, i].reshape(-1, 1)).flatten()

        y_test_unscaled_for_metrics = y_test

        overall_mae_unscaled = mean_absolute_error(y_test_unscaled_for_metrics, predictions_unscaled)
        overall_rmse_unscaled = math.sqrt(mean_squared_error(y_test_unscaled_for_metrics, predictions_unscaled))
        print(f"Test MAE (on unscaled data, overall): {overall_mae_unscaled:.8f}")
        print(f"Test RMSE (on unscaled data, overall): {overall_rmse_unscaled:.8f}")

        for i, dim in enumerate(['V', 'A', 'D']):
            dim_mae = mean_absolute_error(y_test_unscaled_for_metrics[:, i], predictions_unscaled[:, i])
            dim_rmse = math.sqrt(mean_squared_error(y_test_unscaled_for_metrics[:, i], predictions_unscaled[:, i]))
            print(f"  {dim} - MAE (unscaled): {dim_mae:.8f}, RMSE (unscaled): {dim_rmse:.8f}")

        print(f"\n--- Examples of Predicted vs. Actual VAD values (Test Set, Unscaled) ---")
        num_to_display = min(num_examples_to_show, len(predictions_unscaled))
        if num_to_display == 0:
            print("No test examples to display.")
        for i in range(num_to_display):
            pred_v, pred_a, pred_d = predictions_unscaled[i]
            actual_v, actual_a, actual_d = y_test_unscaled_for_metrics[i]
            print(f"Example {i + 1}:")
            print(f"  Predicted: V={pred_v:.4f}, A={pred_a:.4f}, D={pred_d:.4f}")
            print(f"  Actual:    V={actual_v:.4f}, A={actual_a:.4f}, D={actual_d:.4f}")
            print("-" * 20)

    else:
        print("\nNo test data to evaluate or display examples for LSTM.")

    model.save('lstm_vad_model.keras')
    with open('vad_scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)
    print("LSTM model saved as lstm_vad_model.keras and scalers as vad_scalers.pkl")
    return model, history


if __name__ == "__main__":
    ml_ready_data_file = '../../data/Generated_VAD_Dataset/ml_ready_temporal_vad_data.json'
    LOOKBACK_PARAM = 4
    FORECAST_HORIZON_PARAM = 1  # Code is designed for 1-step ahead prediction
    TRAIN_UNTIL_YEAR_PARAM = 1980
    EXAMPLES_TO_SHOW = 5



    if not os.path.exists(ml_ready_data_file):
        print(f"ERROR: Data file not found at {ml_ready_data_file}")
        print("Please ensure the path is correct and the data generation script has been run successfully.")
    else:
        trained_lstm_model, training_history = train_and_evaluate_lstm(
            ml_ready_data_file,
            LOOKBACK_PARAM,
            FORECAST_HORIZON_PARAM,
            TRAIN_UNTIL_YEAR_PARAM,
            num_examples_to_show=EXAMPLES_TO_SHOW
        )
        if trained_lstm_model:
            print("LSTM model training and evaluation finished.")