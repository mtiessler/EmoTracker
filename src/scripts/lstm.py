import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import StandardScaler
import pickle
import logging

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

        if not x_list:
            continue

        vad_series = np.array(list(zip(v_list, a_list, d_list)))

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
            if (i + lookback_window - 1) < len(current_x_timeline):
                all_years_for_X.append(current_x_timeline[i + lookback_window - 1])
            else:
                logging.warning(
                    f"Index out of bounds for year in word {word} for sequence ending at data index {i + lookback_window - 1}. Appending 0 for year.")
                all_years_for_X.append(0)

    if not all_X:
        return np.array([]), np.array([]), np.array([]), np.array([]), lookback_window

    X_data = np.array(all_X)
    y_data = np.array(all_y)
    years_of_X_data = np.array(all_years_for_X)

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
                            num_features=3):
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

    y_train_scaled = np.zeros_like(y_train)
    for i in range(num_features):
        y_train_scaled[:, i:i + 1] = scalers[i].transform(y_train[:, i:i + 1])

    if X_test.size > 0:
        X_test_reshaped = X_test.reshape(-1, num_features)
        X_test_scaled_reshaped = np.zeros_like(X_test_reshaped)
        for i in range(num_features):
            X_test_scaled_reshaped[:, i:i + 1] = scalers[i].transform(X_test_reshaped[:, i:i + 1])
        X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape)

        y_test_scaled = np.zeros_like(y_test)
        for i in range(num_features):
            y_test_scaled[:, i:i + 1] = scalers[i].transform(y_test[:, i:i + 1])
    else:
        X_test_scaled, y_test_scaled = np.array([]), np.array([])

    model = Sequential([
        Input(shape=(lookback_window_from_seq, num_features)),
        LSTM(50, activation='relu', return_sequences=False),
        Dense(num_features)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    if X_test_scaled.size > 0 and y_test_scaled.size > 0:
        test_loss = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
        print(f"Test MSE (LSTM): {test_loss}")
    else:
        print("\nNo test data to evaluate for LSTM.")

    model.save('lstm_vad_model.keras')
    with open('vad_scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)
    print("LSTM model saved as lstm_vad_model.keras and scalers as vad_scalers.pkl")
    return model, history


if __name__ == "__main__":
    ml_ready_data_file = '../../data/Generated_VAD_Dataset/ml_ready_temporal_vad_data.json'
    LOOKBACK_PARAM = 4
    FORECAST_HORIZON_PARAM = 1
    TRAIN_UNTIL_YEAR_PARAM = 1980

    trained_lstm_model, training_history = train_and_evaluate_lstm(
        ml_ready_data_file,
        LOOKBACK_PARAM,
        FORECAST_HORIZON_PARAM,
        TRAIN_UNTIL_YEAR_PARAM
    )
    if trained_lstm_model:
        print("LSTM model training and evaluation finished.")