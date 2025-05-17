import json
import numpy as np
import tensorflow as tf
import pickle
import logging

ML_READY_DATA_PATH = '..\..\data\Generated_VAD_Dataset\ml_ready_temporal_vad_data.json'
MODEL_PATH = 'lstm_vad_model.keras'
SCALERS_PATH = 'vad_scalers.pkl'

LOOKBACK = 4
NUM_FEATURES = 3
TIME_STEP_UNIFIED = 5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_model_and_scalers(model_path, scalers_path):
    try:
        model = tf.keras.models.load_model(model_path)
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        logging.info(f"Model loaded successfully from {model_path}")
        logging.info(f"Scalers loaded successfully from {scalers_path}")
        return model, scalers
    except FileNotFoundError:
        logging.error(f"Error: Model or scalers file not found. Searched for '{model_path}' and '{scalers_path}'.")
        return None, None
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading model/scalers: {e}")
        return None, None


def get_word_latest_valid_sequence(word_name, ml_data_path, lookback_window):
    try:
        with open(ml_data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: ML data file not found at {ml_data_path}")
        return None, None, None
    except Exception as e:
        logging.error(f"Error reading ML data file {ml_data_path}: {e}")
        return None, None, None

    if word_name not in all_data:
        logging.error(f"Word '{word_name}' not found in the dataset.")
        return None, None, None

    word_data = all_data[word_name]
    if 'temporal_vad' not in word_data:
        logging.error(f"Word '{word_name}' does not have 'temporal_vad' data.")
        return None, None, None

    tv_data = word_data['temporal_vad']
    if not ('x' in tv_data and 'v' in tv_data and 'a' in tv_data and 'd' in tv_data):
        logging.error(f"Word '{word_name}' 'temporal_vad' is missing x, v, a, or d keys.")
        return None, None, None

    x_hist_all = tv_data['x']
    v_hist_all = tv_data['v']
    a_hist_all = tv_data['a']
    d_hist_all = tv_data['d']

    if not (len(x_hist_all) == len(v_hist_all) == len(a_hist_all) == len(d_hist_all)):
        logging.error(f"Word '{word_name}': Mismatch in lengths of historical x,v,a,d lists.")
        return None, None, None

    historical_vad_tuples = []
    for i in range(len(x_hist_all)):
        v, a, d = v_hist_all[i], a_hist_all[i], d_hist_all[i]
        if v is None or a is None or d is None or np.isnan(v) or np.isnan(a) or np.isnan(d):
            historical_vad_tuples.append(None)
        else:
            historical_vad_tuples.append([float(v), float(a), float(d)])

    last_valid_sequence_end_idx = -1
    for i in range(len(historical_vad_tuples) - 1, lookback_window - 2, -1):
        is_valid_segment = True
        for j in range(lookback_window):
            if historical_vad_tuples[i - j] is None:
                is_valid_segment = False
                break
        if is_valid_segment:
            last_valid_sequence_end_idx = i
            break

    if last_valid_sequence_end_idx == -1 or last_valid_sequence_end_idx < lookback_window - 1:
        logging.error(
            f"Word '{word_name}' does not have enough recent non-NaN data (at least {lookback_window} points) to start forecasting.")
        return None, None, None

    start_idx = last_valid_sequence_end_idx - lookback_window + 1
    initial_sequence_unscaled = np.array(historical_vad_tuples[start_idx: last_valid_sequence_end_idx + 1])
    last_known_year = int(x_hist_all[last_valid_sequence_end_idx])

    logging.info(
        f"Using data from year {x_hist_all[start_idx]} to {last_known_year} for word '{word_name}' as initial sequence.")
    return initial_sequence_unscaled, last_known_year, x_hist_all


def scale_data(data_unscaled, scalers, num_features):
    if data_unscaled.ndim == 1:
        data_unscaled = data_unscaled.reshape(1, -1)

    data_scaled = np.zeros_like(data_unscaled, dtype=float)
    for i in range(num_features):
        data_scaled[:, i] = scalers[i].transform(data_unscaled[:, i].reshape(-1, 1)).flatten()
    return data_scaled


def inverse_scale_data(data_scaled, scalers, num_features):
    if data_scaled.ndim == 1:
        data_scaled = data_scaled.reshape(1, -1)

    data_unscaled = np.zeros_like(data_scaled, dtype=float)
    for i in range(num_features):
        data_unscaled[:, i] = scalers[i].inverse_transform(data_scaled[:, i].reshape(-1, 1)).flatten()
    return data_unscaled


def forecast_vad_autoregressive(model, initial_sequence_unscaled, scalers, target_forecast_year, last_known_year,
                                time_step, lookback_window, num_features):
    if initial_sequence_unscaled.shape[0] != lookback_window:
        logging.error(
            f"Initial sequence length ({initial_sequence_unscaled.shape[0]}) does not match lookback window ({lookback_window}).")
        return [], []

    current_sequence_scaled = scale_data(initial_sequence_unscaled, scalers, num_features)

    forecasted_vad_unscaled_list = []
    forecasted_years = []

    current_year = last_known_year

    while current_year < target_forecast_year:
        input_for_model = current_sequence_scaled.reshape(1, lookback_window, num_features)
        predicted_step_scaled = model.predict(input_for_model, verbose=0)[0]
        predicted_step_unscaled = inverse_scale_data(predicted_step_scaled, scalers, num_features)[0]

        current_year += time_step
        forecasted_years.append(current_year)
        forecasted_vad_unscaled_list.append(predicted_step_unscaled.tolist())

        new_step_scaled_reshaped = predicted_step_scaled.reshape(1, num_features)
        current_sequence_scaled = np.vstack((current_sequence_scaled[1:], new_step_scaled_reshaped))

        if current_year >= target_forecast_year:
            break

    return forecasted_years, forecasted_vad_unscaled_list


def main_forecast(word_to_forecast, target_year_to_forecast):
    logging.info(f"--- Starting VAD Forecast for word: '{word_to_forecast}' up to year: {target_year_to_forecast} ---")

    model, scalers = load_model_and_scalers(MODEL_PATH, SCALERS_PATH)
    if model is None or scalers is None:
        return

    initial_sequence_unscaled, last_known_year, all_historical_years = get_word_latest_valid_sequence(
        word_to_forecast,
        ML_READY_DATA_PATH,
        LOOKBACK
    )

    if initial_sequence_unscaled is None or last_known_year is None:
        logging.error(f"Could not retrieve valid initial sequence for '{word_to_forecast}'. Aborting forecast.")
        return

    if target_year_to_forecast <= last_known_year:
        logging.warning(
            f"Target year {target_year_to_forecast} is not in the future of the last known data point ({last_known_year}) for '{word_to_forecast}'.")
        logging.info("Displaying historical data instead if available up to the target year.")

        try:
            with open(ML_READY_DATA_PATH, 'r', encoding='utf-8') as f:
                all_ml_data = json.load(f)
            word_hist_data = all_ml_data[word_to_forecast]['temporal_vad']
            print(f"\nHistorical VAD for '{word_to_forecast}':")
            for i, year in enumerate(word_hist_data['x']):
                if year <= target_year_to_forecast:
                    print(
                        f"  Year: {year}, V: {word_hist_data['v'][i]:.4f}, A: {word_hist_data['a'][i]:.4f}, D: {word_hist_data['d'][i]:.4f}")
                if year >= target_year_to_forecast and year > last_known_year:
                    break
        except Exception as e:
            logging.error(f"Could not display historical data: {e}")
        return

    logging.info(f"Last known year for '{word_to_forecast}': {last_known_year}")
    logging.info(f"Initial VAD sequence (unscaled) for forecasting:\n{initial_sequence_unscaled}")

    forecasted_years, forecasted_vad_values = forecast_vad_autoregressive(
        model,
        initial_sequence_unscaled,
        scalers,
        target_year_to_forecast,
        last_known_year,
        TIME_STEP_UNIFIED,
        LOOKBACK,
        NUM_FEATURES
    )

    if forecasted_years:
        print(f"\n--- Forecasted VAD Trajectory for '{word_to_forecast}' ---")
        for i in range(len(forecasted_years)):
            year = forecasted_years[i]
            vad = forecasted_vad_values[i]
            print(f"  Year: {year}, V: {vad[0]:.4f}, A: {vad[1]:.4f}, D: {vad[2]:.4f}")
    else:
        print(f"\nNo forecast generated for '{word_to_forecast}' up to {target_year_to_forecast}.")


if __name__ == "__main__":
    word_input = input("Enter the word to forecast (e.g., explore, technology): ").strip().lower()

    example_future_year_prompt = 2025 + TIME_STEP_UNIFIED * (LOOKBACK + 5)

    while True:
        try:
            year_input_str = input(
                f"Enter the target future year for forecasting (e.g., {example_future_year_prompt}): ").strip()
            year_input = int(year_input_str)
            if year_input < 1000 or year_input > 3000:
                print("Please enter a valid year (e.g., between 1000 and 3000).")
            else:
                break
        except ValueError:
            print("Invalid year format. Please enter a number (e.g., 2030).")

    if word_input and year_input:
        main_forecast(word_input, year_input)
    else:
        print("Word or year not provided. Exiting.")