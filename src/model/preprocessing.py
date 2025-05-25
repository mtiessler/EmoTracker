import json
import numpy as np
import logging
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter1d
from config import NUM_BASE_FEATURES, NUM_MOMENTUM_FEATURES_PER_DIM


def calculate_momentum_features(vad_series, window_size=5, sigma=1.0):
    n_points, n_dims = vad_series.shape
    momentum_features = []

    for i in range(n_points):
        if i < window_size:
            momentum_features.append(np.zeros(n_dims * 8))
            continue

        window_data = vad_series[i - window_size:i]
        feature_vector = []

        for dim in range(n_dims):
            series = window_data[:, dim]

            if len(series) < 3:
                feature_vector.extend([0.0] * 8)
                continue

            x_axis_reg = np.arange(len(series))

            try:
                slope, _, r_value, _, _ = linregress(x_axis_reg, series)
                velocity = slope
                trend_strength = abs(r_value)
                trend_direction = np.sign(slope)
            except ValueError:
                velocity = 0.0
                trend_strength = 0.0
                trend_direction = 0.0

            if np.isnan(velocity): velocity = 0.0
            if np.isnan(trend_strength): trend_strength = 0.0
            if np.isnan(trend_direction): trend_direction = 0.0

            smoothed_series = gaussian_filter1d(series, sigma=sigma, mode='nearest')

            acceleration = 0.0
            if len(smoothed_series) >= 3:
                velocities_diff = np.diff(smoothed_series)
                if len(velocities_diff) >= 2:
                    acceleration = np.mean(np.diff(velocities_diff))
            if np.isnan(acceleration): acceleration = 0.0

            volatility = np.std(series) if len(series) > 1 else 0.0
            if np.isnan(volatility): volatility = 0.0

            momentum_oscillator = 0.0
            if len(series) >= 2:
                recent_change = series[-1] - series[0]
                std_dev_series = np.std(series)
                momentum_oscillator = recent_change / (std_dev_series + 1e-8)
            if np.isnan(momentum_oscillator): momentum_oscillator = 0.0

            relative_strength = 0.0
            if len(series) >= 3:
                mid_point = len(series) // 2
                if mid_point > 0:
                    first_half_mean = np.mean(series[:mid_point])
                    second_half_mean = np.mean(series[mid_point:])
                    relative_strength = (second_half_mean - first_half_mean) / (abs(first_half_mean) + 1e-8)
            if np.isnan(relative_strength): relative_strength = 0.0

            range_position = 0.0
            if len(series) > 1:
                series_min, series_max = np.min(series), np.max(series)
                if series_max != series_min:
                    range_position = (series[-1] - series_min) / (series_max - series_min)
                else:
                    range_position = 0.5
            if np.isnan(range_position): range_position = 0.0

            ema_ratio = 0.0
            if len(series) >= 3:
                alpha = 2.0 / (len(series) + 1)
                ema = series[0]
                for val_idx in range(1, len(series)):
                    ema = alpha * series[val_idx] + (1 - alpha) * ema
                sma = np.mean(series)
                ema_ratio = (ema - sma) / (abs(sma) + 1e-8)
            if np.isnan(ema_ratio): ema_ratio = 0.0

            feature_vector.extend([
                velocity,
                acceleration,
                trend_strength * trend_direction,
                volatility,
                momentum_oscillator,
                relative_strength,
                range_position,
                ema_ratio
            ])

        momentum_features.append(np.array(feature_vector, dtype=float))

    return np.array(momentum_features, dtype=float)


def load_and_validate_data(ml_ready_data_path):
    try:
        with open(ml_ready_data_path, 'r', encoding='utf-8') as f:
            processed_data_unified = json.load(f)
        return processed_data_unified
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading JSON data: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading {ml_ready_data_path}: {e}")
        return None


def extract_global_timeline(processed_data_unified):
    global_timeline = []
    for word_data_val in processed_data_unified.values():
        if "temporal_vad" in word_data_val and "x" in word_data_val["temporal_vad"] and word_data_val["temporal_vad"][
            "x"]:
            global_timeline = word_data_val["temporal_vad"]["x"]
            break
    return global_timeline


def preprocess_vad_series(vad_series):
    num_base_features = vad_series.shape[1]

    nan_mask_initial = np.isnan(vad_series)
    if nan_mask_initial.sum() > vad_series.size * 0.5:
        return None

    for dim in range(num_base_features):
        dim_series = vad_series[:, dim]
        is_nan_dim = np.isnan(dim_series)
        if is_nan_dim.any():
            valid_indices = np.where(~is_nan_dim)[0]
            nan_indices = np.where(is_nan_dim)[0]
            if len(valid_indices) < 2:
                vad_series[nan_indices, dim] = 0.5
                continue
            vad_series[nan_indices, dim] = np.interp(
                nan_indices,
                valid_indices,
                dim_series[valid_indices]
            )

    if np.isnan(vad_series).any():
        return None

    return vad_series


def create_sequences_pytorch_with_momentum_features(ml_ready_data_path, lookback_window, forecast_horizon,
                                                    train_until_year=None):
    processed_data_unified = load_and_validate_data(ml_ready_data_path)
    if processed_data_unified is None:
        return tuple([np.array([])] * 8 + [0, 0])

    global_timeline = extract_global_timeline(processed_data_unified)
    if not global_timeline:
        logging.error("Could not determine a global timeline.")
        return tuple([np.array([])] * 8 + [lookback_window, 0])

    num_total_momentum_features = NUM_BASE_FEATURES * NUM_MOMENTUM_FEATURES_PER_DIM
    num_input_features = NUM_BASE_FEATURES + num_total_momentum_features

    all_X_featured, all_y_target_diff, all_years_for_X_last_input = [], [], []
    all_last_actual_vals_for_reconstruction, all_words_for_sequences = [], []

    for word, data in processed_data_unified.items():
        temp_vad = data.get("temporal_vad", {})
        if not all(k in temp_vad for k in ('x', 'v', 'a', 'd')):
            continue

        x_list = temp_vad['x']
        v_list = [np.nan if x is None else float(x) for x in temp_vad['v']]
        a_list = [np.nan if x is None else float(x) for x in temp_vad['a']]
        d_list = [np.nan if x is None else float(x) for x in temp_vad['d']]

        if not (len(x_list) == len(v_list) == len(a_list) == len(d_list)) or not x_list:
            continue

        vad_series = np.array(list(zip(v_list, a_list, d_list)), dtype=float)

        min_len_for_momentum_calc = lookback_window + 5
        if vad_series.shape[0] < lookback_window + forecast_horizon + min_len_for_momentum_calc:
            continue

        vad_series = preprocess_vad_series(vad_series)
        if vad_series is None:
            continue

        diff_vad_series = np.diff(vad_series, axis=0)
        momentum_features_full = calculate_momentum_features(vad_series, window_size=min(10, lookback_window))

        for i in range(lookback_window, len(diff_vad_series) - forecast_horizon + 1):
            if i >= len(momentum_features_full) or (i - lookback_window + 1) < 0:
                break

            current_diff_seq = diff_vad_series[i - lookback_window:i]
            current_momentum_seq = momentum_features_full[i - lookback_window + 1:i + 1]

            if current_momentum_seq.shape[0] != lookback_window or current_diff_seq.shape[0] != lookback_window:
                continue

            combined_input_features_for_sequence = []
            for t_step in range(lookback_window):
                diff_feats = current_diff_seq[t_step]
                momentum_feats_for_step = current_momentum_seq[t_step]
                combined_input_features_for_sequence.append(np.concatenate([diff_feats, momentum_feats_for_step]))

            input_seq_featured = np.array(combined_input_features_for_sequence, dtype=float)
            target_diff = diff_vad_series[i]
            last_actual_val_for_reconstruction = vad_series[i]

            if np.isnan(input_seq_featured).any() or np.isnan(target_diff).any() or np.isnan(
                    last_actual_val_for_reconstruction).any():
                continue

            all_X_featured.append(input_seq_featured)
            all_y_target_diff.append(target_diff)
            all_last_actual_vals_for_reconstruction.append(last_actual_val_for_reconstruction)
            all_words_for_sequences.append(word)

            year_index_last_input = i
            year_of_last_input_val = x_list[year_index_last_input] if year_index_last_input < len(x_list) else -1
            all_years_for_X_last_input.append(year_of_last_input_val)

    if not all_X_featured:
        return tuple([np.array([])] * 8 + [lookback_window, num_input_features])

    X_data = np.array(all_X_featured, dtype=np.float32)
    y_data_target_diff = np.array(all_y_target_diff, dtype=np.float32)
    years_of_X_last_input_data = np.array(all_years_for_X_last_input)
    last_actuals_for_reconstruction_data = np.array(all_last_actual_vals_for_reconstruction, dtype=np.float32)
    words_for_sequences_data = np.array(all_words_for_sequences)

    valid_indices = np.where(years_of_X_last_input_data != -1)[0]
    X_data = X_data[valid_indices]
    y_data_target_diff = y_data_target_diff[valid_indices]
    years_of_X_last_input_data = years_of_X_last_input_data[valid_indices]
    last_actuals_for_reconstruction_data = last_actuals_for_reconstruction_data[valid_indices]
    words_for_sequences_data = words_for_sequences_data[valid_indices]

    if X_data.size == 0:
        return tuple([np.array([])] * 8 + [lookback_window, num_input_features])

    y_actual_values = last_actuals_for_reconstruction_data + y_data_target_diff

    if train_until_year is not None and years_of_X_last_input_data.size > 0:
        train_idx = np.where(years_of_X_last_input_data <= train_until_year)[0]
        test_idx = np.where(years_of_X_last_input_data > train_until_year)[0]

        X_train, y_train_diff = X_data[train_idx], y_data_target_diff[train_idx]
        X_test, y_test_diff = X_data[test_idx], y_data_target_diff[test_idx]

        y_test_actual = y_actual_values[test_idx] if test_idx.size > 0 else np.array([])
        last_actuals_for_test_reconstruction = last_actuals_for_reconstruction_data[
            test_idx] if test_idx.size > 0 else np.array([])
        words_test = words_for_sequences_data[test_idx] if test_idx.size > 0 else np.array([])
        years_X_test = years_of_X_last_input_data[test_idx] if test_idx.size > 0 else np.array([])
    else:
        X_train, y_train_diff = X_data, y_data_target_diff
        X_test, y_test_diff = np.array([]), np.array([])
        y_test_actual = np.array([])
        last_actuals_for_test_reconstruction = np.array([])
        words_test = np.array([])
        years_X_test = np.array([])
    # todo class holding these vals
    return (X_train,
            y_train_diff,
            X_test,
            y_test_diff,
            y_test_actual,
            last_actuals_for_test_reconstruction,
            words_test,
            years_X_test,
            lookback_window,
            num_input_features)