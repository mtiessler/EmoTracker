import numpy as np
import torch
import logging
from features import calculate_momentum_features
import config


def _validate_prediction_inputs(word_to_predict, predict_from_year, predict_until_year):
    if not config.resources_loaded_pytorch:
        raise Exception('API resources not loaded. Check backend logs.')

    if word_to_predict not in config.full_vad_data or "temporal_vad" not in config.full_vad_data[word_to_predict]:
        raise Exception(f"Word '{word_to_predict}' not found or no temporal data.")

    if predict_from_year >= predict_until_year:
        raise Exception(
            f"predict_from_year ({predict_from_year}) must be less than predict_until_year ({predict_until_year})")


def _extract_historical_vad_data(word_to_predict, predict_from_year):
    word_data_history = config.full_vad_data[word_to_predict]["temporal_vad"]
    x_list_history = word_data_history["x"]
    v_list_history = [np.nan if x is None else float(x) for x in word_data_history['v']]
    a_list_history = [np.nan if x is None else float(x) for x in word_data_history['a']]
    d_list_history = [np.nan if x is None else float(x) for x in word_data_history['d']]

    if predict_from_year not in x_list_history:
        raise Exception(f"Year {predict_from_year} not in trajectory for '{word_to_predict}'.")

    target_index_in_history = x_list_history.index(predict_from_year)

    return v_list_history, a_list_history, d_list_history, target_index_in_history


def _validate_historical_data_length(target_index_in_history, word_to_predict, predict_from_year):
    momentum_calc_window_internal = min(10, config.lookback_window)
    required_vad_hist_len = config.lookback_window + momentum_calc_window_internal

    if target_index_in_history + 1 < required_vad_hist_len:
        raise Exception(
            f"Not enough historical data for {word_to_predict}. Need {required_vad_hist_len} "
            f"VAD points ending at {predict_from_year} for initial features, have {target_index_in_history + 1}.")

    return momentum_calc_window_internal, required_vad_hist_len


def _extract_relevant_vad_segment(v_list_history, a_list_history, d_list_history,
                                  target_index_in_history, required_vad_hist_len):
    hist_vad_start_idx = target_index_in_history + 1 - required_vad_hist_len

    relevant_v = v_list_history[hist_vad_start_idx: target_index_in_history + 1]
    relevant_a = a_list_history[hist_vad_start_idx: target_index_in_history + 1]
    relevant_d = d_list_history[hist_vad_start_idx: target_index_in_history + 1]

    return np.array(list(zip(relevant_v, relevant_a, relevant_d)), dtype=float)


def _interpolate_nan_values(vad_segment, word_to_predict):
    num_base_features = 3

    for dim in range(num_base_features):
        dim_series = vad_segment[:, dim]
        is_nan_dim = np.isnan(dim_series)

        if is_nan_dim.any():
            valid_indices = np.where(~is_nan_dim)[0]
            nan_idx_in_segment = np.where(is_nan_dim)[0]

            if len(valid_indices) < 2:
                vad_segment[nan_idx_in_segment, dim] = 0.5
                continue

            vad_segment[nan_idx_in_segment, dim] = np.interp(
                nan_idx_in_segment, valid_indices, dim_series[valid_indices]
            )

    if np.isnan(vad_segment).any():
        logging.warning(f"NaNs remain in VAD for {word_to_predict} after interpolation. "
                        f"Predictions might be affected.")
        vad_segment = np.nan_to_num(vad_segment, nan=0.5)

    return vad_segment


def _construct_initial_input_sequence(vad_segment, momentum_calc_window_internal):
    diff_on_segment = np.diff(vad_segment, axis=0)
    momentum_on_segment = calculate_momentum_features(vad_segment, window_size=momentum_calc_window_internal)

    initial_input_unscaled_list = []
    idx_of_vad_at_predict_from_year = len(vad_segment) - 1

    for k in range(config.lookback_window):
        diff_vec = diff_on_segment[idx_of_vad_at_predict_from_year - config.lookback_window + k]
        momentum_vec = momentum_on_segment[idx_of_vad_at_predict_from_year - config.lookback_window + 1 + k]
        combined_f = np.concatenate([diff_vec, momentum_vec])
        initial_input_unscaled_list.append(combined_f)

    return np.array(initial_input_unscaled_list, dtype=np.float32)


def _validate_feature_shape(features, expected_shape):
    if features.shape != expected_shape:
        err_msg = f"Feature shape error. Expected {expected_shape}, got {features.shape}"
        logging.error(err_msg)
        raise Exception(err_msg)


def _scale_input_features(unscaled_features):
    scaled_features = np.zeros_like(unscaled_features)

    for i_feat in range(config.num_input_features):
        mean = config.input_scaler_params_list[i_feat]['mean']
        scale = config.input_scaler_params_list[i_feat]['scale']

        if scale != 0:
            scaled_features[:, i_feat] = (unscaled_features[:, i_feat] - mean) / scale
        else:
            scaled_features[:, i_feat] = 0.0

    return scaled_features


def _unscale_output_features(scaled_output):
    unscaled_output = np.zeros(config.num_output_features, dtype=np.float32)

    for i_out_feat in range(config.num_output_features):
        mean = config.output_scaler_params_list[i_out_feat]['mean']
        scale = config.output_scaler_params_list[i_out_feat]['scale']

        if scale != 0:
            unscaled_output[i_out_feat] = (scaled_output[i_out_feat] * scale) + mean
        else:
            unscaled_output[i_out_feat] = mean

    return unscaled_output


def _make_single_prediction(scaled_input):
    input_tensor = torch.from_numpy(scaled_input.astype(np.float32)).unsqueeze(0).to(config.device)

    with torch.no_grad():
        scaled_predicted_diff_tensor = config.model(input_tensor)

    return scaled_predicted_diff_tensor.cpu().numpy()[0]


def _reconstruct_vad_from_diff(last_vad, predicted_diff):
    return last_vad + predicted_diff


def _create_prediction_output(current_year, reconstructed_vad):
    return {
        'time': current_year,
        'v': float(reconstructed_vad[0]),
        'a': float(reconstructed_vad[1]),
        'd': float(reconstructed_vad[2])
    }


def _update_vad_segment(vad_segment, new_vad):
    num_base_features = 3
    return np.vstack([vad_segment[1:], new_vad.reshape(1, num_base_features)])


def _calculate_new_momentum_features(vad_segment, momentum_calc_window_internal):
    if len(vad_segment) < momentum_calc_window_internal:
        return np.zeros(config.num_input_features - 3)

    all_momentums_new_segment = calculate_momentum_features(vad_segment, window_size=momentum_calc_window_internal)
    return all_momentums_new_segment[-1]


def _construct_new_feature_vector(predicted_diff, new_momentum):
    new_full_feature_unscaled = np.concatenate((predicted_diff, new_momentum))

    if new_full_feature_unscaled.shape[0] != config.num_input_features:
        err_msg = f"New feature vector shape error. Expected {config.num_input_features}, got {new_full_feature_unscaled.shape[0]}"
        logging.error(err_msg)
        raise Exception(err_msg)

    return new_full_feature_unscaled


def _scale_new_feature_vector(unscaled_features):
    scaled_features = np.zeros(config.num_input_features, dtype=np.float32)

    for i_feat in range(config.num_input_features):
        mean = config.input_scaler_params_list[i_feat]['mean']
        scale = config.input_scaler_params_list[i_feat]['scale']

        if scale != 0:
            scaled_features[i_feat] = (unscaled_features[i_feat] - mean) / scale
        else:
            scaled_features[i_feat] = 0.0

    return scaled_features


def _update_model_input(current_input, new_feature_vector):
    return np.vstack((current_input[1:], new_feature_vector.reshape(1, config.num_input_features)))


def _check_safety_break(predictions_count, predict_from_year, predict_until_year):
    time_span_years = predict_until_year - predict_from_year
    expected_prediction_steps = time_span_years / config.TIME_STEP_YEARS
    safety_buffer = 2 * config.lookback_window
    max_expected_predictions = expected_prediction_steps + safety_buffer

    if predictions_count > max_expected_predictions:
        logging.warning("Iterative prediction loop exceeded a safe number of steps. Breaking.")
        return True

    return False

def _run_iterative_prediction_loop(initial_scaled_input, vad_segment, predict_from_year,
                                   predict_until_year, momentum_calc_window_internal):
    all_predictions = []
    current_year = predict_from_year
    last_actual_vad = vad_segment[-1, :].copy()
    current_scaled_input = initial_scaled_input.copy()
    current_vad_segment = vad_segment.copy()

    while current_year < predict_until_year:
        scaled_predicted_diff = _make_single_prediction(current_scaled_input)
        unscaled_predicted_diff = _unscale_output_features(scaled_predicted_diff)
        reconstructed_vad = _reconstruct_vad_from_diff(last_actual_vad, unscaled_predicted_diff)

        current_year += config.TIME_STEP_YEARS
        prediction_output = _create_prediction_output(current_year, reconstructed_vad)
        all_predictions.append(prediction_output)

        last_actual_vad = reconstructed_vad.copy()
        current_vad_segment = _update_vad_segment(current_vad_segment, reconstructed_vad)

        new_momentum = _calculate_new_momentum_features(current_vad_segment, momentum_calc_window_internal)
        new_feature_unscaled = _construct_new_feature_vector(unscaled_predicted_diff, new_momentum)
        new_feature_scaled = _scale_new_feature_vector(new_feature_unscaled)

        current_scaled_input = _update_model_input(current_scaled_input, new_feature_scaled)

        if _check_safety_break(len(all_predictions), predict_from_year, predict_until_year):
            break

    return all_predictions


def predict_vad_trajectory(word_to_predict, predict_from_year, predict_until_year):
    _validate_prediction_inputs(word_to_predict, predict_from_year, predict_until_year)

    v_history, a_history, d_history, target_index = _extract_historical_vad_data(word_to_predict, predict_from_year)

    momentum_window, required_hist_len = _validate_historical_data_length(target_index, word_to_predict,
                                                                          predict_from_year)

    vad_segment = _extract_relevant_vad_segment(v_history, a_history, d_history, target_index, required_hist_len)

    vad_segment = _interpolate_nan_values(vad_segment, word_to_predict)

    initial_input_unscaled = _construct_initial_input_sequence(vad_segment, momentum_window)

    expected_shape = (config.lookback_window, config.num_input_features)
    _validate_feature_shape(initial_input_unscaled, expected_shape)

    initial_input_scaled = _scale_input_features(initial_input_unscaled)

    predictions = _run_iterative_prediction_loop(
        initial_input_scaled, vad_segment, predict_from_year,
        predict_until_year, momentum_window
    )

    logging.info(
        f"PyTorch iterative prediction successful for '{word_to_predict}'. Predicted {len(predictions)} steps.")

    return predictions


def predict_single_step(word_to_predict, predict_from_year):
    predictions = predict_vad_trajectory(word_to_predict, predict_from_year, predict_from_year + config.TIME_STEP_YEARS)
    return predictions[0] if predictions else None


def get_prediction_metadata(word_to_predict, predict_from_year):
    _validate_prediction_inputs(word_to_predict, predict_from_year, predict_from_year + config.TIME_STEP_YEARS)

    v_history, a_history, d_history, target_index = _extract_historical_vad_data(word_to_predict, predict_from_year)
    momentum_window, required_hist_len = _validate_historical_data_length(target_index, word_to_predict,
                                                                          predict_from_year)

    return {
        'word': word_to_predict,
        'available_history_points': target_index + 1,
        'required_history_points': required_hist_len,
        'momentum_window_size': momentum_window,
        'lookback_window': config.lookback_window,
        'time_step_years': config.TIME_STEP_YEARS,
        'num_input_features': config.num_input_features,
        'num_output_features': config.num_output_features
    }