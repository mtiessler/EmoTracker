import pickle
import json
import os
import string
import logging
from nltk.corpus import stopwords
import copy
import numpy as np
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RELATIVE_SENSE_DATA_DIR_COMPONENTS = ['..', '..', 'data', 'Diachronic_Sense_Modeling']
RELATIVE_VAD_LEX_DIR_COMPONENTS = ['..', '..', 'data', 'VAD_Lexicons', 'NRC-VAD-Lexicon-v2.1']
RELATIVE_OUTPUT_DIR_COMPONENTS = ['..', '..', 'data', 'Generated_VAD_Dataset']

SENSE_DATA_FILENAME = 'prob_fitting_10.data'
VAD_LEXICON_FILENAME = 'NRC-VAD-Lexicon-v2.1.txt'
INTERMEDIATE_OUTPUT_FILENAME = 'temporal_vad_with_full_sense_data.json'
ML_READY_OUTPUT_FILENAME = 'ml_ready_temporal_vad_data.json'

DEFAULT_VAD = [0.5, 0.5, 0.5]

START_YEAR_UNIFIED = 1820
END_YEAR_UNIFIED = 2010
TIME_STEP_UNIFIED = 5


def get_paths():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    def build_path(relative_components, filename):
        relative_dir = os.path.join(*relative_components)
        absolute_dir = os.path.normpath(os.path.join(script_dir,
                                                     relative_dir))
        return os.path.join(absolute_dir, filename)

    sense_data_path = build_path(RELATIVE_SENSE_DATA_DIR_COMPONENTS, SENSE_DATA_FILENAME)
    vad_lexicon_path = build_path(RELATIVE_VAD_LEX_DIR_COMPONENTS, VAD_LEXICON_FILENAME)
    intermediate_output_path = build_path(RELATIVE_OUTPUT_DIR_COMPONENTS, INTERMEDIATE_OUTPUT_FILENAME)
    ml_ready_output_path = build_path(RELATIVE_OUTPUT_DIR_COMPONENTS, ML_READY_OUTPUT_FILENAME)
    return sense_data_path, vad_lexicon_path, intermediate_output_path, ml_ready_output_path


def load_vad_lexicon(lexicon_path):
    vad_lexicon = {}
    try:
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            parts = first_line.split('\t')
            try:
                float(parts[1]), float(parts[2]), float(parts[3])
                vad_lexicon[parts[0].lower()] = {'v': float(parts[1]), 'a': float(parts[2]), 'd': float(parts[3])}
            except (IndexError, ValueError):
                pass
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 4:
                    try:
                        word = parts[0].lower()
                        v, a, d = float(parts[1]), float(parts[2]), float(parts[3])
                        vad_lexicon[word] = {'v': v, 'a': a, 'd': d}
                    except (ValueError, IndexError):
                        logging.warning(f"Could not parse line: {line.strip()}")
    except FileNotFoundError:
        logging.error(f"VAD Lexicon file not found: {lexicon_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading VAD lexicon: {e}")
        return None
    return vad_lexicon


def preprocess(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    except LookupError:
        logging.error("NLTK stopwords not found. Ensure NLTK data is downloaded: python -m nltk.downloader stopwords")
        return []
    return tokens


def calculate_sense_vad(definition, vad_lexicon):
    tokens = preprocess(definition)
    if not tokens: return [float(v) for v in DEFAULT_VAD]
    v_sum, a_sum, d_sum, match_count = 0.0, 0.0, 0.0, 0
    for token in tokens:
        if token in vad_lexicon:
            vad = vad_lexicon[token]
            v_sum += vad['v']
            a_sum += vad['a']
            d_sum += vad['d']
            match_count += 1
    if match_count > 0:
        return [float(v_sum / match_count), float(a_sum / match_count), float(d_sum / match_count)]
    else:
        return [float(v) for v in DEFAULT_VAD]


def generate_initial_dataset(sense_data_path, vad_lexicon):
    try:
        with open(sense_data_path, 'rb') as infile:
            sense_data = pickle.load(infile)
    except FileNotFoundError:
        logging.error(f"Sense data file not found: {sense_data_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading sense data: {e}")
        return None

    if vad_lexicon is None: return None

    senses_without_vad_match, total_senses = 0, 0
    words_to_remove = []
    for word, senses in sense_data.items():
        if not isinstance(senses, dict):
            logging.warning(f"Senses for word '{word}' is not a dict. Marking for removal.")
            words_to_remove.append(word)
            continue
        valid_senses_found_for_word = False
        for sense_id, sense_info in senses.items():
            if isinstance(sense_info, dict) and 'definition' in sense_info:
                total_senses += 1
                valid_senses_found_for_word = True
                sense_vad = calculate_sense_vad(sense_info['definition'], vad_lexicon)
                sense_info['vad'] = sense_vad
                if sense_vad == DEFAULT_VAD:
                    senses_without_vad_match += 1
        if not valid_senses_found_for_word:
            logging.warning(f"No valid senses found for word '{word}'. Marking for removal.")
            words_to_remove.append(word)

    for word in words_to_remove:
        if word in sense_data: del sense_data[word]

    output_dataset = {}
    for word, senses in sense_data.items():
        if not isinstance(senses, dict) or not senses:
            continue

        years_list_for_word, num_time_steps = None, 0
        for sense_info_outer in senses.values():
            if isinstance(sense_info_outer, dict) and 'x' in sense_info_outer and isinstance(sense_info_outer['x'],
                                                                                 list) and sense_info_outer[
                'x']:
                years_list_for_word = [int(y) for y in sense_info_outer['x']]
                num_time_steps = len(years_list_for_word)
                break
        if years_list_for_word is None:
            logging.warning(f"Word '{word}' has no senses with time data ('x' array). Skipping word.")
            continue

        word_v_trajectory = [0.0] * num_time_steps
        word_a_trajectory = [0.0] * num_time_steps
        word_d_trajectory = [0.0] * num_time_steps
        for t_idx in range(num_time_steps):
            v_sum_t, a_sum_t, d_sum_t = 0.0, 0.0, 0.0
            for sense_id, sense_info in senses.items():
                if (isinstance(sense_info, dict) and 'vad' in sense_info and
                        'y_fitting' in sense_info and isinstance(sense_info['y_fitting'], list) and
                        len(sense_info[
                                'y_fitting']) == num_time_steps):  # Ensure y_fitting aligns with word's num_time_steps
                    try:
                        proportion = float(sense_info['y_fitting'][t_idx])
                        if math.isnan(proportion):  # Check for NaN explicitly after conversion
                            proportion = 0.0
                    except (ValueError, TypeError):
                        proportion = 0.0  # Default if conversion fails

                    proportion = max(0.0, proportion)  # Ensure proportion is not negative
                    sense_vad = sense_info['vad']
                    v_sum_t += proportion * sense_vad[0]
                    a_sum_t += proportion * sense_vad[1]
                    d_sum_t += proportion * sense_vad[2]
            word_v_trajectory[t_idx], word_a_trajectory[t_idx], word_d_trajectory[t_idx] = float(v_sum_t), float(
                a_sum_t), float(d_sum_t)

        word_temporal_data = {'x': years_list_for_word, 'v': word_v_trajectory, 'a': word_a_trajectory,
                              'd': word_d_trajectory}

        word_senses_data_cleaned = {}
        for sense_id, sense_info in senses.items():
            if (isinstance(sense_info, dict) and
                    all(k in sense_info for k in ['definition', 'vad', 'y', 'y_fitting'])):
                # Ensure y and y_fitting also align with the word's primary x-axis (num_time_steps)
                if isinstance(sense_info['y'], list) and len(sense_info['y']) == num_time_steps and \
                        isinstance(sense_info['y_fitting'], list) and len(sense_info['y_fitting']) == num_time_steps:
                    word_senses_data_cleaned[sense_id] = {
                        'definition': sense_info['definition'],
                        'vad': [float(val) for val in sense_info['vad']],
                        'y': [float(val) for val in sense_info['y']],  # Ensure y values are floats
                        'y_fitting': [float(val) for val in sense_info['y_fitting']]
                    }
        output_dataset[word] = {'temporal_vad': word_temporal_data, 'senses': word_senses_data_cleaned}
    logging.info(
        f"Senses processed: {total_senses}. Senses without VAD match (using default): {senses_without_vad_match}")
    return output_dataset


def _interpolate_series(x_original, series_original, target_step_size):
    if not x_original or not series_original or len(x_original) != len(series_original) or len(x_original) < 2:
        return x_original, series_original

    x_interpolated = [float(x_original[0])]
    series_interpolated = [float(series_original[0])]

    for i in range(len(x_original) - 1):
        x1, x2 = float(x_original[i]), float(x_original[i + 1])
        y1_val = series_original[i]
        y2_val = series_original[i + 1]

        y1_is_nan = math.isnan(float(y1_val)) if isinstance(y1_val, (int, float, np.number)) else True
        y2_is_nan = math.isnan(float(y2_val)) if isinstance(y2_val, (int, float, np.number)) else True

        y1 = float(y1_val) if not y1_is_nan else np.nan  # Use np.nan for consistent NaN type
        y2 = float(y2_val) if not y2_is_nan else np.nan

        if (x2 - x1) > target_step_size:
            mid_x = (x1 + x2) / 2.0
            if y1_is_nan or y2_is_nan:
                mid_y = np.nan
            else:
                mid_y = (y1 + y2) / 2.0
            x_interpolated.append(mid_x)
            series_interpolated.append(mid_y)
        x_interpolated.append(x2)
        series_interpolated.append(y2)

    final_map = dict(zip(x_interpolated, series_interpolated))
    sorted_x = sorted(final_map.keys())
    return sorted_x, [final_map[k] for k in sorted_x]


def _unify_series_to_timeline(x_series, y_series, global_unified_timeline, nan_value):
    series_map = {float(x_val): float(y_val) for x_val, y_val in zip(x_series, y_series)}
    unified_y_list = []
    for target_year_int in global_unified_timeline:
        target_year_float = float(target_year_int)
        val = nan_value
        if target_year_float in series_map:
            val = series_map[target_year_float]
        unified_y_list.append(val)
    return unified_y_list


def augment_and_unify_ml_data(initial_dataset_param, start_year, end_year, time_step):
    if not initial_dataset_param: return {}

    data_for_processing = copy.deepcopy(initial_dataset_param)
    nan_value = np.nan

    temp_timeline_np = np.arange(start_year, end_year + time_step, time_step, dtype=int)
    if temp_timeline_np.size > 0 and temp_timeline_np[-1] > end_year:
        temp_timeline_np = np.arange(start_year, end_year + 1, time_step,
                                     dtype=int)  # +1 to include end_year if it's a step
        temp_timeline_np = temp_timeline_np[temp_timeline_np <= end_year]

    unified_timeline_global = [int(year) for year in temp_timeline_np]

    ml_ready_dataset = {}

    for word, original_word_info in data_for_processing.items():
        processed_word_temporal_vad = {}
        processed_word_senses = {}

        original_tv_data = original_word_info.get('temporal_vad', {})
        x_orig_tv = [int(y) for y in original_tv_data.get('x', []) if y is not None]
        v_orig_tv = [float(v) if isinstance(v, (int, float, np.number)) and not math.isnan(float(v)) else nan_value for
                     v in original_tv_data.get('v', [])]
        a_orig_tv = [float(a) if isinstance(a, (int, float, np.number)) and not math.isnan(float(a)) else nan_value for
                     a in original_tv_data.get('a', [])]
        d_orig_tv = [float(d) if isinstance(d, (int, float, np.number)) and not math.isnan(float(d)) else nan_value for
                     d in original_tv_data.get('d', [])]

        if x_orig_tv and len(x_orig_tv) == len(v_orig_tv) and len(x_orig_tv) == len(a_orig_tv) and len(
                x_orig_tv) == len(d_orig_tv):
            x_aug_tv, v_aug_tv = _interpolate_series(x_orig_tv, v_orig_tv, time_step)
            _, a_aug_tv = _interpolate_series(x_orig_tv, a_orig_tv, time_step)
            _, d_aug_tv = _interpolate_series(x_orig_tv, d_orig_tv, time_step)

            v_unified = _unify_series_to_timeline(x_aug_tv, v_aug_tv, unified_timeline_global, nan_value)
            a_unified = _unify_series_to_timeline(x_aug_tv, a_aug_tv, unified_timeline_global, nan_value)
            d_unified = _unify_series_to_timeline(x_aug_tv, d_aug_tv, unified_timeline_global, nan_value)

            processed_word_temporal_vad = {
                "x": unified_timeline_global,
                "v": v_unified, "a": a_unified, "d": d_unified
            }
        else:
            logging.warning(f"Temporal VAD data for '{word}' is missing, incomplete, or mismatched. Filling with NaNs.")
            processed_word_temporal_vad = {
                "x": unified_timeline_global,
                "v": [nan_value] * len(unified_timeline_global),
                "a": [nan_value] * len(unified_timeline_global),
                "d": [nan_value] * len(unified_timeline_global)
            }

        if 'senses' in original_word_info:
            for sense_id, original_sense_info in original_word_info['senses'].items():
                y_orig_sense = [
                    float(val) if isinstance(val, (int, float, np.number)) and not math.isnan(float(val)) else nan_value
                    for val in original_sense_info.get('y', [])]
                y_fitting_orig_sense = [
                    float(val) if isinstance(val, (int, float, np.number)) and not math.isnan(float(val)) else nan_value
                    for val in original_sense_info.get('y_fitting', [])]

                y_unified_sense = [nan_value] * len(unified_timeline_global)
                y_fitting_unified_sense = [nan_value] * len(unified_timeline_global)

                if x_orig_tv and len(x_orig_tv) == len(y_orig_sense) and len(x_orig_tv) == len(y_fitting_orig_sense):
                    x_aug_sense, y_aug_sense = _interpolate_series(x_orig_tv, y_orig_sense, time_step)
                    _, y_fitting_aug_sense = _interpolate_series(x_orig_tv, y_fitting_orig_sense, time_step)

                    y_unified_sense = _unify_series_to_timeline(x_aug_sense, y_aug_sense, unified_timeline_global,
                                                                nan_value)
                    y_fitting_unified_sense = _unify_series_to_timeline(x_aug_sense, y_fitting_aug_sense,
                                                                        unified_timeline_global, nan_value)
                else:
                    logging.warning(
                        f"Sense data y/y_fitting for '{word}' sense '{sense_id}' is missing, incomplete, or mismatched with word's x-axis. Filling with NaNs.")

                processed_word_senses[sense_id] = {
                    "definition": original_sense_info.get("definition"),
                    "vad": [float(v) for v in original_sense_info.get("vad", DEFAULT_VAD)],  # Ensure floats
                    "y": y_unified_sense,
                    "y_fitting": y_fitting_unified_sense
                }

        ml_ready_dataset[word] = {
            "temporal_vad": processed_word_temporal_vad,
            "senses": processed_word_senses
        }
    return ml_ready_dataset


def replace_nan_with_none(obj):
    if isinstance(obj, dict):
        return {k: replace_nan_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan_with_none(elem) for elem in obj]
    elif isinstance(obj, (float, np.floating)) and np.isnan(obj):
        return None
    return obj


def main():
    sense_data_p, vad_lexicon_p, intermediate_out_p, ml_ready_out_p = get_paths()

    vad_lexicon = load_vad_lexicon(vad_lexicon_p)
    if vad_lexicon is None:
        logging.error("Failed to load VAD lexicon. Exiting.")
        return

    initial_dataset = generate_initial_dataset(sense_data_p, vad_lexicon)
    if initial_dataset is None:
        logging.error("Failed to generate initial dataset. Exiting.")
        return

    initial_dataset_fixed_for_json = replace_nan_with_none(initial_dataset)

    logging.info(f"Saving intermediate dataset (before augmentation/unification) to: {intermediate_out_p}")
    try:
        output_dir_intermediate = os.path.dirname(intermediate_out_p)
        os.makedirs(output_dir_intermediate, exist_ok=True)
        with open(intermediate_out_p, 'w', encoding='utf-8') as outfile:
            json.dump(initial_dataset_fixed_for_json, outfile, ensure_ascii=False,
                      indent=4)
        logging.info(f"Intermediate dataset saved to {intermediate_out_p}")
    except Exception as e:
        logging.error(f"Error saving intermediate dataset: {e}")

    logging.info("Starting ML data augmentation and unification (including senses)...")

    ml_data = augment_and_unify_ml_data(initial_dataset,
                                        start_year=START_YEAR_UNIFIED,
                                        end_year=END_YEAR_UNIFIED,
                                        time_step=TIME_STEP_UNIFIED)
    if not ml_data:
        logging.error("Failed to generate ML-ready data. Exiting.")
        return

    ml_data_fixed_for_json = replace_nan_with_none(ml_data)


    logging.info(f"Saving ML-ready dataset to: {ml_ready_out_p}")
    try:
        output_dir_ml = os.path.dirname(ml_ready_out_p)
        os.makedirs(output_dir_ml, exist_ok=True)
        with open(ml_ready_out_p, 'w', encoding='utf-8') as outfile:
            json.dump(ml_data_fixed_for_json, outfile, ensure_ascii=False, indent=4)
        logging.info(f"ML-ready dataset saved to {ml_ready_out_p}")
    except Exception as e:
        logging.error(f"Error saving ML-ready dataset: {e}")

    if 'explore' in ml_data and 'temporal_vad' in ml_data['explore']:
        print("\nSample data for 'explore' in ML-ready dataset (Python internal representation):")
        explore_tv = ml_data['explore']['temporal_vad']
        if 'x' in explore_tv:
            explore_x = explore_tv['x']
            print(f"Word 'explore' temporal_vad.x (sample): {explore_x[:6]}... (Total: {len(explore_x)})")
        if 'v' in explore_tv:
            print(
                f"Word 'explore' temporal_vad.v (sample): {[x if not np.isnan(x) else 'NaN' for x in explore_tv['v'][:3]]}... (Total: {len(explore_tv['v'])})")
        if 'senses' in ml_data['explore'] and ml_data['explore']['senses']:
            first_sense_id = list(ml_data['explore']['senses'].keys())[0]
            first_sense_data = ml_data['explore']['senses'][first_sense_id]
            y_sample = first_sense_data.get('y', [])
            print(
                f"Word 'explore' first sense ('{first_sense_id}') y (sample): {[x if not np.isnan(x) else 'NaN' for x in y_sample[:3]]}... (Total: {len(y_sample)})")
            print(f"Senses data for 'explore' included.")


if __name__ == "__main__":
    try:
        main()
    except Exception as main_e:
        logging.exception(f"Critical Error in main execution: {main_e}")
        exit(1)