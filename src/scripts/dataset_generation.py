import pickle
import json
import os
import string
import logging
from nltk.corpus import stopwords

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Constants ---
RELATIVE_SENSE_DATA_DIR_COMPONENTS = ['..', '..', 'data', 'Diachronic_Sense_Modeling']
RELATIVE_VAD_LEX_DIR_COMPONENTS = ['..', '..', 'data', 'VAD_Lexicons', 'NRC-VAD-Lexicon-v2.1']
RELATIVE_OUTPUT_DIR_COMPONENTS = ['..', '..', 'data', 'Generated_VAD_Dataset']

SENSE_DATA_FILENAME = 'prob_fitting_10.data'
VAD_LEXICON_FILENAME = 'NRC-VAD-Lexicon-v2.1.txt'
OUTPUT_FILENAME = 'temporal_vad_with_full_sense_data.json'

DEFAULT_VAD = [0.5, 0.5, 0.5] # Neutral VAD

def get_paths():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
        logging.warning("__file__ not defined. Using CWD as base.")

    def build_path(relative_components, filename):
        relative_dir = os.path.join(*relative_components)
        absolute_dir = os.path.normpath(os.path.join(script_dir, relative_dir))
        return os.path.join(absolute_dir, filename)

    sense_data_path = build_path(RELATIVE_SENSE_DATA_DIR_COMPONENTS, SENSE_DATA_FILENAME)
    vad_lexicon_path = build_path(RELATIVE_VAD_LEX_DIR_COMPONENTS, VAD_LEXICON_FILENAME)
    output_path = build_path(RELATIVE_OUTPUT_DIR_COMPONENTS, OUTPUT_FILENAME)
    return sense_data_path, vad_lexicon_path, output_path

# --- VAD Lexicon Loading ---
def load_vad_lexicon(lexicon_path):
    vad_lexicon = {}
    logging.info(f"Loading VAD lexicon from: {lexicon_path}")
    try:
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            parts = first_line.split('\t')
            try:
                float(parts[1]), float(parts[2]), float(parts[3])
                vad_lexicon[parts[0].lower()] = {'v': float(parts[1]), 'a': float(parts[2]), 'd': float(parts[3])}
            except (IndexError, ValueError): # Assume header
                logging.info(f"Skipping potential header line: {first_line}")
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
        logging.error(f"VAD Lexicon file not found: {lexicon_path}"); return None
    except Exception as e:
        logging.error(f"Error loading VAD lexicon: {e}"); return None
    logging.info(f"Loaded {len(vad_lexicon)} VAD entries.")
    return vad_lexicon

def preprocess(text):
    if not isinstance(text, str):
        logging.warning(f"Invalid definition type: {type(text)}"); return []
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    except LookupError:
        logging.error("NLTK stopwords not found. Download via: python -m nltk.downloader stopwords"); return []
    return tokens

def calculate_sense_vad(definition, vad_lexicon):
    tokens = preprocess(definition)
    if not tokens: return DEFAULT_VAD
    v_sum, a_sum, d_sum, match_count = 0.0, 0.0, 0.0, 0
    for token in tokens:
        if token in vad_lexicon:
            vad = vad_lexicon[token]
            v_sum += vad['v']; a_sum += vad['a']; d_sum += vad['d']
            match_count += 1
    if match_count > 0:
        return [v_sum / match_count, a_sum / match_count, d_sum / match_count]
    else:
        return DEFAULT_VAD

def generate_temporal_vad_dataset(sense_data_path, vad_lexicon_path, output_path):
    # 1. Load Sense Data
    logging.info(f"Loading sense data from: {sense_data_path}")
    try:
        with open(sense_data_path, 'rb') as infile: sense_data = pickle.load(infile)
        logging.info(f"Loaded sense data for {len(sense_data)} words.")
    except FileNotFoundError: logging.error(f"Sense data file not found: {sense_data_path}"); return
    except pickle.UnpicklingError as e: logging.error(f"Error unpickling sense data: {e}"); return
    except Exception as e: logging.error(f"Unexpected error loading sense data: {e}"); return

    # 2. Load VAD Lexicon
    vad_lexicon = load_vad_lexicon(vad_lexicon_path)
    if vad_lexicon is None: logging.error("VAD lexicon load failed. Aborting."); return

    # 3. Pre-calculate VAD for all Senses
    logging.info("Calculating VAD scores for all sense definitions...")
    senses_without_vad_match, total_senses = 0, 0
    words_to_remove = []
    for word, senses in sense_data.items():
        if not isinstance(senses, dict):
             logging.warning(f"Word '{word}': Unexpected data format ({type(senses)}). Skipping.")
             words_to_remove.append(word); continue
        valid_senses_found = False
        for sense_id, sense_info in senses.items():
            if isinstance(sense_info, dict) and 'definition' in sense_info:
                total_senses += 1; valid_senses_found = True
                sense_vad = calculate_sense_vad(sense_info['definition'], vad_lexicon)
                sense_info['vad'] = sense_vad # Store calculated VAD back
                if sense_vad == DEFAULT_VAD: senses_without_vad_match += 1
            else:
                 logging.warning(f"Sense '{sense_id}' for '{word}': Missing definition/invalid format. Skipping.")
        if not valid_senses_found:
             logging.warning(f"Word '{word}': No valid senses found. Skipping.")
             words_to_remove.append(word)
    # Remove words with no processable senses
    for word in words_to_remove:
        if word in sense_data: del sense_data[word]
    logging.info(f"Sense VAD calculation complete. {senses_without_vad_match}/{total_senses} senses got default VAD.")
    if words_to_remove: logging.info(f"Removed {len(words_to_remove)} words lacking valid sense data.")

    # 4. Calculate Temporal Word VAD & Structure Output
    logging.info("Generating temporal VAD trajectories and structuring output...")
    output_dataset = {}
    words_processed = 0
    for word, senses in sense_data.items():
        if not isinstance(senses, dict) or not senses: continue

        # Determine years and num_time_steps from first valid sense
        years, num_time_steps = None, 0
        for sense_info in senses.values():
             if isinstance(sense_info, dict) and 'x' in sense_info and isinstance(sense_info['x'], list) and sense_info['x']:
                 years, num_time_steps = sense_info['x'], len(sense_info['x']); break
        if years is None:
             logging.warning(f"Word '{word}': Cannot determine time steps. Skipping."); continue

        # Calculate temporal trajectory
        word_v_trajectory = [0.0] * num_time_steps
        word_a_trajectory = [0.0] * num_time_steps
        word_d_trajectory = [0.0] * num_time_steps
        for t_idx in range(num_time_steps):
            v_sum_t, a_sum_t, d_sum_t = 0.0, 0.0, 0.0
            for sense_id, sense_info in senses.items():
                 if (isinstance(sense_info, dict) and 'vad' in sense_info and
                     'y_fitting' in sense_info and isinstance(sense_info['y_fitting'], list) and
                     len(sense_info['y_fitting']) == num_time_steps):
                    proportion = max(0.0, sense_info['y_fitting'][t_idx])
                    sense_vad = sense_info['vad']
                    v_sum_t += proportion * sense_vad[0]
                    a_sum_t += proportion * sense_vad[1]
                    d_sum_t += proportion * sense_vad[2]
            word_v_trajectory[t_idx], word_a_trajectory[t_idx], word_d_trajectory[t_idx] = v_sum_t, a_sum_t, d_sum_t

        word_temporal_data = {'x': years, 'v': word_v_trajectory, 'a': word_a_trajectory, 'd': word_d_trajectory}

        word_senses_data = {}
        for sense_id, sense_info in senses.items():
             if (isinstance(sense_info, dict) and
                 all(k in sense_info for k in ['definition', 'vad', 'y', 'y_fitting'])):
                 if isinstance(sense_info['y'], list) and isinstance(sense_info['y_fitting'], list):
                     word_senses_data[sense_id] = {
                         'definition': sense_info['definition'],
                         'vad': sense_info['vad'],
                         'y': sense_info['y'],
                         'y_fitting': sense_info['y_fitting']
                     }
                 else:
                      logging.debug(f"Sense {sense_id} ('{word}'): Skipping output due to non-list type for y/y_fitting.")
             else:
                  logging.debug(f"Sense {sense_id} ('{word}'): Skipping output due to missing required keys.")

        output_dataset[word] = {'temporal_vad': word_temporal_data, 'senses': word_senses_data}

        words_processed += 1
        if words_processed % 100 == 0: logging.info(f"Processed {words_processed} words...")
    logging.info(f"Finished generating VAD trajectories for {words_processed} words.")

    # 5. Save Output
    logging.info(f"Saving generated dataset to: {output_path}")
    try:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True) # Create dir if needed
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(output_dataset, outfile, ensure_ascii=False, indent=4)
        logging.info("Dataset saved successfully.")
    except IOError as e: logging.error(f"IOError saving JSON '{output_path}': {e}")
    except TypeError as e: logging.error(f"TypeError serializing dataset to JSON: {e}")
    except Exception as e: logging.error(f"Unexpected error saving dataset: {e}")

if __name__ == "__main__":
    try:
        sense_data_p, vad_lexicon_p, output_p = get_paths()
        logging.info(f"Sense data path: {sense_data_p}")
        logging.info(f"VAD lexicon path: {vad_lexicon_p}")
        logging.info(f"Output dataset path: {output_p}")

        generate_temporal_vad_dataset(sense_data_p, vad_lexicon_p, output_p)

    except Exception as main_e:
        logging.exception(f"Unexpected Error in main execution block: {main_e}")
        exit(1)