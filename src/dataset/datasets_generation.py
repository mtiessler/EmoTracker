import os
import sys
import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pickle
import copy
import string
import math
from nltk.corpus import stopwords

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_VAD = [0.5, 0.5, 0.5]
START_YEAR_UNIFIED = 1820
END_YEAR_UNIFIED = 2010
TIME_STEP_UNIFIED = 5
MAX_NULL_VALUES_PER_DIMENSION = 15

RELATIVE_SENSE_DATA_DIR_COMPONENTS = ['..', '..', 'data', 'Diachronic_Sense_Modeling']
RELATIVE_OUTPUT_DIR_COMPONENTS = ['..', '..', 'data', 'Generated_VAD_Dataset']
SENSE_DATA_FILENAME = 'prob_fitting_10.data'


class LexiconDatasetGenerator:
    def __init__(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()

        def build_path(relative_components, filename=None):
            relative_dir = os.path.join(*relative_components)
            absolute_dir = os.path.normpath(os.path.join(script_dir, relative_dir))
            if filename:
                return os.path.join(absolute_dir, filename)
            return absolute_dir

        self.sense_data_path = build_path(RELATIVE_SENSE_DATA_DIR_COMPONENTS, SENSE_DATA_FILENAME)
        self.output_base_dir = Path(build_path(RELATIVE_OUTPUT_DIR_COMPONENTS))
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        self.lexicon_configs = {
            'nrc': {
                'path': build_path(['..', '..', 'data', 'VAD_Lexicons', 'NRC-VAD-Lexicon-v2.1'],
                                   'NRC-VAD-Lexicon-v2.1.txt'),
                'loader': self._load_nrc_lexicon
            },
            'warriner': {
                'path': build_path(['..', '..', 'data', 'VAD_Lexicons', 'Warriner_et_al'],
                                   'Ratings_Warriner_et_al.csv'),
                'loader': self._load_warriner_lexicon
            },
            'memolon': {
                'path': build_path(['..', '..', 'data', 'VAD_Lexicons', 'MEmoLon'], 'en.tsv'),
                'loader': self._load_memolon_lexicon
            }
        }

    def _load_nrc_lexicon(self, file_path):
        lexicon = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 4:
                        try:
                            word = parts[0].lower()
                            v, a, d = float(parts[1]), float(parts[2]), float(parts[3])
                            lexicon[word] = {'v': v, 'a': a, 'd': d}
                        except (ValueError, IndexError):
                            continue
            logger.info(f"Loaded NRC lexicon: {len(lexicon)} words")
        except Exception as e:
            logger.error(f"Failed to load NRC lexicon: {e}")
        return lexicon

    def _load_warriner_lexicon(self, file_path):
        """Load Warriner lexicon from CSV file"""
        lexicon = {}
        try:
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)

            word_col = None
            v_col = None
            a_col = None
            d_col = None

            for col in df.columns:
                col_lower = col.lower()
                if 'word' in col_lower and word_col is None:
                    word_col = col
                elif ('v.mean' in col_lower or 'valence' in col_lower) and v_col is None:
                    v_col = col
                elif ('a.mean' in col_lower or 'arousal' in col_lower) and a_col is None:
                    a_col = col
                elif ('d.mean' in col_lower or 'dominance' in col_lower) and d_col is None:
                    d_col = col

            if not all([word_col, v_col, a_col, d_col]):
                logger.error(f"Could not find required columns. Available: {list(df.columns)}")
                return {}

            for _, row in df.iterrows():
                try:
                    word = str(row[word_col]).lower().strip()
                    v = float(row[v_col])
                    a = float(row[a_col])
                    d = float(row[d_col])

                    lexicon[word] = {
                        'v': (v - 1) / 8,
                        'a': (a - 1) / 8,
                        'd': (d - 1) / 8
                    }
                except (ValueError, TypeError):
                    continue

            logger.info(f"Loaded Warriner lexicon: {len(lexicon)} words")
        except Exception as e:
            logger.error(f"Failed to load Warriner lexicon: {e}")
        return lexicon

    def _load_memolon_lexicon(self, file_path):
        lexicon = {}
        try:
            df = pd.read_csv(file_path, sep='\t')

            for _, row in df.iterrows():
                try:
                    word = str(row['word']).lower().strip()
                    v = float(row['valence'])
                    a = float(row['arousal'])
                    d = float(row['dominance'])

                    lexicon[word] = {
                        'v': (v - 1) / 8,
                        'a': (a - 1) / 8,
                        'd': (d - 1) / 8
                    }
                except (ValueError, TypeError, KeyError):
                    continue

            logger.info(f"Loaded MEmoLon lexicon: {len(lexicon)} words")
        except Exception as e:
            logger.error(f"Failed to load MEmoLon lexicon: {e}")
        return lexicon

    def _preprocess_text(self, text):
        if not isinstance(text, str):
            return []
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
        except LookupError:
            logger.warning("NLTK stopwords not found")
            pass
        return tokens

    def _calculate_sense_vad(self, definition, lexicon):
        tokens = self._preprocess_text(definition)
        if not tokens:
            return [float(v) for v in DEFAULT_VAD]

        v_sum, a_sum, d_sum, match_count = 0.0, 0.0, 0.0, 0
        for token in tokens:
            if token in lexicon:
                vad = lexicon[token]
                v_sum += vad['v']
                a_sum += vad['a']
                d_sum += vad['d']
                match_count += 1

        if match_count > 0:
            return [v_sum / match_count, a_sum / match_count, d_sum / match_count]
        else:
            return [float(v) for v in DEFAULT_VAD]

    def _load_sense_data(self):
        try:
            with open(self.sense_data_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load sense data: {e}")
            return None

    def _interpolate_series(self, x_original, series_original, target_step_size):
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

            y1 = float(y1_val) if not y1_is_nan else np.nan
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

    def _unify_to_timeline(self, x_series, y_series, global_timeline):
        series_map = {float(x): float(y) for x, y in zip(x_series, y_series)}
        unified_y = []
        for year in global_timeline:
            val = series_map.get(float(year), np.nan)
            unified_y.append(val)
        return unified_y

    def _filter_words_with_too_many_nulls(self, dataset, max_nulls=MAX_NULL_VALUES_PER_DIMENSION):
        filtered = {}
        removed_count = 0

        for word, data in dataset.items():
            if 'temporal_vad' not in data:
                continue

            tv = data['temporal_vad']
            v_vals = tv.get('v', [])
            a_vals = tv.get('a', [])
            d_vals = tv.get('d', [])

            v_nulls = sum(
                1 for val in v_vals if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)))
            a_nulls = sum(
                1 for val in a_vals if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)))
            d_nulls = sum(
                1 for val in d_vals if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)))

            if v_nulls <= max_nulls and a_nulls <= max_nulls and d_nulls <= max_nulls:
                filtered[word] = data
            else:
                removed_count += 1

        logger.info(f"Filtered out {removed_count} words with too many nulls. Remaining: {len(filtered)}")
        return filtered

    def _replace_nan_with_none(self, obj):
        if isinstance(obj, dict):
            return {k: self._replace_nan_with_none(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_nan_with_none(elem) for elem in obj]
        elif isinstance(obj, (float, np.floating)) and np.isnan(obj):
            return None
        return obj

    def _process_sense_data(self, sense_data, lexicon):
        output_dataset = {}
        total_senses = 0
        default_senses = 0

        for word, senses in sense_data.items():
            if not isinstance(senses, dict) or not senses:
                continue

            years_list = None
            num_time_steps = 0

            for sense_info in senses.values():
                if isinstance(sense_info, dict) and 'x' in sense_info and isinstance(sense_info['x'], list) and \
                        sense_info['x']:
                    years_list = [int(y) for y in sense_info['x']]
                    num_time_steps = len(years_list)
                    break

            if years_list is None:
                continue

            # Process each sense
            word_senses = {}
            for sense_id, sense_info in senses.items():
                if isinstance(sense_info, dict) and 'definition' in sense_info:
                    total_senses += 1
                    sense_vad = self._calculate_sense_vad(sense_info['definition'], lexicon)

                    if sense_vad == DEFAULT_VAD:
                        default_senses += 1

                    sense_copy = copy.deepcopy(sense_info)
                    sense_copy['vad'] = sense_vad
                    word_senses[sense_id] = sense_copy

            # Calculate word-level VAD trajectories
            word_v_traj = [0.0] * num_time_steps
            word_a_traj = [0.0] * num_time_steps
            word_d_traj = [0.0] * num_time_steps

            for t_idx in range(num_time_steps):
                v_sum, a_sum, d_sum = 0.0, 0.0, 0.0

                for sense_id, sense_info in word_senses.items():
                    if ('y_fitting' in sense_info and
                            isinstance(sense_info['y_fitting'], list) and
                            len(sense_info['y_fitting']) == num_time_steps):

                        try:
                            proportion = float(sense_info['y_fitting'][t_idx])
                            if math.isnan(proportion):
                                proportion = 0.0
                        except (ValueError, TypeError):
                            proportion = 0.0

                        proportion = max(0.0, proportion)
                        sense_vad = sense_info['vad']
                        v_sum += proportion * sense_vad[0]
                        a_sum += proportion * sense_vad[1]
                        d_sum += proportion * sense_vad[2]

                word_v_traj[t_idx] = v_sum
                word_a_traj[t_idx] = a_sum
                word_d_traj[t_idx] = d_sum

            output_dataset[word] = {
                'temporal_vad': {
                    'x': years_list,
                    'v': word_v_traj,
                    'a': word_a_traj,
                    'd': word_d_traj
                },
                'senses': word_senses
            }

        logger.info(f"Processed {total_senses} senses, {default_senses} used defaults")
        logger.info(f"Generated data for {len(output_dataset)} words")

        return output_dataset

    def _augment_and_unify(self, initial_data):
        timeline = list(range(START_YEAR_UNIFIED, END_YEAR_UNIFIED + 1, TIME_STEP_UNIFIED))
        unified_data = {}

        for word, word_data in initial_data.items():
            tv_data = word_data.get('temporal_vad', {})
            x_orig = tv_data.get('x', [])
            v_orig = tv_data.get('v', [])
            a_orig = tv_data.get('a', [])
            d_orig = tv_data.get('d', [])

            if x_orig and len(x_orig) == len(v_orig) == len(a_orig) == len(d_orig):
                # Interpolate each dimension
                x_interp, v_interp = self._interpolate_series(x_orig, v_orig, TIME_STEP_UNIFIED)
                _, a_interp = self._interpolate_series(x_orig, a_orig, TIME_STEP_UNIFIED)
                _, d_interp = self._interpolate_series(x_orig, d_orig, TIME_STEP_UNIFIED)

                # Unify to global timeline
                v_unified = self._unify_to_timeline(x_interp, v_interp, timeline)
                a_unified = self._unify_to_timeline(x_interp, a_interp, timeline)
                d_unified = self._unify_to_timeline(x_interp, d_interp, timeline)
            else:
                # Fill with NaN if data is incomplete
                v_unified = [np.nan] * len(timeline)
                a_unified = [np.nan] * len(timeline)
                d_unified = [np.nan] * len(timeline)

            unified_data[word] = {
                'temporal_vad': {
                    'x': timeline,
                    'v': v_unified,
                    'a': a_unified,
                    'd': d_unified
                },
                'senses': word_data.get('senses', {})
            }

        return unified_data

    def generate_dataset(self, lexicon_name):
        logger.info(f"Generating dataset with {lexicon_name.upper()} lexicon")

        config = self.lexicon_configs[lexicon_name]
        lexicon_path = config['path']

        if not os.path.exists(lexicon_path):
            logger.error(f"Lexicon file not found: {lexicon_path}")
            return None

        # Load lexicon
        lexicon = config['loader'](lexicon_path)
        if not lexicon:
            logger.error(f"Failed to load {lexicon_name} lexicon")
            return None

        # Load sense data
        sense_data = self._load_sense_data()
        if sense_data is None:
            return None

        # Process sense data
        logger.info("Processing sense data...")
        processed_data = self._process_sense_data(sense_data, lexicon)

        if not processed_data:
            logger.error("Failed to process sense data")
            return None

        # Augment and unify data
        logger.info("Augmenting and unifying to timeline...")
        ml_data = self._augment_and_unify(processed_data)

        # Filter words with too many nulls
        logger.info("Filtering words with too many nulls...")
        ml_data_filtered = self._filter_words_with_too_many_nulls(ml_data)

        # Create output directory
        output_dir = self.output_base_dir / f"dataset_{lexicon_name}"
        output_dir.mkdir(exist_ok=True)

        # Save dataset
        dataset_path = output_dir / f"emotracker_{lexicon_name}.json"
        ml_data_json = self._replace_nan_with_none(ml_data_filtered)

        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(ml_data_json, f, indent=2)

        logger.info(f"Dataset saved: {dataset_path}")
        logger.info(f"Final dataset: {len(ml_data_filtered)} words")

        return {
            'lexicon_name': lexicon_name,
            'lexicon_size': len(lexicon),
            'dataset_size': len(ml_data_filtered),
            'dataset_path': str(dataset_path),
            'coverage': len(ml_data_filtered) / len(lexicon) if len(lexicon) > 0 else 0
        }


def main():
    print("VAD Lexicon Dataset Generation")
    print("=" * 50)

    generator = LexiconDatasetGenerator()

    if not os.path.exists(generator.sense_data_path):
        logger.error(f"Sense data not found: {generator.sense_data_path}")
        return

    results = {}

    for lexicon_name in ['nrc', 'warriner', 'memolon']:
        print(f"\nProcessing {lexicon_name.upper()} lexicon...")

        dataset_result = generator.generate_dataset(lexicon_name)

        if dataset_result:
            results[lexicon_name] = dataset_result
            print(f"✓ {lexicon_name.upper()} dataset generated successfully")
            print(f"  - Lexicon size: {dataset_result['lexicon_size']:,} words")
            print(f"  - Dataset size: {dataset_result['dataset_size']:,} words")
            print(f"  - Coverage: {dataset_result['coverage']:.1%}")
            print(f"  - Saved to: {dataset_result['dataset_path']}")
        else:
            print(f"✗ {lexicon_name.upper()} dataset generation failed")

    # Save summary results
    if results:
        results_path = generator.output_base_dir / "dataset_generation_summary.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n" + "=" * 50)
        print("DATASET GENERATION SUMMARY")
        print("=" * 50)
        print(f"{'Lexicon':<12} {'Lexicon Size':<15} {'Dataset Size':<15} {'Coverage':<10}")
        print("-" * 50)

        for lexicon_name, data in results.items():
            size = data['lexicon_size']
            dataset_size = data['dataset_size']
            coverage = data['coverage']
            print(f"{lexicon_name.upper():<12} {size:<15,} {dataset_size:<15,} {coverage:<10.1%}")

        print(f"\nSummary saved to: {results_path}")
        print(f"Dataset files saved in: {generator.output_base_dir}")


if __name__ == "__main__":
    main()