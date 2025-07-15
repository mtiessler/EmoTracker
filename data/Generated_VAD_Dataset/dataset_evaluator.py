import json
import os
import numpy as np

ML_DATA_FILE_PATH = os.path.join('dataset_nrc', 'emotracker_nrc.json')
LOOKBACK_WINDOW = 15 # todo sync with config


def analyze_dataset_size(file_path, lookback_window):
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at '{file_path}'")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error: Could not read or parse JSON file: {e}")
        return None

    if not data:
        print("Error: The data file is empty.")
        return None

    num_words = len(data)
    first_word = next(iter(data))

    try:
        timeline = data[first_word]['temporal_vad']['x']
        num_timesteps = len(timeline)
        start_year, end_year = timeline[0], timeline[-1]
    except (KeyError, IndexError):
        print("Error: Cannot determine timeline from the data file. It may be malformed.")
        return None

    total_possible_vad_points = num_words * num_timesteps
    valid_training_samples = 0
    words_with_no_samples = []

    required_sequence_length = lookback_window + 1

    for word, word_data in data.items():
        samples_for_this_word = 0
        try:
            v = [val if val is not None else np.nan for val in word_data['temporal_vad']['v']]
            a = [val if val is not None else np.nan for val in word_data['temporal_vad']['a']]
            d = [val if val is not None else np.nan for val in word_data['temporal_vad']['d']]

            if len(v) < required_sequence_length:
                continue

            for i in range(num_timesteps - required_sequence_length + 1):
                v_window = v[i: i + required_sequence_length]
                a_window = a[i: i + required_sequence_length]
                d_window = d[i: i + required_sequence_length]

                is_valid = not (np.isnan(v_window).any() or \
                                np.isnan(a_window).any() or \
                                np.isnan(d_window).any())

                if is_valid:
                    samples_for_this_word += 1

        except (KeyError, TypeError):
            continue

        if samples_for_this_word == 0:
            words_with_no_samples.append(word)

        valid_training_samples += samples_for_this_word

    return {
        "num_words": num_words,
        "num_timesteps": num_timesteps,
        "start_year": start_year,
        "end_year": end_year,
        "total_possible_vad_points": total_possible_vad_points,
        "lookback_window": lookback_window,
        "valid_training_samples": valid_training_samples,
        "words_with_no_samples": len(words_with_no_samples)
    }


def main():
    print("Analyzing ML Dataset Size...")
    print("=" * 50)

    results = analyze_dataset_size(ML_DATA_FILE_PATH, LOOKBACK_WINDOW)

    if results:
        print("\n--- DATASET DIMENSIONS ---")
        print(f"Unique Words: {results['num_words']}")
        print(f"Timeline: {results['start_year']}-{results['end_year']}")
        print(f"Timesteps: {results['num_timesteps']} (5-year granularity)")
        print(f"Total VAD Data Points (words x timesteps): {results['total_possible_vad_points']:,}")

        print("\n--- TRAINING SAMPLE ANALYSIS ---")
        print(f"Model Lookback Window: {results['lookback_window']} timesteps (75 years)")
        print("A single training sample requires a continuous sequence of")
        print(f"{results['lookback_window'] + 1} non-null VAD vectors.")

        print("\n--- SUMMARY ---")
        print(f"Total Valid Training Samples Available: {results['valid_training_samples']:,}")
        print(f"Number of Words with No Valid Samples: {results['words_with_no_samples']}")
        print("=" * 50)


if __name__ == "__main__":
    main()