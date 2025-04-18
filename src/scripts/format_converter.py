import pickle
import json
import sys
import os

# --- Configuration Constants ---
RELATIVE_DATA_DIR_COMPONENTS = ['..', '..', 'data', 'Diachronic_Sense_Modeling']
PICKLE_FILENAME = 'prob_fitting_10.data'
JSON_FILENAME = 'prob_fitting_10.json'

def get_file_paths():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    data_dir_relative_path = os.path.join(*RELATIVE_DATA_DIR_COMPONENTS)
    data_dir_absolute = os.path.join(script_dir, data_dir_relative_path)
    data_dir_absolute = os.path.normpath(data_dir_absolute)
    pickle_path = os.path.join(data_dir_absolute, PICKLE_FILENAME)
    json_path = os.path.join(data_dir_absolute, JSON_FILENAME)

    return pickle_path, json_path


def convert_pickle_to_json(pickle_path, json_path):
    """
    Reads data from a Python pickle file and writes it to a JSON file.
    Logs errors to stderr, but is otherwise silent.

    Args:
        pickle_path (str): The full, platform-agnostic path to the input pickle file.
        json_path (str): The full, platform-agnostic path to the output JSON file.
    """
    data = None
    try:
        with open(pickle_path, 'rb') as infile:
            data = pickle.load(infile)
    except FileNotFoundError:
        print(f"Error: Pickle file not found at '{pickle_path}'.", file=sys.stderr)
        return
    except pickle.UnpicklingError as e:
        print(f"Error: Could not unpickle data from '{pickle_path}'. File might be corrupted.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error: An unexpected error occurred while reading '{pickle_path}': {e}", file=sys.stderr)
        return

    output_dir = os.path.dirname(json_path)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            print(f"Error: Could not create output directory '{output_dir}'. Check permissions.", file=sys.stderr)
            print(f"Details: {e}", file=sys.stderr)
            return # Stop if cannot create directory

    try:
        with open(json_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
    except IOError as e:
        print(f"Error: Could not write to JSON file '{json_path}'. Check permissions or disk space.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
    except TypeError as e:
         print(f"Error: Data structure contains types not serializable to JSON.", file=sys.stderr)
         print(f"Details: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error: An unexpected error occurred while writing '{json_path}': {e}", file=sys.stderr)


if __name__ == "__main__":
    pickle_file_path, json_file_path = get_file_paths()
    convert_pickle_to_json(pickle_file_path, json_file_path)