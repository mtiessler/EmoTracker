import json
import os
import logging
import torch
from models import EnhancedLSTMForecast

# -- Globals
model = None
input_scaler_params_list = None
output_scaler_params_list = None
lookback_window = 0
num_input_features = 0
num_output_features = 0
model_hyperparams = {}
full_vad_data = None
TIME_STEP_YEARS = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# File paths
DATA_DIR = os.path.join('..', 'data')
MODEL_ASSETS_DIR = os.path.join(DATA_DIR, 'model_assets_pytorch')
PYTORCH_MODEL_FILENAME = os.path.join(MODEL_ASSETS_DIR, 'lstm_vad_model_pytorch.pth')
BACKEND_CONFIG_FILENAME = os.path.join(MODEL_ASSETS_DIR, 'backend_model_config_pytorch.json')
script_dir = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.normpath(os.path.join(DATA_DIR, 'Generated_VAD_Dataset', 'ml_ready_temporal_vad_data.json'))

logging.info(f"Attempting to use device: {device}")


def load_resources():
    global model, input_scaler_params_list, output_scaler_params_list, lookback_window
    global num_input_features, num_output_features, model_hyperparams, device, full_vad_data

    try:
        if not os.path.exists(BACKEND_CONFIG_FILENAME):
            logging.error(f"Backend config file not found at {BACKEND_CONFIG_FILENAME}")
            return False
        with open(BACKEND_CONFIG_FILENAME, 'r') as f:
            backend_config = json.load(f)
            lookback_window = backend_config.get('lookback_window', 10)
            num_input_features = backend_config.get('num_input_features', 6)
            num_output_features = backend_config.get('num_output_features', 3)
            input_scaler_params_list = backend_config.get('input_scalers_params')
            output_scaler_params_list = backend_config.get('output_scalers_params')
            model_hyperparams = backend_config.get('model_params', {})

        if not input_scaler_params_list or len(input_scaler_params_list) != num_input_features:
            logging.error(
                f"Input scaler parameters missing or mismatched. Expected {num_input_features}, "
                f"got {len(input_scaler_params_list) if input_scaler_params_list else 0}.")
            return False

        if not output_scaler_params_list or len(output_scaler_params_list) != num_output_features:
            logging.error(
                f"Output scaler parameters missing or mismatched. Expected {num_output_features}, "
                f"got {len(output_scaler_params_list) if output_scaler_params_list else 0}.")
            return False

        if not model_hyperparams:
            logging.error("Model hyperparameters missing in backend config.")
            return False

        logging.info(
            f"Backend config loaded: Lookback={lookback_window}, "
            f"InputFeatures={num_input_features}, OutputFeatures={num_output_features}")

        if not os.path.exists(DATASET_PATH):
            logging.error(f"Main VAD data file not found at {DATASET_PATH}")
            return False

        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            full_vad_data = json.load(f)
        logging.info(f"Main VAD data loaded successfully from {DATASET_PATH}")

        if not os.path.exists(PYTORCH_MODEL_FILENAME):
            logging.error(f"LSTM model file not found at {PYTORCH_MODEL_FILENAME}")
            return False

        model_instance = EnhancedLSTMForecast(
            input_size=model_hyperparams.get('input_size', num_input_features),
            hidden_size=model_hyperparams.get('hidden_size', 128),
            num_layers=model_hyperparams.get('num_layers', 2),
            output_size=model_hyperparams.get('output_size', num_output_features),
            dropout_prob=model_hyperparams.get('dropout_prob', 0.1)
        )

        model_instance.load_state_dict(torch.load(PYTORCH_MODEL_FILENAME, map_location=device))
        model_instance.to(device)
        model_instance.eval()
        model = model_instance
        logging.info(f"LSTM model loaded successfully! Moved it to => {device}.")

        dummy_input = torch.randn(1, lookback_window, num_input_features, dtype=torch.float32).to(device)
        with torch.no_grad():
            _ = model(dummy_input)
        logging.info("LSTM Model warmed up! :).")
        return True

    except Exception as e:
        logging.error(f"Error loading PyTorch resources: {e}", exc_info=True)
        model = None
        input_scaler_params_list = None
        output_scaler_params_list = None
        full_vad_data = None
        return False


resources_loaded_pytorch = load_resources()