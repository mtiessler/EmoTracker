from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import torch.nn as nn
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

MODEL_ASSETS_DIR = 'model_assets_pytorch'
PYTORCH_MODEL_FILENAME = os.path.join(MODEL_ASSETS_DIR, 'lstm_vad_model_pytorch.pth')
BACKEND_CONFIG_FILENAME = os.path.join(MODEL_ASSETS_DIR, 'backend_model_config_pytorch.json')
script_dir = os.path.dirname(os.path.abspath(__file__))
ML_READY_DATA_PATH = os.path.normpath(
    os.path.join(script_dir, 'data', 'Generated_VAD_Dataset', 'ml_ready_temporal_vad_data.json'))

class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.0):
        super(LSTMForecast, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout_prob if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = None
scaler_params_list = None
lookback_window = 0
num_features = 0
model_hyperparams = {}
full_vad_data = None
TIME_STEP_YEARS = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Attempting to use device: {device}")

def load_resources_pytorch():
    global model, scaler_params_list, lookback_window, num_features, model_hyperparams, device, full_vad_data

    try:
        if not os.path.exists(BACKEND_CONFIG_FILENAME):
            logging.error(f"Backend config file not found at {BACKEND_CONFIG_FILENAME}")
            return False
        with open(BACKEND_CONFIG_FILENAME, 'r') as f:
            backend_config = json.load(f)
            lookback_window = backend_config.get('lookback_window', 4)
            num_features = backend_config.get('num_features', 3)
            scaler_params_list = backend_config.get('scalers_params')
            model_hyperparams = backend_config.get('model_params', {})

        if not scaler_params_list or len(scaler_params_list) != num_features:
            logging.error("Scaler parameters missing or mismatched in backend config.")
            return False
        if not model_hyperparams:
            logging.error("Model hyperparameters missing in backend config.")
            return False
        logging.info(f"Backend config loaded: Lookback={lookback_window}, Features={num_features}")

        if not os.path.exists(ML_READY_DATA_PATH):
            logging.error(f"Main VAD data file not found at {ML_READY_DATA_PATH}")
            return False
        with open(ML_READY_DATA_PATH, 'r', encoding='utf-8') as f:
            full_vad_data = json.load(f)
        logging.info(f"Main VAD data loaded successfully from {ML_READY_DATA_PATH}")

        if not os.path.exists(PYTORCH_MODEL_FILENAME):
            logging.error(f"PyTorch model file not found at {PYTORCH_MODEL_FILENAME}")
            return False
        model_instance = LSTMForecast(
            input_size=model_hyperparams.get('input_size', num_features),
            hidden_size=model_hyperparams.get('hidden_size', 50),
            num_layers=model_hyperparams.get('num_layers', 1),
            output_size=model_hyperparams.get('output_size', num_features),
            dropout_prob=model_hyperparams.get('dropout_prob', 0.0)
        )
        model_instance.load_state_dict(torch.load(PYTORCH_MODEL_FILENAME, map_location=device))
        model_instance.to(device)
        model_instance.eval()
        model = model_instance
        logging.info(f"PyTorch model loaded successfully and moved to {device}.")

        dummy_input = torch.randn(1, lookback_window, num_features, dtype=torch.float32).to(device)
        with torch.no_grad():
            _ = model(dummy_input)
        logging.info("PyTorch model warmed up.")
        return True

    except Exception as e:
        logging.error(f"Error loading PyTorch resources: {e}", exc_info=True)
        model = None
        scaler_params_list = None
        full_vad_data = None
        return False

resources_loaded_pytorch = load_resources_pytorch()

@app.route('/predict', methods=['POST'])
def predict_pytorch():
    global model, scaler_params_list, lookback_window, num_features, device, TIME_STEP_YEARS, full_vad_data

    if not resources_loaded_pytorch or model is None or scaler_params_list is None or full_vad_data is None:
        return jsonify({'error': 'API resources not loaded. Check backend logs.'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided in request.'}), 400

        word_to_predict = data.get('word')
        predict_from_year = data.get('predict_from_year')
        predict_until_year = data.get('predict_until_year')

        if not word_to_predict or predict_from_year is None or predict_until_year is None:
            return jsonify(
                {'error': 'Missing "word", "predict_from_year", or "predict_until_year" in request data'}), 400

        if word_to_predict not in full_vad_data or "temporal_vad" not in full_vad_data[word_to_predict]:
            return jsonify(
                {'error': f"Word '{word_to_predict}' not found in dataset or has no temporal_vad data."}), 404

        word_data = full_vad_data[word_to_predict]["temporal_vad"]
        x_list = word_data["x"]
        v_list = [np.nan if x is None else float(x) for x in word_data['v']]
        a_list = [np.nan if x is None else float(x) for x in word_data['a']]
        d_list = [np.nan if x is None else float(x) for x in word_data['d']]

        try:
            predict_from_year = int(predict_from_year)
            predict_until_year = int(predict_until_year)
        except ValueError:
            return jsonify({'error': '"predict_from_year" and "predict_until_year" must be integers.'}), 400

        if predict_from_year not in x_list:
            return jsonify(
                {'error': f"Year {predict_from_year} not found in trajectory for word '{word_to_predict}'."}), 400

        target_index = x_list.index(predict_from_year)

        if target_index < lookback_window - 1:
            return jsonify({
                               'error': f"Not enough historical data before {predict_from_year} for word '{word_to_predict}' to form a lookback window of {lookback_window}."}), 400

        initial_sequence_raw = []
        for i in range(lookback_window):
            idx = target_index - (lookback_window - 1) + i
            v = v_list[idx]
            a = a_list[idx]
            d = d_list[idx]
            if np.isnan(v) or np.isnan(a) or np.isnan(d):
                return jsonify({
                                   'error': f"NaN value encountered in lookback sequence for {word_to_predict} at year {x_list[idx]}."}), 400
            initial_sequence_raw.append([v, a, d])

        current_sequence_np = np.array(initial_sequence_raw, dtype=np.float32)

        all_predictions_output = []
        current_year = predict_from_year

        current_sequence_for_model_scaled = np.zeros_like(current_sequence_np)
        for i in range(num_features):
            mean = scaler_params_list[i]['mean']
            scale = scaler_params_list[i]['scale']
            if scale == 0:
                current_sequence_for_model_scaled[:, i] = 0
            else:
                current_sequence_for_model_scaled[:, i] = (current_sequence_np[:, i] - mean) / scale

        while current_year < predict_until_year:
            input_tensor = torch.from_numpy(current_sequence_for_model_scaled).unsqueeze(0).to(device)

            with torch.no_grad():
                scaled_prediction_tensor = model(input_tensor)

            next_step_scaled_prediction_np = scaled_prediction_tensor.cpu().numpy()[0]

            unscaled_prediction_for_output = np.zeros(num_features, dtype=np.float32)
            for i in range(num_features):
                mean = scaler_params_list[i]['mean']
                scale = scaler_params_list[i]['scale']
                unscaled_prediction_for_output[i] = (next_step_scaled_prediction_np[i] * scale) + mean

            current_year += TIME_STEP_YEARS

            all_predictions_output.append({
                'time': current_year,
                'v': float(unscaled_prediction_for_output[0]),
                'a': float(unscaled_prediction_for_output[1]),
                'd': float(unscaled_prediction_for_output[2])
            })

            new_row_scaled = next_step_scaled_prediction_np.reshape(1, num_features)
            current_sequence_for_model_scaled = np.vstack(
                (current_sequence_for_model_scaled[1:], new_row_scaled)
            )

            if len(all_predictions_output) > (
                    (predict_until_year - predict_from_year) / TIME_STEP_YEARS + lookback_window + 5):
                logging.warning("Iterative prediction loop exceeded a safe number of steps. Breaking.")
                break

        logging.info(
            f"PyTorch iterative prediction successful for '{word_to_predict}'. Predicted {len(all_predictions_output)} steps up to year {current_year}.")
        return jsonify({'predictions': all_predictions_output})

    except Exception as e:
        app.logger.error(f"PyTorch Iterative Prediction error: {e}", exc_info=True)
        return jsonify({'error': 'An internal error occurred during PyTorch iterative prediction.'}), 500

if __name__ == '__main__':
    if not resources_loaded_pytorch:
        logging.error("Failed to load PyTorch ML resources. Backend might not work correctly.")
    app.run(debug=True, port=5000)