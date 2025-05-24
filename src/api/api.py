from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import pickle
import os
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

MODEL_ASSETS_DIR = 'model_assets'
KERAS_MODEL_FILENAME = os.path.join(MODEL_ASSETS_DIR, 'lstm_vad_model.keras')
SCALERS_PICKLE_FILENAME = os.path.join(MODEL_ASSETS_DIR, 'vad_scalers.pkl')
BACKEND_CONFIG_FILENAME = os.path.join(MODEL_ASSETS_DIR, 'backend_model_config.json')

model = None
scalers = None
lookback_window = 0
num_features = 0


def load_resources():
    global model, scalers, lookback_window, num_features
    try:
        if not os.path.exists(KERAS_MODEL_FILENAME):
            logging.error(f"Keras model file not found at {KERAS_MODEL_FILENAME}")
            return False
        if not os.path.exists(SCALERS_PICKLE_FILENAME):
            logging.error(f"Scalers file not found at {SCALERS_PICKLE_FILENAME}")
            return False
        if not os.path.exists(BACKEND_CONFIG_FILENAME):
            logging.error(f"Backend config file not found at {BACKEND_CONFIG_FILENAME}")
            return False

        model = tf.keras.models.load_model(KERAS_MODEL_FILENAME)
        logging.info("Keras model loaded successfully.")

        with open(SCALERS_PICKLE_FILENAME, 'rb') as f:
            scalers = pickle.load(f)
        logging.info("Scalers loaded successfully.")

        with open(BACKEND_CONFIG_FILENAME, 'r') as f:
            backend_config = json.load(f)
            lookback_window = backend_config.get('lookback_window', 4)
            num_features = backend_config.get('num_features', 3)
        logging.info(f"Backend config loaded: Lookback={lookback_window}, Features={num_features}")

        dummy_input = np.random.rand(1, lookback_window, num_features).astype(np.float32)
        _ = model.predict(dummy_input)
        logging.info("Model warmed up with a dummy prediction.")
        return True

    except Exception as e:
        logging.error(f"Error loading resources: {e}")
        model = None
        scalers = None
        return False


resources_loaded = load_resources()


@app.route('/predict', methods=['POST'])
def predict():
    global model, scalers, lookback_window, num_features

    if not resources_loaded or model is None or scalers is None:
        return jsonify({'error': 'Model or scalers not loaded. Check backend logs.'}), 500

    try:
        data = request.get_json()
        if not data or 'sequence' not in data:
            return jsonify({'error': 'Missing "sequence" in request data'}), 400

        input_sequence_raw = data['sequence']

        if not isinstance(input_sequence_raw, list) or len(input_sequence_raw) != lookback_window:
            return jsonify({'error': f'Input "sequence" must be a list of length {lookback_window}'}), 400

        for item in input_sequence_raw:
            if not isinstance(item, list) or len(item) != num_features:
                return jsonify(
                    {'error': f'Each item in the sequence must be a list of {num_features} numbers (V, A, D)'}), 400
            for val in item:
                if not isinstance(val, (int, float)):
                    return jsonify({'error': 'All VAD values in the sequence must be numbers'}), 400

        input_sequence_np = np.array(input_sequence_raw, dtype=float)

        if input_sequence_np.shape != (lookback_window, num_features):
            return jsonify({
                               'error': f'Sequence shape is incorrect. Expected ({lookback_window}, {num_features}), got {input_sequence_np.shape}'}), 400

        scaled_input_sequence = np.zeros_like(input_sequence_np)
        for i in range(num_features):
            feature_column = input_sequence_np[:, i].reshape(-1, 1)
            scaled_input_sequence[:, i] = scalers[i].transform(feature_column).flatten()

        input_tensor = scaled_input_sequence.reshape(1, lookback_window, num_features)
        scaled_prediction = model.predict(input_tensor)[0]

        unscaled_prediction = np.zeros(num_features)
        for i in range(num_features):
            unscaled_prediction[i] = scalers[i].inverse_transform(scaled_prediction[i].reshape(-1, 1)).flatten()[0]

        logging.info(f"Prediction successful for input: {str(input_sequence_raw[:1])[:100]}...")
        return jsonify({
            'v': unscaled_prediction[0],
            'a': unscaled_prediction[1],
            'd': unscaled_prediction[2]
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': 'An internal error occurred during prediction.'}), 500


if __name__ == '__main__':
    if not resources_loaded:
        logging.error("Failed to load ML resources. Backend might not work correctly.")
    app.run(debug=True, port=5000)