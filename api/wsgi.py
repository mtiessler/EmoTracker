from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import config
from prediction import predict_vad_trajectory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict_pytorch():
    if not config.resources_loaded_pytorch:
        return jsonify({'error': 'API resources not loaded. Check backend logs.'}), 500

    try:
        data = request.get_json()
        word_to_predict = data.get('word')
        predict_from_year = int(data.get('predict_from_year'))
        predict_until_year = int(data.get('predict_until_year'))

        predictions = predict_vad_trajectory(word_to_predict, predict_from_year, predict_until_year)

        return jsonify({'predictions': predictions})

    except Exception as e:
        app.logger.error(f"PyTorch error: {e}", exc_info=True)
        return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500


if __name__ == '__main__':
    if not config.resources_loaded_pytorch:
        logging.error("Failed to load PyTorch ML resources. Backend might not work correctly.")
    app.run(debug=True, port=5000)