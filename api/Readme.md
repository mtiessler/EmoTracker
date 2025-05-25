# EmoTracker - VAD Prediction API

A Flask-based REST API for predicting VAD (Valence, Arousal, Dominance) trajectories of words over time using a PyTorch LSTM model with attention mechanism and momentum-based features.

## Architecture

The API uses an enhanced LSTM model with:
- **Multi-head attention mechanism** for better temporal understanding
- **Momentum-based features** (8 features per VAD dimension: velocity, acceleration, trend strength, volatility, etc.)
- **Iterative prediction** for multi-step forecasting
- **Feature scaling** for stable training and prediction

## Project Structure

```
EmoTracker/
├── api/
│   ├── __init__.py
│   ├── api_request_example.http    # Example API requests
│   ├── config.py                   # Configuration and resource loading
│   ├── features.py                 # Momentum feature engineering
│   ├── models.py                   # PyTorch model definitions
│   ├── prediction.py               # Core prediction logic
│   ├── requirements.txt            # Python dependencies
│   ├── wsgi.py                     # WSGI entry point
│   └── README.md                   # This documentation
└── data/
    ├── Generated_VAD_Dataset/
    │   ├── ml_ready_temporal_vad_data.json          # Historical VAD data
    │   └── temporal_vad_with_full_sense_data.json   # Extended VAD dataset
    ├── model_assets_pytorch/
    │   ├── backend_model_config_pytorch.json        # Model config & scalers
    │   ├── lstm_vad_model_pytorch.pth              # Trained model weights
    │   ├── vad_scalers_input_pytorch.pkl           # Input feature scalers
    │   └── vad_scalers_output_pytorch.pkl          # Output feature scalers
    ├── Diachronic_Sense_Modeling/                  # Sense modeling data
    ├── imgs/                                       # Images and visualizations
    └── VAD_Lexicons/                              # VAD lexicon resources
```

## Quick Start

### Option 1: Docker (Recommended)

From the project root directory:

```bash
# Build the Docker image
docker build -t emotracker-api .

# Run the container
docker run -p 5000:5000 emotracker-api
```

### Option 2: Local Installation

From the project root directory:

```bash
# Install dependencies
cd api
pip install -r requirements.txt

# run
python wsgi.py
```

The API will be available at `http://localhost:5000`

## Required Data Structure

The system expects the following data files:

### Model Assets (`data/model_assets_pytorch/`)
- **`lstm_vad_model_pytorch.pth`**: PyTorch model state dict
- **`backend_model_config_pytorch.json`**: Configuration containing:
  - `lookback_window`: LSTM sequence length
  - `num_input_features`: Feature count (diffs + momentum)
  - `num_output_features`: Output count (3 for V,A,D diffs)
  - `input_scalers_params`: Feature scaling parameters
  - `output_scalers_params`: Output scaling parameters
  - `model_params`: Model hyperparameters
- **`vad_scalers_input_pytorch.pkl`**: Input feature scalers (optional)
- **`vad_scalers_output_pytorch.pkl`**: Output feature scalers (optional)

### Dataset (`data/Generated_VAD_Dataset/`)
- **`ml_ready_temporal_vad_data.json`**: Historical word VAD trajectories
- **`temporal_vad_with_full_sense_data.json`**: Extended dataset with sense information

## 🔌 API Reference

### POST `/predict`

Predict VAD trajectory for a word over a specified time period.

**Request Body:**
```json
{
    "word": "alien",
    "predict_from_year": 2020,
    "predict_until_year": 2030
}
```

**Success Response (200):**
```json
{
    "predictions": [
        {
            "time": 2025,
            "v": 0.745,
            "a": 0.623,
            "d": 0.812
        },
        {
            "time": 2030,
            "v": 0.739,
            "a": 0.631,
            "d": 0.808
        }
    ]
}
```

**Error Responses:**
- `404 Not Found`: Word not found in dataset
- `400 Bad Request`: Invalid year range or insufficient historical data
- `500 Internal Server Error`: Model loading or prediction error

### Example Request (api/api_request_example.http)
```http
POST http://localhost:5000/predict
Content-Type: application/json

{
    "word": "love",
    "predict_from_year": 2020,
    "predict_until_year": 2025
}
```

## Configuration

### Default Model Parameters
- **Input Features**: 27 (3 VAD differences + 24 momentum features)
- **Hidden Size**: 128
- **LSTM Layers**: 2
- **Attention Heads**: 8
- **Dropout Rate**: 0.1
- **Lookback Window**: 10 timesteps
- **Time Step**: 5 years per prediction
- **Momentum Window**: 5-10 timesteps for feature calculation

### Feature Scaling
- Input and output features use StandardScaler normalization
- Scaler parameters stored in `backend_model_config_pytorch.json`
- Separate scalers for input features and output differences

## 🔧 Development & Testing

### Running the API Locally
```bash
cd api
python wsgi.py 
```

## Performance Metrics

### Response Times (typical)
- **Single prediction**: 50-200ms
- **Short trajectory** (2-3 steps): 100-300ms  
- **Long trajectory** (10+ steps): 500ms-2s

### Memory Usage
- **Base model loading**: 10-50MB
- **Per prediction**: 1-5MB additional
- **Scales linearly** with prediction horizon

### Accuracy Considerations
- **Best performance**: 5-20 year predictions
- **Accuracy decreases** with longer horizons due to error accumulation
- **Historical data quality** significantly impacts predictions

## Deployment

### Production Deployment
```bash
# Using Docker
docker build -t emotracker-api .
docker run -d -p 5000:5000 --name vad-api emotracker-api

# Using gunicorn (from api/ directory)
pip install gunicorn
gunicorn --bind 0.0.0.0:5000 wsgi:app
```
