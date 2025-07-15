# EmoTracker VAD Prediction API

EmoTracker REST API for predicting VAD (Valence, Arousal, Dominance) trajectories of words over time using a PyTorch LSTM model with attention mechanism and momentum-based features.

## Architecture

The API uses an enhanced LSTM model with:
- Multi-head attention mechanism for better temporal understanding
- Momentum-based features (8 features per VAD dimension: velocity, acceleration, trend strength, volatility, etc.)
- Iterative prediction for multi-step forecasting
- Feature scaling for stable training and prediction

## Project Structure

```
api/
├── __init__.py
├── api_request_example.http    # Example API requests
├── config.py                   # Configuration and resource loading
├── features.py                 # Momentum feature engineering
├── models.py                   # PyTorch model definitions
├── prediction.py               # Core prediction logic
├── requirements.txt            # Python dependencies
├── wsgi.py                     # WSGI entry point
├── Dockerfile                  # Docker configuration
└── forecasting_evaluation_results/
    └── __init__.py
```

## Setup and Installation

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

# Run the API
python wsgi.py
```

The API will be available at `http://localhost:5000`

## Dependencies

```
flask
flask-cors
numpy
torch
scipy
```

## Required Data Structure

The system expects the following data files in the project root:

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

## API Reference

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

### Example Request

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


## File Descriptions

### Core Files

- **`wsgi.py`**: Flask application entry point with CORS enabled
- **`config.py`**: Loads model assets, datasets, and configuration
- **`models.py`**: PyTorch LSTM model with attention mechanism
- **`prediction.py`**: Core prediction logic with iterative forecasting
- **`features.py`**: Momentum feature engineering (velocity, acceleration, volatility)

### Configuration Files

- **`requirements.txt`**: Python dependencies
- **`Dockerfile`**: Container configuration with Python 3.9 slim base
- **`api_request_example.http`**: Example API requests for testing

## Development and Testing

### Running Locally
```bash
cd api
python wsgi.py
```

### Testing the API
Use the provided example in `api_request_example.http` or curl:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"word": "love", "predict_from_year": 2020, "predict_until_year": 2025}'
```

## Reproducing Analysis Results

### Running Empirical Evaluation

To reproduce the forecasting analysis and generate performance metrics:

```bash
cd api
python forecasting_empirical_evaluation.py
```

This script performs comprehensive evaluation of the model's forecasting performance and generates detailed analysis reports.

### Evaluation Process

The evaluation script:
1. **Loads all available words** from the historical dataset
2. **Performs 30-year forecasting** (1980 → 2010) for each word
3. **Calculates performance metrics** including MAE and RMSE for each VAD dimension
4. **Generates visualizations** and detailed performance reports
5. **Creates categorical performance analysis** (excellent, good, fair, poor)

### Generated Output Files

The script creates a `forecasting_evaluation_results/` directory containing:

#### Performance Data
- **`word_performance_results.csv`**: Raw performance metrics for all words
- **`error_log.txt`**: Words that failed prediction with error details
- **`performance_summary.txt`**: Comprehensive statistical summary

#### Analysis Reports
- **`analysis_report.txt`**: Detailed performance breakdown by dimension
- Top/worst performing words ranked by MAE and RMSE
- Statistical distributions and performance categories
- Dimension-specific correlation analysis

#### Visualizations
- **`1_mae_distribution_simple.png`**: Overall MAE distribution
- **`1_rmse_distribution_simple.png`**: Overall RMSE distribution
- **`2_mae_by_dimension.png`**: MAE performance by VAD dimension
- **`3a_best_performers_mae.png`**: Top 15 best performing words (MAE)
- **`3b_best_performers_rmse.png`**: Top 15 best performing words (RMSE)
- **`4a_worst_performers_mae.png`**: Top 15 worst performing words (MAE)
- **`4b_worst_performers_rmse.png`**: Top 15 worst performing words (RMSE)

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

### Environment Variables
- `FLASK_ENV`: Set to `production` for production deployment
- `PYTHONPATH`: Set to project root for proper imports

### Common Issues

1. **Model loading errors**: Ensure all required data files are present in `data/` directory
2. **Import errors**: Check that `PYTHONPATH` includes the project root
3. **Memory issues**: Reduce prediction horizon for long trajectories
4. **Word not found**: Verify the word exists in the historical dataset
