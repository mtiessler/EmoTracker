# EmoTracker - Temporal VAD Emotion Tracking System

> "The emotions are often the masters of reason."  
> — *Sigmund Freud, paraphrasing from* **The Ego and the Id** *(1923)*

EmoTracker is an advanced framework for modeling how emotional associations of words (represented by Valence, Arousal, and Dominance (VAD)) evolve over time and forecast their evolution.

Unlike traditional emotion lexicons that treat word affect as static, EmoTracker combines **sense-aware temporal embeddings** with the **NRC-VAD lexicon** to infer diachronic emotional trajectories for English words. 

It also uses a **LSTM architecture** with **advanced momentum-based feature engineering** and **multi-head attention mechanisms** to predict diachronic emotional trajectories for English words.

---

## Key Features

* **LSTM with Advanced Momentum Tracking**: 8 sophisticated momentum features per VAD dimension capturing velocity, acceleration, volatility, and trend patterns
* **Interactive Visualization Dashboard**: React-based platform for exploring temporal VAD trajectories
* **Automated Dataset Generation**: Pipeline for creating diachronic VAD datasets from sense modeling data
* **Multi-dimensional Analysis**: Support for 2D, 3D, and 4D VAD visualizations with forecasting capabilities

---

## Motivation

Words like *gay*, *virus*, *abandon*, and *liberal* have undergone emotional and semantic shifts over time. Existing resources provide static affective values, but EmoTracker models dynamic emotional evolution:

```
VAD(w, t+Δt) = LSTM(momentum_features(VAD_history(w, t-n:t)))
```

Where momentum features include 
- velocity 
- acceleration 
- trend strength 
- volatility
- temporal oscillators.

---

## LSTM Architecture

### Advanced Momentum Feature Engineering

EmoTracker Model uses **27 input features** per timestep, combining base VAD differences with momentum tracking:

**Base Features (3)**:
- Δv, Δa, Δd (VAD difference values)

**Advanced Momentum Features (24)**: 8 metrics × 3 VAD dimensions
- **Velocity**: Linear regression slope indicating trend direction and speed
- **Acceleration**: Second derivative capturing rate of change in velocity  
- **Trend Strength × Direction**: R-value weighted by trend direction for consistency
- **Volatility**: Standard deviation measuring uncertainty and variability
- **Momentum Oscillator**: Recent change relative to historical volatility
- **Relative Strength**: First vs second half comparison within sliding window
- **Range Position**: Current value position within historical min/max range
- **EMA Ratio**: Exponential vs Simple Moving Average relationship

### Neural Architecture Components

**LSTM Core**:
```
EnhancedLSTMForecast(
  (input_projection): Linear(in_features=27, out_features=128, bias=True)
  (lstm): LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.2)
  (attention): MultiheadAttention(8 heads, embed_dim=128)
  (layer_norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  (layer_norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  (fc1): Linear(in_features=128, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=3, bias=True)
  (dropout): Dropout(p=0.2, inplace=False)
  (activation): GELU(approximate='none')
)
```

**Training Pipeline**:
- **Difference-based modeling**: `VAD_pred(t+1) = VAD_actual(t) + Δ_pred(t+1)`
- **Lookback Window**: 15 timesteps for temporal context
- **Optimizer**: AdamW with weight decay and learning rate scheduling
- **Regularization**: Dropout (0.2), gradient clipping, early stopping

### Architecture Overview

**![LSTM Architecture](data/imgs/lstm.png)**

*Figure 1: LSTM architecture with momentum feature engineering, multi-head attention, and residual connections for temporal VAD prediction.*

---

## Project Structure

```
EmoTracker/
│
├── api/                     # Flask API Backend
│   ├── config.py           # Resource loading and model configuration
│   ├── features.py         # Advanced momentum feature engineering
│   ├── models.py           # LSTM model definition
│   ├── prediction.py       # Iterative VAD trajectory prediction
│   └── wsgi.py            # Flask web server and API endpoints
│
├── src/
│   ├── dataset/            # Dataset Generation Pipeline
│   │   ├── dataset_generation.py  # VAD dataset creation from sense data
│   │   └── format_converter.py    # Pickle to JSON conversion utilities
│   │
│   └── model/              # LSTM Training Pipeline
│       ├── config.py       # Training hyperparameters and paths
│       ├── dataset.py      # PyTorch dataset wrapper
│       ├── main.py         # Training orchestration
│       ├── model.py        # LSTM architecture
│       ├── preprocessing.py # Feature engineering and data preparation
│       ├── trainer.py      # Training loop with validation and metrics
│       └── utils.py        # Utility functions and helpers
│
├── client/                 # React Visualization Dashboard
│   └── src/               # Interactive VAD trajectory visualizations
│
├── data/
│   ├── Generated_VAD_Dataset/     # ML-ready temporal VAD data
│   ├── model_assets_pytorch/      # Trained models and configurations
│   └── [raw sense modeling data] # Input datasets
│
└── requirements.txt
```

*Figure 2: EmoTracker project organization showing API backend, model training pipeline, and visualization dashboard components.*

---

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Temporal VAD Dataset

```bash
cd src/dataset/
python dataset_generation.py
```

This creates `ml_ready_temporal_vad_data.json` with unified temporal VAD trajectories.

### 3. Train LSTM Model

```bash
cd src/model/
python main.py
```

Trains the LSTM with momentum features and saves model assets to `data/model_assets_pytorch/`.

### 4. Launch Prediction API

```bash
cd api/
python wsgi.py
```

Starts Flask API server on `http://localhost:5000` with `/predict` endpoint.

### 5. Start Visualization Dashboard

```bash
cd client/
npm install && npm start
```

Launches React dashboard for interactive VAD trajectory exploration.

---

## API Usage

### Predict VAD Trajectory

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "word": "abandon",
    "predict_from_year": 2010,
    "predict_until_year": 2040
  }'
```

**Response**:
```json
{
  "predictions": [
    {
      "time": 2015,
      "v": 0.284,
      "a": 0.156,
      "d": 0.239
    },
    ...
  ]
}
```

---

## Visualization Dashboard Features

The React-based dashboard provides:

* **Multi-word Comparison**: Plot VAD trajectories for multiple words simultaneously
* **Forecasting Visualization**: Display historical data with predicted future trajectories
* **Multi-dimensional Views**: 
  - 2D plots (V/A/D over time)
  - 3D VAD space visualization
  - 4D plots with sense proportion coloring
* **Interactive Controls**: Word selection, forecast target years, sense filtering
* **Real-time API Integration**: Live predictions through backend API

### Dashboard Screenshots

**![2D VAD Trajectory View](data/imgs/2d_vad_trajectory.png)**

*Figure 3: 2D temporal visualization showing VAD values over time for the word "alien" with forecasting capabilities. Solid lines represent historical data, dotted lines show LSTM predictions.*

**![3D VAD Space Visualization](data/imgs/3d_vad_space.png)**

*Figure 4: 3D VAD space visualization displaying emotional trajectory through valence-arousal-dominance dimensions. Dot shape represents temporal progression from historical (rounded) to predicted (squared) periods.*

**![Multi-word Comparison](data/imgs/multi_word_comparison.png)**

*Figure 5: Multi-word VAD trajectory comparison showing emotional evolution patterns across different lexical items with synchronized time axes and forecast extensions.*

---

## Model Performance

### Training Configuration
- **Input Features**: 27 (3 base VAD differences + 24 momentum features)
- **Architecture**: 2-layer LSTM (128 hidden units) + Multi-head Attention (8 heads)
- **Lookback Window**: 15 timesteps (75 years at 5-year intervals)
- **Training Split**: Pre-1980 for training, post-1980 for validation
- **Regularization**: Dropout (0.2), Early Stopping, Gradient Clipping
- **Optimizer**: AdamW with ReduceLROnPlateau scheduling

### Training Results
- **Total Epochs**: 46/100 (Early stopping triggered)
- **Best Validation Loss**: 0.11506 (Epoch 36)
- **Final Training Loss**: 0.04871
- **Training Time**: ~17 minutes on CPU

### Evaluation Metrics (Test Set)
- **Overall Test MAE**: 0.000278 
- **Overall Test RMSE**: 0.001134
- **Per-dimension Performance**:
  - **Valence**: MAE = 0.000355, RMSE = 0.001428
  - **Arousal**: MAE = 0.000247, RMSE = 0.001056  
  - **Dominance**: MAE = 0.000232, RMSE = 0.000840

### Model Accuracy Examples
**Word: "body" - Predicted vs Actual VAD Values**
```
Year 2000: Predicted(0.1636, -0.1769, 0.0758) vs Actual(0.1636, -0.1770, 0.0757)
Year 2005: Predicted(0.1654, -0.1768, 0.0792) vs Actual(0.1655, -0.1770, 0.0792)
Year 2010: Predicted(0.1675, -0.1769, 0.0829) vs Actual(0.1675, -0.1770, 0.0828)
```

**Still to do**

**![Training Performance](data/imgs/training_performance.png)**

*Figure 7: Training and validation loss curves showing model convergence with early stopping at epoch 46, achieving great accuracy with sub-0.001 MAE across all VAD dimensions.*

**![Feature Importance Analysis](data/imgs/feature_importance.png)**

*Figure 8: Analysis of momentum feature contributions showing relative importance of velocity, acceleration, volatility, and trend features for VAD prediction accuracy.*

---

## Innovations

1. **Advanced Momentum Tracking**: 8 sophisticated temporal features capture complex emotional dynamics beyond simple differences
2. **Attention-Enhanced LSTM**: Multi-head attention mechanisms identify important temporal relationships
3. **Production-Ready Architecture**: Complete model persistence, scaling, and API integration
4. **Exceptional Accuracy**: Sub-millisecond MAE/RMSE on temporal VAD prediction tasks
5. **Multi-dimensional Analysis**: Support for sense-aware emotional trajectory modeling
6. **3D Trajectory Visualization**: A novel way to visualize and understand diachronic emotional evolution

---

## Research Applications

EmoTracker enables research in:

* **Computational Linguistics**: Diachronic semantic change detection
* **Digital Humanities**: Historical emotion analysis in literary corpora  
* **Social Science**: Tracking societal attitude shifts through language
* **NLP**: Temporal emotion-aware language models
* **Lexicography**: Dynamic emotion lexicon construction

---

## References

* Hu et al. (2019). *Diachronic Sense Modeling with Deep Contextualized Word Embeddings*. ACL 2019.
* Mohammad (2018). *Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words*. LREC 2018.

... still to finish :P

---
