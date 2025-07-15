import os
import logging
from trainer import train_and_evaluate_pytorch_lstm
from config import (
    ML_READY_OUTPUT_FILENAME_FOR_GENERATION, MODEL_ASSETS_DIR,
    LOOKBACK_PARAM, FORECAST_HORIZON_PARAM, TRAIN_UNTIL_YEAR_PARAM,
    NUM_EPOCHS, NUM_EXAMPLES_TO_SHOW, HIDDEN_SIZE_PARAM,
    NUM_LSTM_LAYERS_PARAM, DROPOUT_PARAM, LEARNING_RATE_PARAM,
    BATCH_SIZE
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def setup_directories_and_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(MODEL_ASSETS_DIR, exist_ok=True)
    data_gen_output_dir = os.path.normpath(os.path.join(script_dir, '..', '..', 'data', 'Generated_VAD_Dataset'))
    ml_ready_data_file = os.path.join(data_gen_output_dir, ML_READY_OUTPUT_FILENAME_FOR_GENERATION)
    return ml_ready_data_file


def main():
    ml_ready_data_file = setup_directories_and_paths()

    if not os.path.exists(ml_ready_data_file):
        logging.error(f"Data file not found: {ml_ready_data_file}")
        logging.error("Please run the data generation script "
            "(the one that produced emotracker_nrc.json) first.")
        return

    trained_model, _, _ = train_and_evaluate_pytorch_lstm(
        ml_ready_data_path=ml_ready_data_file,
        lookback_param=LOOKBACK_PARAM,
        forecast_horizon_param=FORECAST_HORIZON_PARAM,
        train_until_year_param=TRAIN_UNTIL_YEAR_PARAM,
        hidden_size=HIDDEN_SIZE_PARAM,
        num_lstm_layers=NUM_LSTM_LAYERS_PARAM,
        dropout=DROPOUT_PARAM,
        learning_rate=LEARNING_RATE_PARAM,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        num_examples_to_show=NUM_EXAMPLES_TO_SHOW
    )

    if trained_model:
        print("LSTM Training and asset saving complete.")
    else:
        print("LSTM Training failed.")


if __name__ == "__main__":
    main()
