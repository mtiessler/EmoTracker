import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path("..") / ".env")


class Config:
    # File paths
    FILE_PATH = os.path.join('..', 'data', 'vad_lexicon.csv')
    TOKENIZER_NAME = os.getenv('TOKENIZER_NAME', 'roberta-base')
    REPO_NAME = os.getenv('REPO_NAME')

    # Hyperparameters for exploration
    MAX_LENS = [64, 128, 256]
    BATCH_SIZES = [16, 32, 64]
    EPOCHS_LIST = [3, 5, 7]
    LEARNING_RATES = [2e-5, 1e-5, 5e-6]
    DROPOUT_RATES = [0.1, 0.2, 0.3]
    WARMUP_STEPS_LIST = [0, 100, 500]

    # Default hyperparameters (will be updated after exploration)
    MAX_LEN = 128
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    DROPOUT_RATE = 0.1
    WARMUP_STEPS = 100