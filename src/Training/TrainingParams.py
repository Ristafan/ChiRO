from datetime import datetime
from src.DataSetSplit.TrainingClasses import bat_species


# Preprocessing Progress
SPLITS_ALREADY_COMPUTED = False
SPECTROGRAMS_ALREADY_COMPUTED = False

# Data split Parameters
DATASET_NAME = "BatCalls-Environment"
USE_MIN_FILES_PER_CLASS = True
TOTAL_FILES_PER_CLASS = 100
IGNORED_LABELS = None                       # e.g., ['Chiroptera generally', 'Noise']
MERGE_LABELS = [bat_species]                # e.g., ['Myotis', 'Plecotus']
SPLIT_METHOD = "balanced"                   # Options: 'balanced', 'random'
TRAIN_RATIO = 0.7
TEST_RATIO = 0.2                            # Validation ratio will be automatically calculated as 1 - train_ratio - test_ratio
SEED = 42

# Preprocessing Parameters
HIGHPASS_CUTOFF_FREQ = 16000
N_FFT = 4096
HOP_LENGTH = None                            # Default will be set to n_fft // 4
WIN_LENGTH = 2048
DENOISE_OPTION = "mean_subtraction"           # Options: 'mean_subtraction', 'median_filter'

# Training Model Parameters
MODEL = "AlphaV2"
MODEL_NAME = f"alphaV2_{datetime.now().strftime('%H-%M-%S')}.pth"
BATCH_SIZE = 2
NUM_EPOCHS = 2
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
BATCH_NORM = False                          # Set to True if you want to use Batch Normalization in the model

# WandB Parameters
USE_WANDB = True
WANDB_PROJECT = "ChiRO"
WANDB_ENTITY = "martin-faehnrich-university-of-z-rich"
WANDB_JOB_TYPE = "training"
WANDB_API_KEY = "32b08e4c860b935b2cd9c30774889b952ffefe0d"

