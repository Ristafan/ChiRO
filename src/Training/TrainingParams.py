from datetime import datetime
import os

from src.Architectures.AlphaV2 import AlphaV2
from src.DataSetSplit.TrainingClasses import bat_species, bat_species_fixed


# Device Parameters
DEVICE = "cpu"     # Options: 'cuda' for GPU, 'cpu' for CPU

# Preprocessing Progress
SPLITS_ALREADY_COMPUTED = True
SPECTROGRAMS_ALREADY_COMPUTED = True

# Data split Parameters
DATASET_NAME = "BatCalls-Environment"
USE_MIN_FILES_PER_CLASS = True
TOTAL_FILES_PER_CLASS = 100
IGNORED_LABELS = None                       # e.g., ['Chiroptera generally', 'Noise']
MERGE_LABELS = [bat_species_fixed]                # e.g., ['Myotis', 'Plecotus']
SPLIT_METHOD = True                     # True for splitting by location, False for random split
SPLIT_RATIOS = (0.8, 0.1, 0.10)       # Train, Validation, Test split ratios
SEED = 42

# Preprocessing Parameters
SAMPLE_RATE = 192000
HIGHPASS_CUTOFF_FREQ = 16000
N_FFT = 4096
WIN_LENGTH = 2048
HOP_LENGTH = WIN_LENGTH // 2
DENOISE_OPTION = "mean_subtraction"           # Options: 'mean_subtraction', 'median_filter'

# Training Model Parameters
MODEL = "AlphaMIL"
MODEL_NAME = f"{MODEL}_{datetime.now().strftime('%H-%M-%S')}.pth"
MODEL_ARCHITECTURE = AlphaV2
OPTIMIZER = "Adam"  # Options: 'Adam', 'SGD'
BATCH_SIZE = 2
NUM_EPOCHS = 4
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.3
BATCH_NORM = True                          # Set to True if you want to use Batch Normalization in the model
EARLY_STOPPING = True
PATIENCE = 3

# Windows Parameters
WINDOW_SIZE = 0.2
OVERLAP_SIZE = 0.1
LOSS_FILTER_THRESHOLD_PERCENTAGE = 0.9  # Percentage of the maximum loss to filter outliers

# WandB Parameters
os.environ["WANDB_MODE"] = "offline"  # Set to "online" for live logging, "offline" for local logging
USE_WANDB = True
WANDB_PROJECT = "ChiRO"
WANDB_ENTITY = "martin-faehnrich-university-of-z-rich"
WANDB_JOB_TYPE = "training"
WANDB_API_KEY = "32b08e4c860b935b2cd9c30774889b952ffefe0d"


class TrainingParams:
    def __init__(self):
        self.device = DEVICE

        self.splits_already_computed = SPLITS_ALREADY_COMPUTED
        self.spectrograms_already_computed = SPECTROGRAMS_ALREADY_COMPUTED

        self.dataset_name = DATASET_NAME
        self.use_min_files_per_class = USE_MIN_FILES_PER_CLASS
        self.total_files_per_class = TOTAL_FILES_PER_CLASS
        self.ignored_labels = IGNORED_LABELS
        self.merge_labels = MERGE_LABELS
        self.split_method = SPLIT_METHOD
        self.split_ratios = SPLIT_RATIOS
        self.seed = SEED

        self.sample_rate = SAMPLE_RATE
        self.highpass_cutoff_freq = HIGHPASS_CUTOFF_FREQ
        self.n_fft = N_FFT
        self.win_length = WIN_LENGTH
        self.hop_length = HOP_LENGTH
        self.denoise_option = DENOISE_OPTION

        self.model = MODEL
        self.model_name = MODEL_NAME
        self.model_architecture = MODEL_ARCHITECTURE
        self.optimizer = OPTIMIZER
        self.batch_size = BATCH_SIZE
        self.num_epochs = NUM_EPOCHS
        self.learning_rate = LEARNING_RATE
        self.dropout_rate = DROPOUT_RATE
        self.batch_norm = BATCH_NORM
        self.early_stopping = EARLY_STOPPING
        self.patience = PATIENCE

        self.window_size = WINDOW_SIZE
        self.overlap_size = OVERLAP_SIZE
        self.loss_filter_threshold_percentage = LOSS_FILTER_THRESHOLD_PERCENTAGE

        # WandB parameters
        self.use_wandb = USE_WANDB
        self.wandb_project = WANDB_PROJECT
        self.wandb_entity = WANDB_ENTITY
        self.wandb_job_type = WANDB_JOB_TYPE
