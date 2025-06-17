import itertools
import random
import sys

import torch
import os
import TrainingParams as tp
from TrainAlpha import main as train_alpha_main
from TrainAlphaAttention import main as train_alpha_attention_main
from TrainBeta import main as train_beta_main
from TrainAlphaBeta import main as train_alpha_beta_main
from TrainAlphaMIL import main as train_alpha_mil_main


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    tp.DEVICE = "cuda"

    # Setup logging
    log_file = "log.out"
    os.makedirs(os.path.dirname(log_file), exist_ok=True) if os.path.dirname(log_file) else None
    sys.stdout = open(log_file, "w", buffering=1)  # line-buffered
    sys.stderr = sys.stdout  # Redirect errors to same file

    # Define hyperparameters
    batch_sizes = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    num_epochs = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    learning_rates = [0.001, 0.0001, 0.00001, 0.000001]
    dropout_rates = [0.1, 0.2, 0.3, 0.4]
    loss_filter_threshold_percentage = [0.75, 0.8, 0.85, 0.9, 0.95]

    # Paired parameters
    window_size = [1.0, 1.0, 0.23, 0.1, 0.2, 0.27]
    overlap_size = [0.2, 0.3, 0.12, 0.05, 0.1, 0.13]

    paired_window_overlap = list(zip(window_size, overlap_size))

    # Create and shuffle all combinations
    hyperparameter_combinations = list(itertools.product(
        batch_sizes,
        num_epochs,
        learning_rates,
        dropout_rates,
        loss_filter_threshold_percentage,
        paired_window_overlap
    ))
    random.shuffle(hyperparameter_combinations)

    # Loop through each configuration
    for idx, (bs, ep, lr, dr, lft, (ws, os)) in enumerate(hyperparameter_combinations):
        print(f"\n=== Running config {idx+1}/{len(hyperparameter_combinations)} ===")
        print(f"BATCH_SIZE={bs}, NUM_EPOCHS={ep}, LEARNING_RATE={lr}, DROPOUT_RATE={dr},")
        print(f"LOSS_FILTER_THRESHOLD_PERCENTAGE={lft}, WINDOW_SIZE={ws}, OVERLAP_SIZE={os}")

        # Update training parameters
        tp.BATCH_SIZE = bs
        tp.NUM_EPOCHS = ep
        tp.LEARNING_RATE = lr
        tp.DROPOUT_RATE = dr
        tp.LOSS_FILTER_THRESHOLD_PERCENTAGE = lft
        tp.WINDOW_SIZE = ws
        tp.OVERLAP_SIZE = os

        try:
            train_alpha_attention_main()
        except Exception as e:
            print(f"Training failed on config {idx+1}: {e}")
