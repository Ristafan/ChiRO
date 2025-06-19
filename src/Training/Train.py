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
    print("Starting hyperparameter tuning...", flush=True)
    torch.multiprocessing.set_start_method('spawn', force=True)
    tp.DEVICE = "cuda"

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

    training_params = tp.TrainingParams()

    # Loop through each configuration
    for idx, (bs, ep, lr, dr, lft, (ws, os)) in enumerate(hyperparameter_combinations):
        print(f"\n=== Running config {idx+1}/{len(hyperparameter_combinations)} ===")
        print(f"BATCH_SIZE={bs}, NUM_EPOCHS={ep}, LEARNING_RATE={lr}, DROPOUT_RATE={dr},")
        print(f"LOSS_FILTER_THRESHOLD_PERCENTAGE={lft}, WINDOW_SIZE={ws}, OVERLAP_SIZE={os}")

        # Update training parameters
        training_params.batch_size = bs
        training_params.num_epochs = ep
        training_params.learning_rate = lr
        training_params.dropout_rate = dr
        training_params.loss_filter_threshold_percentage = lft
        training_params.window_size = ws
        training_params.overlap_size = os
        training_params.splits_already_computed = True
        training_params.spectrograms_already_computed = True

        # Create files in first run
        if idx == 0:
            training_params.splits_already_computed = False
            training_params.spectrograms_already_computed = False

        try:
            train_alpha_mil_main(training_params)
        except Exception as e:
            print(f"Training failed on config {idx+1}: {e}")
