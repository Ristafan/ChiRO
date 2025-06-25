import itertools
import json
import random
import torch
import TrainingParams as tp
from TrainA_SectionDynamic import main as train_alpha_section_dynamic
from src.Training.Train_A import main

if __name__ == "__main__":
    print("Starting hyperparameter tuning...", flush=True)
    torch.multiprocessing.set_start_method('spawn', force=True)
    tp.DEVICE = "cuda"

    dropout_rate = [0.0]
    batch_norm = [True]

    learning_rate = [0.001, 0.0001]
    optimizer = ["Adam"]

    early_stopping = [True]
    patience = [2]
    batch_size = [6]
    epochs = [10]
    global_pooling = ["avg", "max"]

    dataset_seed = [42]
    training_architectures = ["AlphaResNet50"]

    # Conditional
    window_size_overlap_size = [[0.23, 0.12], [0.27, 0.13], [0.2, 0.1], [1.0, 0.3]]
    loss_filter_threshold = [0.75]

    num_heads = [1, 2]

    # Create and shuffle all combinations
    hyperparameter_combinations = list(itertools.product(
        optimizer,
        batch_size,
        epochs,
        learning_rate,
        dropout_rate,
        loss_filter_threshold,
        window_size_overlap_size,
        num_heads,
        batch_norm,
        early_stopping,
        patience,
        global_pooling
    ))
    random.shuffle(hyperparameter_combinations)

    for architecture in training_architectures:
        training_params = tp.TrainingParams()

        # Set fixed parameters
        training_params.model = "AlphaSectionDynamic"
        training_params.dataset_name = "BatCalls-Environment"

        training_params.model_architecture = architecture
        training_params.model_summary = str(architecture)

        # Loop through each configuration
        for idx, (op, bs, ep, lr, dr, lft, ws_os, heads, bn, es, pt, gp) in enumerate(hyperparameter_combinations):
            print(f"Running configuration {idx + 1}/{len(hyperparameter_combinations)}: "
                  f"Batch Size={bs}, Epochs={ep}, Learning Rate={lr}, "
                  f"Dropout Rate={dr}, Loss Filter Threshold={lft}, "
                  f"Window Size={ws_os[0]}, Overlap Size={ws_os[1]}, "
                  f"Num Heads={heads}, Batch Norm={bn}, Early Stopping={es}, Patience={pt}, Optimizer={op}, global_pooling={gp}")

            # Update training parameters
            training_params.batch_size = bs
            training_params.num_epochs = ep
            training_params.learning_rate = lr
            training_params.dropout_rate = dr
            training_params.loss_filter_threshold_percentage = lft
            training_params.window_size = ws_os[0]
            training_params.overlap_size = ws_os[1]
            training_params.batch_norm = bn
            training_params.early_stopping = es
            training_params.patience = pt
            training_params.optimizer = op
            training_params.attention_heads = heads
            training_params.global_pooling = gp
            training_params.model_name = f"{training_params.model}_{training_params.model_architecture}_{op}_{bs}_{ep}_{lr}_{dr}_{lft}_{ws_os[0]}_{ws_os[1]}_{heads}_{bn}_{es}_{pt}".replace(".", "_")

            training_params.splits_already_computed = True
            training_params.spectrograms_already_computed = True

            # Create files in first run
            if idx == 0:
                training_params.splits_already_computed = True
                training_params.spectrograms_already_computed = True

            try:
                train_alpha_section_dynamic(training_params)
            except Exception as e:
                print(f"Training failed on config {idx+1}: {e}")


# Syncing with WandB // run from wandb directory:
# Get-ChildItem -Directory -Filter "offline-run-*" | ForEach-Object { wandb sync $_.Name }
# Get-ChildItem -Directory -Filter "run-*" | ForEach-Object { wandb sync $_.Name }
