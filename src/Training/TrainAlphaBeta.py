import math

from torch import nn
from tqdm import tqdm
import os
import wandb
from torch.utils.data import DataLoader
import torch
from torch.nn import functional as F
from datetime import datetime

from src.DataSetSplit.TrainingClasses import eptesicus_species, myotis_species, nyctalus_species, pipistrellus_species, \
    Chiroptera_generally
from src.Preprocessing.Preprocessor import Preprocessor
from src.Architectures.AlphaBetaV1 import AlphaBetaV1
from src.utils import load_config

# Set memory allocation configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def train_model(model, train_loader, num_epochs=10, learning_rate=0.001,
                noise_label=7):
    """
    Trains the AlphaBeta model.

    Args:
        model (nn.Module): The AlphaBeta model.
        train_loader (DataLoader): DataLoader for the training data.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        noise_label (int): The label ID for noise/environment sounds.
        generic_bat_label (int): The label ID for generic bat calls.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize lists to store statistics for the entire dataset
    all_means = []
    all_stds = []

    # Use the combined loss function
    def criterion(output_alpha, target_alpha, output_beta, target_beta):
        """Wrapper for the combined_loss function"""
        return combined_loss(output_alpha, target_alpha, output_beta, target_beta,
                             noise_label=noise_label)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    wandb.watch(model, criterion, log="all", log_freq=10)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_alpha = 0
        total_alpha = 0
        correct_beta = 0
        total_beta = 0

        # Add progress bar for each epoch
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")

        for batch_idx, (spectrograms, labels) in enumerate(train_pbar):
            spectrograms = spectrograms.to(device)
            labels_alpha = labels[:, 0].to(device)  # Bat call/noise labels
            labels_beta = labels[:, 1].to(device)  # Genus labels

            # Calculate and print batch statistics
            batch_mean = spectrograms.mean()
            batch_std = spectrograms.std()

            # Store batch statistics for the entire dataset
            all_means.append(batch_mean.item())
            all_stds.append(batch_std.item())

            optimizer.zero_grad()

            # Use automatic mixed precision
            outputs_alpha, outputs_beta = model(spectrograms)

            loss = criterion(outputs_alpha, labels_alpha, outputs_beta, labels_beta)

            # Perform backward pass
            loss.backward()

            running_loss += loss.item()

            # Calculate accuracy for alpha head (bat call/noise)
            _, predicted_alpha = torch.max(outputs_alpha.data, 1)
            total_alpha += labels_alpha.size(0)
            correct_alpha += (predicted_alpha == labels_alpha).sum().item()

            # Calculate accuracy for beta head (genus), only on bat calls
            bat_mask = (labels_alpha != noise_label)  # Use the noise_label
            if bat_mask.any():
                _, predicted_beta = torch.max(outputs_beta[bat_mask].data, 1)
                total_beta += labels_beta[bat_mask].size(0)
                correct_beta += (predicted_beta == labels_beta[bat_mask]).sum().item()

            # Update progress bar
            train_pbar.set_postfix({
                'loss': loss.item(),
                'acc_alpha': 100 * correct_alpha / total_alpha,
                'acc_beta': 100 * correct_beta / total_beta if total_beta > 0 else 0,
            })

        train_loss = running_loss / len(train_loader)
        train_acc_alpha = 100 * correct_alpha / total_alpha
        train_acc_beta = 100 * correct_beta / total_beta if total_beta > 0 else 0
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, "
              f"Accuracy Alpha: {train_acc_alpha:.2f}%, Accuracy Beta: {train_acc_beta:.2f}%")

        # Log training metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc_alpha": train_acc_alpha,
            "train_acc_beta": train_acc_beta,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

    # Calculate and print statistics for the entire dataset
    dataset_mean = sum(all_means) / len(all_means)
    dataset_std = (sum((x - dataset_mean) ** 2 for x in all_stds) / (len(all_stds) - 1)) ** 0.5 if len(
        all_stds) > 1 else 0
    print(f"Dataset: Mean = {dataset_mean:.4f}, Std = {dataset_std:.4f}")

    print("Training complete!")
    return model


def combined_loss(output_alpha, target_alpha, output_beta, target_beta, noise_label):
    # Binary cross-entropy loss for head alpha
    loss_alpha = F.cross_entropy(output_alpha, target_alpha)

    # Cross-entropy loss for head beta, applied only when target_alpha is not the noise_label
    bat_mask = (target_alpha != noise_label)
    if bat_mask.any():
        loss_beta = F.cross_entropy(output_beta[bat_mask], target_beta[bat_mask])
    else:
        loss_beta = torch.tensor(0.0, device=output_alpha.device)

    # Combine the losses
    # combined_loss = loss_alpha + loss_beta
    combined_loss = loss_alpha + loss_beta * math.log(2)  # Adjust the weight of loss_beta

    return combined_loss


def collate_fn(batch, noise_label=7):
    spectrograms = [item[0] for item in batch]
    labels_alpha = []
    labels_beta = []
    for item in batch:
        # item[1] is now a scalar tensor, so we extract the value directly
        label_value = item[1].item()
        # Derive alpha label: 1 for bat call, 0 for noise
        if label_value == noise_label:  # Use the noise_label
            alpha_label = 0
        else:
            alpha_label = 1
        labels_alpha.append(alpha_label)
        labels_beta.append(label_value)

    # Find max time length
    max_len = max(spec.shape[-1] for spec in spectrograms)

    # Pad time dimension (last dim) to max_len
    padded_specs = [F.pad(spec, (0, max_len - spec.shape[-1])) for spec in spectrograms]

    # Now they all should have the same shape: [1, freq_bins, max_len]
    spectrograms = torch.stack(padded_specs)
    labels_alpha = torch.tensor(labels_alpha)
    labels_beta = torch.tensor(labels_beta)
    labels = torch.stack((labels_alpha, labels_beta), dim=1)
    return spectrograms, labels


def main():
    # Define whether spectrograms are already computed
    splits_already_computed = True
    spectrogram_already_computed = True

    # Load configuration paths
    config = load_config()
    train_files_and_labels_path = config['dataset']['train_files_and_labels_path_alpha_beta']
    original_files_and_labels_path = config['dataset']['original_files_and_labels_path']
    root_files_path = config['dataset']['files_path_root']
    spectrograms_path = config['spectrogram']['spectrograms_dir']
    model_path = config['model']['alpha_beta']

    wandb.login(key="32b08e4c860b935b2cd9c30774889b952ffefe0d")

    run = wandb.init(
        project="ChiRO",
        entity="martin-faehnrich-university-of-z-rich",
        job_type="training",
        config={
            "notes": "Training AlphaBeta model with dynamic labels",
            "learning_rate": 0.001,
            "dataset": "Example-BatCalls-Environment",
            "num_epochs": 2,
            "batch_size": 2,
            "model": "AlphaBetaV1",
            "model_name": f"alphaBeta_{datetime.now().strftime('%H-%M-%S')}.pth",
        },
    )

    wb_config = wandb.config

    # Load Audio Files, Labels and create spectrograms
    preprocessor = Preprocessor(train_files_and_labels_path, spectrograms_path, root_files_path)

    if not splits_already_computed:
        noise_label = preprocessor.create_data_splits(original_files_and_labels_path,
                                                      use_min_files_per_class=True,
                                                      total_files_per_class=100,
                                                      ignored_labels=None,
                                                      merge_labels=[eptesicus_species, myotis_species, nyctalus_species,
                                                                    pipistrellus_species, Chiroptera_generally],
                                                      split_method="balanced",
                                                      train_ratio=0.7,
                                                      test_ratio=0.2,
                                                      seed=42)
    else:
        noise_label = 1

    print(f"Noise label: {noise_label}")

    if not spectrogram_already_computed:
        preprocessor.create_spectrograms_stft()

    train_dataset = preprocessor.create_bat_call_dataset()

    train_loader = DataLoader(train_dataset, batch_size=wb_config.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=2)

    # Initialize Model
    model = AlphaBetaV1(num_genera=8)

    # Log model architecture
    wandb.log({"model_summary": str(model)})

    model = train_model(model, train_loader, num_epochs=wb_config.num_epochs,
                        learning_rate=wb_config.learning_rate,
                        noise_label=noise_label)  # Pass noise label

    # Ensure the directory exists
    os.makedirs(model_path, exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), os.path.join(model_path, wb_config.model_name))

    # Also save a checkpoint with more information
    checkpoint = {
        'epoch': wb_config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None,
        'loss': None,
        'config': {k: v for k, v in wb_config.items()}
    }
    torch.save(checkpoint, os.path.join(model_path, 'checkpoint_' + wb_config.model_name))

    # Number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run.config.update({"num_params": f'The number of params is {num_params}'})

    # Finish the run
    run.finish()
