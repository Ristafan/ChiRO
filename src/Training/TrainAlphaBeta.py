import math

from torch import nn
from tqdm import tqdm
import os
import wandb
from torch.utils.data import DataLoader
import torch
from torch.nn import functional as F
from datetime import datetime
from src.Preprocessing.AudioLoader import AudioLoader
from src.Preprocessing.LabelsLoader import LabelsLoader
from src.Preprocessing.SpectrogramProcessor import SpectrogramProcessor
from src.Architectures.AlphaBetaV1 import AlphaBetaV1
from src.Preprocessing.BatCallDataset import BatCallDataset
from src.utils import load_config

# Set memory allocation configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def train_model(model, train_loader, num_epochs=10, learning_rate=0.001,
                noise_label=7, generic_bat_label=6):
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

    # Enable mixed precision training
    scaler = torch.cuda.amp.GradScaler()

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

        # Clear cache before each epoch
        torch.cuda.empty_cache()

        # Add progress bar for each epoch
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for batch_idx, (spectrograms, labels) in enumerate(train_pbar):
            spectrograms = spectrograms.to(device)
            labels_alpha = labels[:, 0].to(device)  # Bat call/noise labels
            labels_beta = labels[:, 1].to(device)  # Genus labels
            # Convert to half precision to save memory
            spectrograms = spectrograms.half()

            optimizer.zero_grad()

            # Use automatic mixed precision
            with torch.cuda.amp.autocast():
                outputs_alpha, outputs_beta = model(spectrograms)
                loss = criterion(outputs_alpha, labels_alpha, outputs_beta, labels_beta)

            # Scale gradients and perform backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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

            # Clear memory every few batches if needed
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

        train_loss = running_loss / len(train_loader)
        train_acc_alpha = 100 * correct_alpha / total_alpha
        train_acc_beta = 100 * correct_beta / total_beta if total_beta > 0 else 0
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, "
              f"Accuracy Alpha: {train_acc_alpha:.2f}%, Accuracy Beta: {train_acc_beta:.2f}%")

        # Log training metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc_alpha": train_acc_alpha,
            "train_acc_beta": train_acc_beta,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

    print("Training complete!")
    return model


def combined_loss(output_alpha, target_alpha, output_beta, target_beta, noise_label):
    """
    Calculates the combined loss for the AlphaBeta model, dynamically
    adjusting based on whether the input is a bat call or noise, and
    incorporating beta prediction probability into alpha loss for bat calls.

    Args:
        output_alpha (torch.Tensor): Output from head_alpha (shape: [batch_size, 2]).
        target_alpha (torch.Tensor): Ground truth labels for bat call/noise (shape: [batch_size]).
        output_beta (torch.Tensor): Output from head_beta (shape: [batch_size, num_genera]).
        target_beta (torch.Tensor): Ground truth labels for genus (shape: [batch_size]).
        noise_label (int): The label ID for noise/environment sounds.

    Returns:
        torch.Tensor: The combined loss (scalar).
    """
    # Binary cross-entropy loss for head alpha
    loss_alpha = F.cross_entropy(output_alpha, target_alpha, reduction='none')  # Get per-sample losses

    # Cross-entropy loss for head beta, applied only when target_alpha is not the noise_label
    bat_mask = (target_alpha != noise_label)
    if bat_mask.any():
        loss_beta = F.cross_entropy(output_beta[bat_mask], target_beta[bat_mask])

        # Get probabilities of beta predictions for bat calls
        beta_probs = F.softmax(output_beta[bat_mask], dim=1)  # shape: [num_bat_calls, num_genera]
        # Get the probabilities of the correct genus
        correct_beta_probs = beta_probs.gather(dim=1, index=target_beta[bat_mask].unsqueeze(1)).squeeze(1)

        # Incorporate beta probabilities into alpha loss for bat calls
        modified_alpha_loss = loss_alpha.clone()  # avoid modifying original
        modified_alpha_loss[bat_mask] = - target_alpha[bat_mask].float() * torch.log(correct_beta_probs) # changed the formula

        combined_loss = modified_alpha_loss.mean() + loss_beta

    else:
        combined_loss = loss_alpha.mean()  # Only include alpha loss for noise

    return combined_loss


"""
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
"""

def collate_fn(batch):
    import torch.nn.functional as F

    spectrograms = [item[0] for item in batch]
    labels_alpha = []
    labels_beta = []
    for item in batch:
        # item[1] is now a scalar tensor, so we extract the value directly
        label_value = item[1].item()
        # Derive alpha label: 1 for bat call, 0 for noise
        if label_value == 7:  # Use the noise_label
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


if __name__ == '__main__':
    # Load configuration paths
    config = load_config()

    # Set up paths
    spectrogram_already_computed = False
    example_data = False

    # Paths for example data
    files_path = config['example_data']['train_files_labels']
    spectrograms_path = config['spectrogram']['spectrograms_dir']
    model_path = config['model']['alpha_beta']

    if not example_data:
        files_path = config['dataset']['train_files_labels']
        spectrograms_path = config['spectrogram']['spectrograms_dir']

    wandb.login(key="32b08e4c860b935b2cd9c30774889b952ffefe0d")

    run = wandb.init(
        project="ChiRO",
        entity="martin-faehnrich-university-of-z-rich",
        job_type="training",
        config={
            "notes": "Training AlphaBeta model with dynamic labels",
            "learning_rate": 0.001,
            "dataset": "Example-BatCalls-Environment",
            "num_epochs": 4,
            "batch_size": 8,
            "model": "AlphaBetaV1",
            "model_name": f"alphaBeta_{datetime.now().strftime('%H-%M-%S')}.pth",
            "noise_label": 7,  # Add noise_label to config
            "generic_bat_label": 6, # Add generic bat label to config
        },
    )

    wb_config = wandb.config
    noise_label = wb_config.noise_label # Get the noise label from config
    generic_bat_label = wb_config.generic_bat_label # Get the generic bat label.

    if not spectrogram_already_computed:
        # Load Audio Files and create spectrograms
        audio_loader = AudioLoader()
        audio_loader.load_audio_from_exel(files_path)
        waveforms = audio_loader.get_data()
        names = audio_loader.get_file_names_from_excel(files_path)

        # Create Spectrograms
        for i in tqdm(range(len(waveforms)), desc="Creating Spectrograms"):
            sp = SpectrogramProcessor(waveforms[i])
            sp.apply_highpass_filter()
            sp.compute_spectrogram()
            sp.denoise_spectrogram()
            sp.save_spectrogram(f'{names[i]}', spectrograms_path + '/')

    # Load training labels from Excel
    labels_loader = LabelsLoader(files_path, filename_column="Filename", text_column="label")
    labels_loader.load_labels_excel()

    # Create training Dataset & DataLoader
    train_dataset = BatCallDataset(spectrograms_path, labels_loader)
    train_loader = DataLoader(train_dataset, batch_size=wb_config.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=2)

    # Initialize Model
    model = AlphaBetaV1(num_genera=8)

    # Log model architecture
    wandb.log({"model_summary": str(model)})

    model = train_model(model, train_loader, num_epochs=wb_config.num_epochs,
                        learning_rate=wb_config.learning_rate,
                        noise_label=noise_label, generic_bat_label=generic_bat_label) # Pass noise label

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
