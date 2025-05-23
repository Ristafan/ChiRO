from torch import nn
from tqdm import tqdm
import os
import wandb
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from datetime import datetime

from src.DataSetSplit.TrainingClasses import bat_species
from src.Preprocessing.Preprocessor import Preprocessor
from src.Architectures.AlphaV2 import AlphaV2
from src.utils import load_config

# Set memory allocation configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def train_model(model, train_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    wandb.watch(model, criterion, log="all", log_freq=10)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Add progress bar for each epoch
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for batch_idx, (spectrograms, labels) in enumerate(train_pbar):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            # Convert to half precision to save memory
            spectrograms = spectrograms.half()

            optimizer.zero_grad()

            # Use automatic mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(spectrograms)  # Process whole batch at once
                loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            train_pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})

            # Clear memory every few batches if needed
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                # Debug memory
                # print(f"Memory after batch {batch_idx}: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

        # Log training metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

    print("Training complete!")

    return model


def collate_fn(batch):
    spectrograms = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Find max time length
    max_len = max(spec.shape[-1] for spec in spectrograms)

    # Pad time dimension (last dim) to max_len
    padded_specs = [F.pad(spec, (0, max_len - spec.shape[-1])) for spec in spectrograms]

    # Now they all should have the same shape: [1, freq_bins, max_len]
    spectrograms = torch.stack(padded_specs)

    labels = torch.stack(labels, dim=0)
    return spectrograms, labels


if __name__ == '__main__':
    # Define whether spectrograms are already computed
    splits_aleady_computed = False
    spectrogram_already_computed = False

    # Load configuration paths
    config = load_config()
    train_files_and_labels_path = config['dataset']['train_files_and_labels_path_alpha']
    original_files_and_labels_path = config['dataset']['original_files_and_labels_path']
    root_files_path = config['dataset']['files_path_root']
    spectrograms_path = config['spectrogram']['spectrograms_dir']
    model_path = config['model']['alpha']

    wandb.login(key="32b08e4c860b935b2cd9c30774889b952ffefe0d")

    run = wandb.init(
        project="ChiRO",
        entity="martin-faehnrich-university-of-z-rich",
        job_type="training",
        config={
            "notes": "",
            "learning_rate": 0.001,
            "dataset": "BatCalls-Environment",
            "num_epochs": 2,
            "batch_size": 2,
            "model": "AlphaV2",
            "model_name": f"alphaV2_{datetime.now().strftime('%H-%M-%S')}.pth",
        },
    )

    wb_config = wandb.config

    # Load Audio Files, Labels and create spectrograms
    preprocessor = Preprocessor(train_files_and_labels_path, spectrograms_path, root_files_path)

    if not splits_aleady_computed:
        _ = preprocessor.create_data_splits(original_files_and_labels_path,
                                                  use_min_files_per_class=True,
                                                  total_files_per_class=100,
                                                  ignored_labels=None,
                                                  merge_labels=[bat_species],
                                                  split_method="balanced",
                                                  train_ratio=0.7,
                                                  test_ratio=0.2,
                                                  seed=42)

    if not spectrogram_already_computed:
        preprocessor.create_spectrograms()

    train_dataset = preprocessor.create_bat_file_dataset()

    train_loader = DataLoader(train_dataset, batch_size=wb_config.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=2)

    # Initialize Model
    model = AlphaV2(batch_norm=False)

    # Log model architecture
    wandb.log({"model_summary": str(model)})

    model = train_model(model, train_loader, num_epochs=wb_config.num_epochs, learning_rate=wb_config.learning_rate)

    # Ensure the directory exists
    os.makedirs(model_path, exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), os.path.join(model_path, wb_config.model_name))

    # Also save a checkpoint with more information
    checkpoint = {
        'epoch': wb_config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None,  # You'd capture this from train_model if needed
        'loss': None,  # You'd capture this from train_model if needed
        'config': {k: v for k, v in wb_config.items()}
    }
    torch.save(checkpoint, os.path.join(model_path, 'checkpoint_' + wb_config.model_name))

    # Number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run.config.update({"num_params": f'The number of params is {num_params}'})

    # Finish the run
    run.finish()
