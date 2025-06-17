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
from src.Architectures.AlphaV1_Attention import AlphaV1_Attention
from src.Training.TrainingParams import SPLITS_ALREADY_COMPUTED, SPECTROGRAMS_ALREADY_COMPUTED, USE_MIN_FILES_PER_CLASS, \
    TOTAL_FILES_PER_CLASS, IGNORED_LABELS, MERGE_LABELS, SPLIT_METHOD, SEED, LEARNING_RATE, \
    DATASET_NAME, NUM_EPOCHS, BATCH_SIZE, MODEL, MODEL_NAME, WANDB_API_KEY
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


def main():
    # Load configuration paths
    config = load_config()
    train_files_and_labels_path = config['dataset']['train_files_and_labels_path_alpha']
    original_files_and_labels_path = config['dataset']['original_files_and_labels_path']
    root_files_path = config['dataset']['files_path_root']
    spectrograms_path = config['spectrogram']['spectrograms_dir']
    model_path = config['model']['alpha']

    wandb.login(key=WANDB_API_KEY)

    run = wandb.init(
        project="ChiRO",
        entity="martin-faehnrich-university-of-z-rich",
        job_type="training",
        config={
            "notes": "",
            "learning_rate": LEARNING_RATE,
            "dataset": DATASET_NAME,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "model": MODEL,
            "model_name": MODEL_NAME,
        },
    )

    wb_config = wandb.config

    # Load Audio Files, Labels and create spectrograms
    preprocessor = Preprocessor(train_files_and_labels_path, spectrograms_path, root_files_path)

    if not SPLITS_ALREADY_COMPUTED:
        _ = preprocessor.create_data_splits(original_files_and_labels_path,
                                                  use_min_files_per_class=USE_MIN_FILES_PER_CLASS,
                                                  total_files_per_class=TOTAL_FILES_PER_CLASS,
                                                  ignored_labels=IGNORED_LABELS,
                                                  merge_labels=MERGE_LABELS,
                                                  split_method=SPLIT_METHOD,
                                                  train_ratio=TRAIN_RATIO,
                                                  test_ratio=TEST_RATIO,
                                                  seed=SEED)

    if not SPECTROGRAMS_ALREADY_COMPUTED:
        preprocessor.create_spectrograms_stft()

    train_dataset = preprocessor.create_bat_file_dataset()

    train_loader = DataLoader(train_dataset, batch_size=wb_config.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=2)

    # Initialize Model
    model = AlphaV1_Attention(batch_norm=False)

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
