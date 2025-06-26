import json

from torch import nn
from tqdm import tqdm
import os
import wandb
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from datetime import datetime

from src.Architectures.A_SelfAttention import SelfAttentionNet
from src.Architectures.A_SelfAttentionPositional import AlphaSelfAttentionPositionalNet
from src.Architectures.AlphaResNet50 import AlphaResNet50, Bottleneck
from src.Architectures.AlphaV1_Attention import AlphaV1_Attention
from src.Architectures.AlphaV2_1 import AlphaV2_1
from src.Architectures.AlphaV2_1D import AlphaV2_1D
from src.Architectures.AlphaV2_1D_1 import AlphaV2_1D_1
from src.Architectures.AlphaV3 import AlphaV3
from src.Architectures.AlphaV3_1 import AlphaV3_1
from src.DataSetSplit.TrainingClasses import bat_species
from src.Preprocessing.Preprocessor import Preprocessor
from src.Architectures.AlphaV2 import AlphaV2
from src.Training.TrainingParams import SPLITS_ALREADY_COMPUTED, SPECTROGRAMS_ALREADY_COMPUTED, USE_MIN_FILES_PER_CLASS, \
    TOTAL_FILES_PER_CLASS, IGNORED_LABELS, MERGE_LABELS, SPLIT_METHOD, SEED, LEARNING_RATE, \
    DATASET_NAME, NUM_EPOCHS, BATCH_SIZE, MODEL, MODEL_NAME, WANDB_API_KEY, TrainingParams
from src.utils import load_path_config, update_experiment_configs, create_experiment_dir, log_metrics

# Set memory allocation configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def train_model(model, train_loader, val_loader, config, log_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    best_val_acc = 0
    patience_counter = 0

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")

        for batch_idx, (spectrograms, labels) in enumerate(train_pbar):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            outputs = model(spectrograms)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()

            train_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar
            train_pbar.set_postfix({'loss': loss.item(), 'acc': 100 * train_correct / train_total})

            # Clear memory every few batches if needed
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                # Debug memory
                # print(f"Memory after batch {batch_idx}: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        print(f"Epoch [{epoch+1}/{config.num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        for spectrograms, labels in val_loader:
            inputs, labels = spectrograms.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0

        config_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        }

        log_metrics(log_folder, epoch, config_dict)

        # Early stopping
        if config.early_stopping:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if train_acc < 55.0:
            print(f"Training accuracy {train_acc:.2f}% is below threshold, stopping training.")
            break

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


def main(training_params: TrainingParams = None):
    if training_params is None:
        training_params = TrainingParams()  # Create a default instance if none is provided

    # Load configuration paths
    config = load_path_config()

    train_files_and_labels_path = config['dataset']['train_files_and_labels_path_alpha']
    validation_files_and_labels_path = config['dataset']['validation_files_and_labels_path_alpha']
    original_files_and_labels_path = config['dataset']['original_files_and_labels_path']
    root_files_path = config['dataset']['files_path_root']
    spectrograms_path = config['spectrogram']['spectrograms_dir']
    model_path = config['model']['alpha']
    runs_dir_alpha = config['logs']['runs_dir_alpha']

    # Initialize logging
    log_folder = create_experiment_dir(training_params, runs_dir_alpha)

    # Load Audio Files, Labels and create spectrograms
    preprocessor = Preprocessor(train_files_and_labels_path, spectrograms_path, root_files_path)

    if not training_params.splits_already_computed:
        _ = preprocessor.create_data_splits(original_files_and_labels_path)

    if not training_params.spectrograms_already_computed:
        preprocessor.create_spectrograms_stft(train_files_and_labels_path)
        preprocessor.create_spectrograms_stft(validation_files_and_labels_path)

    train_dataset = preprocessor.create_bat_file_dataset(train_files_and_labels_path)
    train_loader = DataLoader(train_dataset, batch_size=training_params.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=1, pin_memory=True)
    val_dataset = preprocessor.create_bat_file_dataset(validation_files_and_labels_path)
    val_loader = DataLoader(val_dataset, batch_size=training_params.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=1, pin_memory=True)

    if training_params.model_architecture == "AlphaResNet50":
        model = AlphaResNet50(Bottleneck, [3, 4, 6, 3], num_classes=2, dropout_rate=training_params.dropout_rate)
    elif training_params.model_architecture == "AlphaV2":
        model = AlphaV2(training_params.dropout_rate, training_params.batch_norm)
    elif training_params.model_architecture == "AlphaV2_1":
        model = AlphaV2_1(training_params.dropout_rate, training_params.batch_norm)
    elif training_params.model_architecture == "AlphaV3":
        model = AlphaV3(training_params.dropout_rate, training_params.batch_norm)
    elif training_params.model_architecture == "AlphaSelfAttention":
        model = SelfAttentionNet(4, training_params.batch_norm, training_params.dropout_rate, training_params.global_pooling)
    elif training_params.model_architecture == "AlphaAttention":
        model = AlphaV1_Attention(batch_norm=training_params.batch_norm)
    elif training_params.model_architecture == "AlphaSelfAttentionPositional":
        model = AlphaSelfAttentionPositionalNet(4, training_params.batch_norm, training_params.dropout_rate, training_params.global_pooling)
    elif training_params.model_architecture == "AlphaV2_1D":
        model = AlphaV2_1D(training_params.dropout_rate, training_params.batch_norm, global_pooling=training_params.global_pooling)
    elif training_params.model_architecture == "AlphaV2_1D_1":
        model = AlphaV2_1D_1(training_params.dropout_rate, training_params.batch_norm, global_pooling=training_params.global_pooling)
    else:
        model = AlphaV3_1(training_params.dropout_rate, training_params.batch_norm)

    model = train_model(model, train_loader, val_loader, training_params, log_folder)

    # Ensure the directory exists
    os.makedirs(model_path, exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), os.path.join(model_path, f"{training_params.model_name}.pth"))

    # Number of parameters in the model
    training_params.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    update_experiment_configs(log_folder, training_params)
