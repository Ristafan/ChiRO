import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler # Import for Automatic Mixed Precision
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from src.Architectures.AlphaResNet50 import AlphaResNet50, Bottleneck
from src.Architectures.AlphaV2 import AlphaV2
from src.Architectures.AlphaV2_1 import AlphaV2_1
from src.Architectures.AlphaV3 import AlphaV3
from src.Architectures.AlphaV3_1 import AlphaV3_1
from src.Preprocessing.Preprocessor import Preprocessor
from src.Training.TrainingParams import TrainingParams
from src.utils import load_path_config, update_experiment_configs, log_metrics, create_experiment_dir

# Set memory allocation configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def get_frames_from_seconds(time_in_seconds: float, sample_rate: int, hop_length: int) -> int:
    """
    Converts a time duration in seconds to the approximate number of spectrogram frames.
    This calculates how many 'hop_lengths' fit into the given time duration.
    """
    if hop_length <= 0:
        raise ValueError("hop_length must be a positive integer.")

    frames_per_second_spectrogram = sample_rate / hop_length
    return int(time_in_seconds * frames_per_second_spectrogram)


def cut_tensor_into_pieces(
        tensor: torch.Tensor,
        sample_rate: int,
        hop_length: int,
        window_size_s: float,
        overlap_size_s: float
) -> list[torch.Tensor]:
    """
    Cuts a 3D tensor [channels, frequency_bins, time_frames] into overlapping pieces
    based on time durations. Each piece is padded with zeros if it's shorter than the
    specified window_size_s.

    Args:
        tensor (torch.Tensor): The input tensor. Expected shape: [channels, frequency_bins, time_frames].
        sample_rate (int): The sample rate of the audio data.
        hop_length (int): The hop length used to create the spectrogram.
        window_size_s (float): The desired size of each cut piece in seconds.
        overlap_size_s (float): The desired overlap size in seconds.

    Returns:
        list[torch.Tensor]: A list of torch.Tensors, where each is a cut piece
                            of shape [channels, frequency_bins, window_frames].
    """

    if tensor.ndim != 3:
        raise ValueError("Input tensor must be a 3D tensor with shape [channels, frequency_bins, time_frames].")

    if not (sample_rate > 0 and hop_length > 0):
        raise ValueError("sample_rate and hop_length must be positive integers.")

    if not (window_size_s > 0):
        raise ValueError("window_size_s must be a positive float.")

    if not (0 <= overlap_size_s < window_size_s):
        raise ValueError("overlap_size_s must be non-negative and strictly less than window_size_s.")

    total_time_frames = tensor.shape[2]

    # Convert window and overlap sizes from seconds to spectrogram frames
    window_frames = get_frames_from_seconds(window_size_s, sample_rate, hop_length)
    overlap_frames = get_frames_from_seconds(overlap_size_s, sample_rate, hop_length)

    # Ensure window_frames is at least 1, as a window must span at least one frame.
    if window_frames == 0:
        print(f"Warning: Calculated window_frames is 0 for window_size_s={window_size_s}s. Setting to 1 frame for cutting.")
        window_frames = 1

    # Calculate the effective hop (step size) between the start of consecutive windows in frames.
    cutting_hop_frames = window_frames - overlap_frames

    # Ensure cutting_hop_frames is positive to ensure progress and prevent infinite loops.
    if cutting_hop_frames <= 0:
        print(f"Warning: Calculated cutting_hop_frames is {cutting_hop_frames}. Forcing to 1 frame to ensure progress.")
        cutting_hop_frames = 1

    cut_pieces = []
    current_start_frame = 0

    while True: # Loop indefinitely, breaking out when no more valid starting points
        # Determine the actual end frame for the current slice.
        # This will be `current_start_frame + window_frames` or `total_time_frames`, whichever is smaller.
        actual_end_frame_for_slice = min(current_start_frame + window_frames, total_time_frames)

        # If the start frame is beyond or equal to the total frames, we are done.
        # This handles cases where the last hop takes us completely beyond the data.
        if current_start_frame >= total_time_frames:
            break

        # Slice the piece from the tensor
        cut_piece = tensor[:, :, current_start_frame:actual_end_frame_for_slice]

        # Pad the cut_piece if its time dimension is shorter than the required window_frames.
        # This ensures all output pieces have a consistent time dimension for the CNN.
        if cut_piece.shape[2] < window_frames:
            padding_needed = window_frames - cut_piece.shape[2]
            # F.pad expects padding for the last dimension as (padding_left, padding_right).
            # Here, we pad `padding_needed` zeros to the right of the time dimension.
            cut_piece = F.pad(cut_piece, (0, padding_needed))

        # Add the padded piece to the list.
        # A piece might be empty if `window_frames` is 0 or if `current_start_frame` is too large initially.
        if cut_piece.shape[2] > 0:
            cut_pieces.append(cut_piece)

        # Move to the next start frame for the next window.
        current_start_frame += cutting_hop_frames

        # Additional break condition: If the next start frame would be beyond the end of the total frames
        # AND we've already processed a piece that included the very end of the original tensor,
        # then we can stop to avoid processing redundant or empty slices.
        if current_start_frame >= total_time_frames and actual_end_frame_for_slice == total_time_frames:
            break

    return cut_pieces


def train_section_dynamic_alpha(model, train_loader, val_loader, config, log_folder, sample_rate=192000, hop_length=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(reduction='none')

    # Optimizer
    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    window_size_s = config.window_size
    overlap_size_s = config.overlap_size
    loss_filter_threshold_percentage = config.loss_filter_threshold_percentage
    num_epochs = config.num_epochs

    # This dictionary will store which sections of each full spectrogram to train on
    # Key: original_spectrogram_idx (from train_loader)
    # Value: A set of tuples (original_section_idx_within_spectrogram)
    # This set will contain indices of "good" sections from `all_possible_sections`.
    spectrogram_section_map = {}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        best_val_acc = 0
        patience_counter = 0

        # Lists to store losses for filtering at the end of the epoch
        # (spectrogram_idx, section_original_index, loss_value)
        epoch_section_loss_records = []

        # Add progress bar for each epoch
        # train_loader now yields full spectrograms. We'll cut them inside the loop.
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for batch_idx_full_spec, (full_spectrograms_batch, full_labels_batch) in train_pbar:
            # Move full spectrograms to device (they'll be cut and then sections processed)
            full_spectrograms_batch = full_spectrograms_batch.to(device)
            full_labels_batch = full_labels_batch.to(device)

            # --- Dynamic Cutting and Batching of Sections ---

            # This list will hold the actual sections and their corresponding labels and original IDs
            # (section_tensor, section_label_tensor, original_spectrogram_idx, section_idx_within_orig_spec)
            active_sections_for_batch = []

            for i in range(full_spectrograms_batch.shape[0]): # Iterate over each full spectrogram in the batch
                original_spectrogram_idx = batch_idx_full_spec * train_loader.batch_size + i
                full_spectrogram = full_spectrograms_batch[i]
                full_label = full_labels_batch[i] # Assuming label applies to all sections of this spectrogram

                # Cut the current full spectrogram into all possible smaller pieces
                all_possible_sections = cut_tensor_into_pieces(
                    full_spectrogram, # Use the full spectrogram from the batch
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                    window_size_s=window_size_s,
                    overlap_size_s=overlap_size_s
                )

                # Determine which sections to use based on the filtering from the previous epoch
                if epoch == 0 or original_spectrogram_idx not in spectrogram_section_map:
                    # In the first epoch, or if this spectrogram was never processed (unlikely with shuffle=True), use all sections
                    for section_idx, section_data in enumerate(all_possible_sections):
                        active_sections_for_batch.append((section_data, full_label, original_spectrogram_idx, section_idx))
                else:
                    # Use only the "good" sections identified in the previous epoch
                    good_section_indices = spectrogram_section_map[original_spectrogram_idx]
                    for section_idx in good_section_indices:
                        if section_idx < len(all_possible_sections): # Safety check
                            section_data = all_possible_sections[section_idx]
                            active_sections_for_batch.append((section_data, full_label, original_spectrogram_idx, section_idx))

            # --- Now process these active_sections_for_batch in mini-batches for the CNN ---
            # We will create internal mini-batches of sections
            if not active_sections_for_batch:
                continue # Skip if no active sections for this full_spectrograms_batch

            # Shuffle active_sections_for_batch to mix sections from different original spectrograms
            np.random.shuffle(active_sections_for_batch)

            # Process sections in mini-batches (e.g., using your global BATCH_SIZE)
            for section_batch_start_idx in range(0, len(active_sections_for_batch), train_loader.batch_size):
                section_batch_end_idx = min(section_batch_start_idx + train_loader.batch_size, len(active_sections_for_batch))
                current_section_minibatch = active_sections_for_batch[section_batch_start_idx:section_batch_end_idx]

                if not current_section_minibatch:
                    continue

                # Prepare the actual tensors for the model
                input_sections = torch.stack([s[0] for s in current_section_minibatch]).to(device)
                target_labels = torch.stack([s[1] for s in current_section_minibatch]).to(device)
                original_section_info = [(s[2], s[3]) for s in current_section_minibatch] # (orig_spec_idx, section_idx)

                optimizer.zero_grad()

                with autocast(): # Use automatic mixed precision
                    outputs = model(input_sections)
                    # Use reduction='none' to get individual losses for each section in the batch
                    individual_losses = criterion(outputs, target_labels)
                    loss = individual_losses.mean() # Calculate mean loss for backprop

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len(current_section_minibatch) # Accumulate total loss correctly for all sections processed

                # Calculate accuracy for this mini-batch of sections
                _, predicted = torch.max(outputs.data, 1)
                total += target_labels.size(0)
                correct += (predicted == target_labels).sum().item()

                # Record individual section losses for filtering
                with torch.no_grad(): # Don't track gradients for loss recording
                    for k, (orig_spec_idx, sec_idx) in enumerate(original_section_info):
                        epoch_section_loss_records.append((orig_spec_idx, sec_idx, individual_losses[k].item()))

                # Update progress bar
                train_pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})

                # Clear memory every few batches if needed
                # This check now applies to the internal section mini-batches
                if section_batch_start_idx % (train_loader.batch_size * 5) == 0: # Check less frequently
                    torch.cuda.empty_cache()

        # --- End of Epoch: Calculate Average Loss and Filter Sections ---

        # Calculate total average loss over all *processed* sections in this epoch
        if epoch_section_loss_records:
            total_sections_processed = len(epoch_section_loss_records)
            sum_of_all_section_losses = sum([record[2] for record in epoch_section_loss_records])
            average_epoch_loss = sum_of_all_section_losses / total_sections_processed
            print(f"Epoch {epoch+1} Average Section Loss: {average_epoch_loss:.4f}")

            # Prepare `spectrogram_section_map` for the NEXT epoch
            next_epoch_spectrogram_section_map = {}
            for spec_id, section_idx, loss_val in epoch_section_loss_records:
                if loss_val <= loss_filter_threshold_percentage * average_epoch_loss:
                    # This section is "good", include it for the next epoch
                    if spec_id not in next_epoch_spectrogram_section_map:
                        next_epoch_spectrogram_section_map[spec_id] = set() # Use a set for efficient lookup
                    next_epoch_spectrogram_section_map[spec_id].add(section_idx)

            spectrogram_section_map = next_epoch_spectrogram_section_map
            print(f"Number of 'good' spectrograms for next epoch: {len(spectrogram_section_map)}")

        else:
            print("No sections were processed in this epoch. Check data or filtering logic.")
            # If no sections, ensure no filtering happens or handle gracefully
            spectrogram_section_map = {} # Clear for next epoch, or keep as is if this means all are bad

        train_loss_epoch_avg = running_loss / max(1, total_sections_processed) # Avoid division by zero
        train_acc_epoch_avg = 100 * correct / max(1, total)

        print(f"Epoch [{epoch+1}/{num_epochs}], Accumulated Train Loss: {train_loss_epoch_avg:.4f}, Accumulated Train Accuracy: {train_acc_epoch_avg:.2f}%")

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_total = 0.0

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss_total += loss.sum().item()

                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc_epoch_avg = val_correct / val_total if val_total > 0 else 0.0

        # Log to Json
        config_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss_epoch_avg,
            "sections_processed": total_sections_processed,
            "val_loss": val_loss_total / max(1, val_total),
            "train_accuracy": train_acc_epoch_avg,
            "val_accuracy": val_acc_epoch_avg,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "average_loss_epoch": average_epoch_loss if epoch_section_loss_records else None,
            "good_sections": sum(len(v) for v in spectrogram_section_map.values())
        }

        log_metrics(log_folder, epoch, config_dict)

        # Early stopping based on validation_accuracy
        if config.early_stopping:
            if val_acc_epoch_avg > best_val_acc:
                best_val_acc = val_acc_epoch_avg
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
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
    else:
        model = AlphaV3_1(training_params.dropout_rate, training_params.batch_norm)

    model = train_section_dynamic_alpha(model, train_loader, val_loader, training_params, log_folder)

    # Ensure the directory exists
    os.makedirs(model_path, exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), os.path.join(model_path, training_params.model_name))

    # Number of parameters in the model
    training_params.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    update_experiment_configs(log_folder, training_params)
