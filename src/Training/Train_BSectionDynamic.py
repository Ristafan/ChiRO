import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from src.Architectures.GenusClassification.BetaV3 import BetaV3
from src.Preprocessing.Preprocessor import Preprocessor
from src.Training.TrainingParams import TrainingParams
from src.utils import load_path_config, update_experiment_configs, log_metrics, create_experiment_dir

# Set memory allocation configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def get_frames_from_seconds(time_in_seconds: float, sample_rate: int, hop_length: int) -> int:
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
):

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
        actual_end_frame_for_slice = min(current_start_frame + window_frames, total_time_frames)

        # If the start frame is beyond or equal to the total frames, we are done.
        if current_start_frame >= total_time_frames:
            break

        # Slice the piece from the tensor
        cut_piece = tensor[:, :, current_start_frame:actual_end_frame_for_slice]

        # Pad the cut_piece if its time dimension is shorter than the required window_frames.
        if cut_piece.shape[2] < window_frames:
            padding_needed = window_frames - cut_piece.shape[2]
            # Here, we pad `padding_needed` zeros to the right of the time dimension.
            cut_piece = F.pad(cut_piece, (0, padding_needed))

        # Add the padded piece to the list.
        if cut_piece.shape[2] > 0:
            cut_pieces.append(cut_piece)

        # Move to the next start frame for the next window.
        current_start_frame += cutting_hop_frames

        # Additional break condition: If the next start frame would be beyond the end of the total frames
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
    spectrogram_section_map = {}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        best_val_acc = 0
        patience_counter = 0

        # Lists to store losses for filtering at the end of the epoch
        epoch_section_loss_records = []

        # Add progress bar for each epoch
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for batch_idx_full_spec, (full_spectrograms_batch, full_labels_batch) in train_pbar:
            # Move full spectrograms to device (they'll be cut and then sections processed)
            full_spectrograms_batch = full_spectrograms_batch.to(device)
            full_labels_batch = full_labels_batch.to(device)

            # This list will hold the actual sections and their corresponding labels and original IDs
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
                for k, (orig_spec_idx, sec_idx) in enumerate(original_section_info):
                    epoch_section_loss_records.append((orig_spec_idx, sec_idx, individual_losses[k].item()))

                # Update progress bar
                train_pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})

                # Clear memory every few batches if needed
                if section_batch_start_idx % (train_loader.batch_size * 5) == 0: # Check less frequently
                    torch.cuda.empty_cache()


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

    train_files_and_labels_path = config['dataset']['train_files_and_labels_path_beta']
    validation_files_and_labels_path = config['dataset']['validation_files_and_labels_path_beta']
    original_files_and_labels_path = config['dataset']['original_files_and_labels_path']
    root_files_path = config['dataset']['files_path_root']
    spectrograms_path = config['spectrogram']['spectrograms_dir']
    model_path = config['model']['beta']
    runs_dir_alpha = config['logs']['runs_dir_beta']
    pretrained_model_path = config['model']['pretrained_alphaV3']

    # Initialize logging
    log_folder = create_experiment_dir(training_params, runs_dir_alpha)

    # Load Audio Files, Labels and create spectrograms
    preprocessor = Preprocessor(train_files_and_labels_path, spectrograms_path, root_files_path)

    if not training_params.splits_already_computed:
        _ = preprocessor.create_data_splits(original_files_and_labels_path, training_params.merge_labels, training_params.ignored_labels, training_params.seed, training_params.total_files_per_class, training_params.use_min_files_per_class, training_params.split_method, training_params.split_ratios)

    if not training_params.spectrograms_already_computed:
        preprocessor.create_spectrograms_stft(train_files_and_labels_path)
        preprocessor.create_spectrograms_stft(validation_files_and_labels_path)

    train_dataset = preprocessor.create_bat_file_dataset(train_files_and_labels_path)
    train_loader = DataLoader(train_dataset, batch_size=training_params.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=1, pin_memory=True)
    val_dataset = preprocessor.create_bat_file_dataset(validation_files_and_labels_path)
    val_loader = DataLoader(val_dataset, batch_size=training_params.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=1, pin_memory=True)

    beta_model = BetaV3(num_genera=training_params.num_classes,
                             dropout_rate=training_params.dropout_rate,
                             batch_norm=training_params.batch_norm,
                             global_pooling=training_params.global_pooling)
    alpha_state_dict = torch.load(pretrained_model_path)

    beta_model_state_dict = beta_model.state_dict()
    filtered_state_dict = {
        k: v for k, v in alpha_state_dict.items()
        if k in beta_model_state_dict and beta_model_state_dict[k].shape == v.shape
    }
    beta_model.load_state_dict(filtered_state_dict, strict=False)
    for name, param in beta_model.named_parameters():
        if "fc2" not in name:
            param.requires_grad = False

    model = train_section_dynamic_alpha(beta_model, train_loader, val_loader, training_params, log_folder)

    # Ensure the directory exists
    os.makedirs(model_path, exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), os.path.join(model_path, f"{training_params.model_name}.pth"))

    # Number of parameters in the model
    training_params.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    update_experiment_configs(log_folder, training_params)
