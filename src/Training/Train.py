from torch import nn, optim
from tqdm import tqdm
import os
import wandb
from torch.utils.data import DataLoader
import torch


from src.Logging.Logger import Logger
from src.Preprocessing.AudioLoader import AudioLoader
from src.Preprocessing.LabelsLoader import LabelsLoader
from src.Preprocessing.SpectrogramProcessor import SpectrogramProcessor
from src.Preprocessing.SpectrogramLoader import SpectrogramLoader
from src.Architectures.AlphaV1 import AlphaV1
from src.Architectures.AlphaV2 import AlphaV2
from src.Preprocessing.BatCallDataSet import BatCallDataset


# Set memory allocation configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def train_model(model, train_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Enable mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    wandb.watch(model, criterion, log="all", log_freq=10)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Clear cache before each epoch
        torch.cuda.empty_cache()

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

            # Scale gradients and perform backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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
    import torch.nn.functional as F

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
    # Set up paths
    spectrogram_already_computed = False
    example_data = False

    # Paths for example data
    audioloader_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/ExampleData/train"
    spectrograms_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/ExampleData/Spectrograms"
    labels_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/ExampleData/dataset_info/train_dataset_info.xlsx"

    model_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/src/Models/Alpha"

    if not example_data:
        audioloader_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/DataAlpha/dataset_info/train_dataset_info.xlsx"
        spectrograms_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/Spectrograms"
        labels_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/DataAlpha/dataset_info/train_dataset_info.xlsx"

    wandb.login(key="32b08e4c860b935b2cd9c30774889b952ffefe0d")

    run = wandb.init(
        project="ChiRO",
        entity="martin-faehnrich-university-of-z-rich",
        job_type="training",
        config={
            "notes": "",
            "learning_rate": 0.001,
            "dataset": "Example-BatCalls-Environment",
            "num_epochs": 2,
            "batch_size": 8,
            "model": "AlphaV2",
            "model_name": "alphaV2_example.pth",
        },
    )

    config = wandb.config

    if not spectrogram_already_computed:
        # Load Audio Files and create spectrograms
        audio_loader = AudioLoader()
        audio_loader.load_audio_from_exel(audioloader_path)
        waveforms = audio_loader.get_data()
        names = audio_loader.get_file_names_from_excel(audioloader_path)

        # Create Spectrograms
        for i in tqdm(range(len(waveforms)), desc="Creating Spectrograms"):
            sp = SpectrogramProcessor(waveforms[i])
            sp.apply_highpass_filter()
            sp.compute_spectrogram()
            sp.denoise_spectrogram()
            sp.save_spectrogram(f'{names[i]}', spectrograms_path + '/')

    # Load training labels from Excel
    labels_loader = LabelsLoader(labels_path, filename_column="Filename", text_column="label")
    labels_loader.load_labels_excel()

    # Create training Dataset & DataLoader
    train_dataset = BatCallDataset(spectrograms_path, labels_loader)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=2)

    # Initialize Model
    model = AlphaV2()

    # Log model architecture
    wandb.log({"model_summary": str(model)})

    model = train_model(model, train_loader, num_epochs=config.num_epochs, learning_rate=config.learning_rate)

    # Ensure the directory exists
    os.makedirs(model_path, exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), os.path.join(model_path, config.model_name))

    # Also save a checkpoint with more information
    checkpoint = {
        'epoch': config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None,  # You'd capture this from train_model if needed
        'loss': None,  # You'd capture this from train_model if needed
        'config': {k: v for k, v in config.items()}
    }
    torch.save(checkpoint, os.path.join(model_path, 'checkpoint_' + config.model_name))

    # Number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run.config.update({"num_params": f'The number of params is {num_params}'})

    # Finish the run
    run.finish()
