from torch import nn, optim
from tqdm import tqdm
import os
import wandb
from torch.utils.data import DataLoader
import torch


from src.Preprocessing.AudioLoader import AudioLoader
from src.Preprocessing.SpectrogramProcessor import SpectrogramProcessor
from src.Architectures.AlphaV2 import AlphaV2
from src.Preprocessing.BatFileDataSet import BatFileDataSet


def train_model(model, train_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()  # For categorical classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    wandb.watch(model, criterion, log="all", log_freq=10)  # Track gradients & parameters

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for spectrograms, labels in train_loader:
            labels = labels.to(device)
            spectrograms = [spec.to(device) for spec in spectrograms]

            optimizer.zero_grad()
            outputs = [model(spec.unsqueeze(0)) for spec in spectrograms]
            outputs = torch.cat(outputs, dim=0)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

        wandb.log({"epoch": epoch + 1, "loss": epoch_loss})  # Log epoch loss

    print("Training complete!")



def collate_fn(batch):
    # The batch is a list of (spectrogram, label) tuples
    spectrograms = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Stack the labels into a tensor
    labels = torch.stack(labels, dim=0)

    # Return the list of spectrograms and labels
    return spectrograms, labels


if __name__ == '__main__':
    # Set up paths
    spectrogram_already_computed = True
    example_data = False

    audioloader_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/ExampleData"
    spectrograms_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/ExampleSpectrograms"
    labels_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/Labels/labels_exampledata.xlsx"

    model_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/src/Models/Alpha"

    if not example_data:
        audioloader_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/DataAlpha/train"
        spectrograms_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/Spectrograms"
        labels_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/DataAlpha/dataset_info/train_dataset_info.xlsx"

    wandb.login(key="32b08e4c860b935b2cd9c30774889b952ffefe0d")

    run = wandb.init(
        project="ChiRO",
        entity="martin-faehnrich-university-of-z-rich",
        job_type="training",
        config={
            "notes": "TwoBatSpecies_AllEnv",
            "learning_rate": 0.001,  # TODO
            "dataset": "BatCalls-Environment",
            "num_epochs": 10,
            "batch_size": 8,
            "model": AlphaV2,
            "model_name": "alphaV2_01.pth",
        },
    )

    config = wandb.config

    if not spectrogram_already_computed:
        # Load Audio Files
        audio_loader = AudioLoader()
        audio_loader.load_folder(audioloader_path)
        waveforms = audio_loader.get_data()
        names = audio_loader.get_file_names_from_excel(labels_path)

        # Create Spectrograms
        for i in tqdm(range(len(waveforms)), desc="Creating Spectrograms"):
            sp = SpectrogramProcessor(waveforms[i])
            sp.apply_highpass_filter()
            sp.compute_spectrogram()
            sp.compute_mel_spectrogram()
            sp.denoise_spectrogram_mean_subtraction()
            sp.save_spectrogram(f'{names[i]}', spectrograms_path + '/')


    # Create Dataset & DataLoader
    dataset = BatFileDataSet(spectrograms_path, labels_path, filename_column="Filename", text_column="label")
    train_loader = tqdm(DataLoader(dataset, config.batch_size, shuffle=True, collate_fn=collate_fn), desc="Loading Data")

    # Initialize Model & Train
    model = AlphaV2()
    train_model(model, train_loader, config.num_epochs, config.learning_rate)

    # Ensure the directory exists
    os.makedirs(model_path, exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), os.path.join(model_path, config.model_name))
