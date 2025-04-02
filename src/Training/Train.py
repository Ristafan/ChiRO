from torch import nn, optim
from tqdm import tqdm
import os
import wandb
from torch.utils.data import DataLoader

from src.Logging.Logger import Logger
from src.Preprocessing.AudioLoader import AudioLoader
from src.Preprocessing.LabelsLoader import LabelsLoader
from src.Preprocessing.SpectrogramProcessor import SpectrogramProcessor
from src.Preprocessing.SpectrogramLoader import SpectrogramLoader
from src.Architectures.AlphaV1 import AlphaV1
from src.Preprocessing.BatCallDataSet import BatCallDataset


import torch


def train_model(model, train_loader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()  # For binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for spectrograms, labels in train_loader:
            # Move labels to the device (this part is fine)
            labels = labels.to(device)

            # Move each spectrogram in the batch to the device
            spectrograms = [spec.to(device) for spec in spectrograms]

            # Process the spectrograms (this assumes your model can handle a list of tensors)
            optimizer.zero_grad()

            # Pass each spectrogram to the model one by one
            outputs = []
            for spec in spectrograms:
                output = model(spec.unsqueeze(0))  # Add batch dimension
                outputs.append(output)

            # Concatenate outputs and labels
            outputs = torch.cat(outputs, dim=0)

            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

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
    spectrogram_already_computed = False
    audioloader_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/Training"
    spectrograms_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/Spectrograms"
    labels_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/Labels/labels.xlsx"
    model_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/src/Models/Alpha"
    model_name = "alphaV1.pth"
    batch_size = 4
    num_epochs = 10
    learning_rate = 0.001  # TODO
    dataset_name = "BatCallDataset"

    # Start wandb run
    run = wandb.init(
        project="ChiRO",
        entity="MartinF",
        job_type="training",
        config={
            "learning_rate": learning_rate,
            "architecture": model_name,
            "dataset": dataset_name,
            "num_epochs": num_epochs,
            "batch_size": batch_size
        },
    )

    if not spectrogram_already_computed:
        # Load Audio Files
        audio_loader = AudioLoader(audioloader_path)
        audio_loader.load_folder()
        waveforms = audio_loader.get_data()
        names = audio_loader.get_file_names()

        # Create Spectrograms
        for i in tqdm(range(len(waveforms)), desc="Creating Spectrograms"):
            sp = SpectrogramProcessor(waveforms[i])
            sp.apply_highpass_filter()
            sp.compute_spectrogram()
            sp.compute_mel_spectrogram()
            sp.denoise_spectrogram()
            sp.save_spectrogram(f'{names[i]}', spectrograms_path + '/')

    # Load labels from Excel
    labels_loader = LabelsLoader(labels_path, filename_column="Filename", text_column="Label")
    labels_loader.load_labels_excel()

    # Create Dataset & DataLoader
    dataset = BatCallDataset(spectrograms_path, labels_loader)
    train_loader = tqdm(DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn), desc="Loading Data")

    # Initialize Model & Train
    model = AlphaV1()
    train_model(model, train_loader, num_epochs)

    # Ensure the directory exists
    os.makedirs(model_path, exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), os.path.join(model_path, model_name))

