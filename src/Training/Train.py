from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.Logging.Logger import Logger
from src.Preprocessing.AudioLoader import AudioLoader
from src.Preprocessing.LabelsLoader import LabelsLoader
from src.Preprocessing.SpectrogramProcessor import SpectrogramProcessor
from src.Preprocessing.SpectrogramLoader import SpectrogramLoader
from src.Architectures.AlphaV1 import AlphaV1
from src.Preprocessing.BatCallDataSet import BatCallDataset


import torch


def train_model(model, train_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Logger().log_debug(f"Training on {device}")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for spectrograms, labels in train_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

    print("Training complete!")


if __name__ == '__main__':
    # Load Audio Files
    audio_loader = AudioLoader("C:/Users/MartinFaehnrich/Documents/ChiRO/data/Training")
    audio_loader.load_folder()
    waveforms = audio_loader.get_data()
    names = audio_loader.get_file_names()

    # Create Spectrograms
    for i in tqdm(range(len(waveforms))):
        sp = SpectrogramProcessor(waveforms[i])
        sp.apply_highpass_filter()
        sp.compute_spectrogram()
        sp.compute_mel_spectrogram()
        sp.denoise_spectrogram()
        sp.save_spectrogram(f'{names[i]}', 'C:/Users/MartinFaehnrich/Documents/ChiRO/data/Spectrograms/')

    # Load labels from Excel
    labels_loader = LabelsLoader("C:/Users/MartinFaehnrich/Documents/ChiRO/data/Labels/labels.xlsx", filename_column="Filename", text_column="Label")
    labels_loader.load_labels_excel()

    # Create Dataset & DataLoader
    dataset = BatCallDataset("C:/Users/MartinFaehnrich/Documents/ChiRO/data/Spectrograms", labels_loader)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Initialize Model & Train
    model = AlphaV1()
    train_model(model, train_loader, num_epochs=10)

