import os
import torch
from torch.utils.data import Dataset

from src.Preprocessing.LabelsLoader import LabelsLoader


class BatCallDataset(Dataset):
    def __init__(self, spectrogram_dir, labels_loader):
        """
        :param spectrogram_dir: Path to folder containing spectrogram .pt files
        :param labels_loader: Instance of LabelsLoader containing filename-label mappings
        """
        self.spectrogram_dir = spectrogram_dir
        self.labels = labels_loader.get_labels()
        self.file_names = list(self.labels.keys())

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        excel_filename = self.file_names[idx]
        filename = f"spectrogram_{self.file_names[idx]}.WAV.pt"
        spectrogram_path = os.path.join(self.spectrogram_dir, filename)
        spectrogram = torch.load(spectrogram_path)  # Load precomputed spectrogram
        spectrogram = spectrogram.unsqueeze(0)  # Add channel dimension for CNN

        label = torch.tensor(self.labels[excel_filename], dtype=torch.long)
        return spectrogram, label


if __name__ == '__main__':
    # Example usage
    spectrogram_dir = 'C:/Users/MartinFaehnrich/Documents/ChiRO/data/Spectrograms'
    labels_loader = LabelsLoader('C:/Users/MartinFaehnrich/Documents/ChiRO/data/Labels/labels.xlsx')  # Assume LabelsLoader is implemented elsewhere
    labels_loader.load_labels_excel()

    dataset = BatCallDataset(spectrogram_dir, labels_loader)
    print(f"Dataset size: {len(dataset)}")
    spectrogram, label = dataset[0]
    print(f"Spectrogram shape: {spectrogram.shape}")
    print(f"Label: {label}")
