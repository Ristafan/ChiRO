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
        spectrogram = torch.load(spectrogram_path)

        if spectrogram.dim() == 3:  # Fix unwanted extra batch dimensions
            spectrogram = spectrogram.squeeze(0)

        spectrogram = spectrogram.unsqueeze(0)  # Ensure correct shape: [1, height, width]

        label = torch.tensor(self.labels[excel_filename], dtype=torch.long)
        return spectrogram, label

    def get_labels(self):
        labels = []
        for filename in self.file_names:
            labels.append(self.labels[filename])
        return labels


if __name__ == '__main__':
    # Example usage
    spec_dir = 'C:/Users/MartinFaehnrich/Documents/ChiRO/data/Spectrograms'
    lab_loader = LabelsLoader('C:/Users/MartinFaehnrich/Documents/ChiRO/data/Labels/labels.xlsx')
    lab_loader.load_labels_excel()

    dataset = BatCallDataset(spec_dir, lab_loader)
    print(f"Dataset size: {len(dataset)}")
    spec, lab = dataset[0]
    print(f"Spectrogram shape: {spec.shape}")
    print(f"Label: {lab}")
