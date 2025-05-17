import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class BatCallDataset(Dataset):
    def __init__(self, spectrogram_dir, labels_path, filename_column="Filename", text_column="Label"):
        """
        :param spectrogram_dir: Path to folder containing spectrogram .pt files
        :param labels_path: Path to Excel file containing labels
        :param filename_column: Column name in Excel file for filenames
        :param text_column: Column name in Excel file for labels
        """
        self.spectrogram_dir = spectrogram_dir
        self.labels_path = labels_path
        self.filename_column = filename_column
        self.text_column = text_column

        self.filenames = []
        self.labels = {}

        # Load all spectrogram filenames
        self.load_labels_excel()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        excel_filename = self.filenames[idx]
        filename = f"spectrogram_{self.filenames[idx]}.pt"
        spectrogram_path = os.path.join(self.spectrogram_dir, filename)
        spectrogram = torch.load(spectrogram_path)

        if spectrogram.dim() == 3:  # Fix unwanted extra batch dimensions
            spectrogram = spectrogram.squeeze(0)

        spectrogram = spectrogram.unsqueeze(0)  # Ensure correct shape: [1, height, width]

        label = torch.tensor(self.labels[excel_filename], dtype=torch.long)
        return spectrogram, label

    def get_labels(self):
        labels = []
        for filename in self.filenames:
            labels.append(self.labels[filename])
        return labels

    def load_labels_excel(self):
        data = pd.read_excel(self.labels_path)
        self.filenames = data[self.filename_column].tolist()
        self.labels = {row[self.filename_column]: int(row[self.text_column]) for _, row in data.iterrows()}


if __name__ == '__main__':
    # Example usage
    spec_dir = 'C:/Users/MartinFaehnrich/Documents/ChiRO/data/Spectrograms'
    labels_excel = 'C:/Users/MartinFaehnrich/Documents/ChiRO/data/Labels/labels_exampledata.xlsx'

    dataset = BatCallDataset(spec_dir, labels_excel)
    print(f"Dataset size: {len(dataset)}")
    spec, lab = dataset[0]  # Get the first item
    print(f"Spectrogram shape: {spec.shape}")
    print(f"Label: {lab}")
