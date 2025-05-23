import os

import torch
from torch.utils.data import Dataset
import pandas as pd


class BatCallDataSet(Dataset):
    def __init__(self, spectrogram_dir, labels_path, filename_column="Filename", label_column="Label"):
        """
        :param spectrogram_dir: Path to folder containing spectrogram .pt files
        :param labels_path: Path to Excel file containing labels
        :param filename_column: Column name in Excel file for filenames
        :param label_column: Column name in Excel file for labels
        """
        self.spectrogram_dir = spectrogram_dir
        self.labels_path = labels_path
        self.filename_column = filename_column
        self.label_column = label_column

        self.filenames = []
        self.filenames_with_calls = []
        self.labels = {}
        self.num_calls = {}
        self.start_times = {}
        self.end_times = {}

        # Load all spectrogram filenames
        self.load_excel()
        self.create_filenames_with_calls()

    def __len__(self):
        sum_of_calls = 0
        for calls in self.num_calls.values():
            sum_of_calls += calls

        return sum_of_calls

    def __getitem__(self, idx):
        filename_with_call_index = self.filenames_with_calls[idx]
        filename = f"spectrogram_{filename_with_call_index.split("-")[0]}.pt"
        spectrogram_path = os.path.join(self.spectrogram_dir, filename)
        spectrogram = torch.load(spectrogram_path)

        if spectrogram.dim() == 3:  # Fix unwanted extra batch dimensions
            spectrogram = spectrogram.squeeze(0)

        spectrogram = spectrogram.unsqueeze(0)  # Ensure correct shape: [1, height, width]

        # Cut the spectrogram to the start and end times
        start_time = self.start_times[filename_with_call_index]
        end_time = self.end_times[filename_with_call_index]
        spectrogram = spectrogram[:, start_time:end_time, :]

        label = torch.tensor(self.labels[filename_with_call_index.split("-")[0]], dtype=torch.long)

        return spectrogram, label

    def get_labels(self):
        labels = []
        for filename in self.filenames:
            labels.append(self.labels[filename])
        return labels

    def load_excel(self):
        data = pd.read_excel(self.labels_path)
        self.filenames = data[self.filename_column].tolist()
        self.labels = {row[self.filename_column]: int(row[self.label_column]) for _, row in data.iterrows()}
        self.num_calls = {row[self.filename_column]: int(row["num_calls"]) for _, row in data.iterrows()}
        start_times = {row[self.filename_column]: row["start_time"] for _, row in data.iterrows()}
        end_times = {row[self.filename_column]: row["end_time"] for _, row in data.iterrows()}

        # Convert start and end times to lists of floats and convert to frames
        for filename in self.filenames:
            start_time = start_times[filename]
            end_time = end_times[filename]

            if isinstance(start_time, str):
                if start_time == "[]":
                    start_time = []
                else:
                    start_time = start_time.strip("[]").split(" ")
                    start_time = [self.convert_time_to_frames(float(t)) for t in start_time]
            if isinstance(end_time, str):
                if end_time == "[]":
                    end_time = [self.convert_time_to_frames(float(t)) for t in end_time]
                else:
                    end_time = end_time.strip("[]").split(" ")

            self.start_times[filename] = start_time
            self.end_times[filename] = end_time

    @staticmethod
    def convert_time_to_frames(time, sample_rate=192000, win_length=2048, hop_length=None):
        if hop_length is None:
            hop_length = win_length // 2
        return int(time * sample_rate / hop_length)

    def create_filenames_with_calls(self):
        for filename in self.filenames:
            idx = 0
            if self.num_calls[filename] > 0:
                for i in range(self.num_calls[filename]):
                    self.filenames_with_calls.append(f"{filename}-{idx}")
                    self.start_times[f"{filename}-{idx}"] = self.start_times[filename][i]
                    self.end_times[f"{filename}-{idx}"] = self.end_times[filename][i]
                    idx += 1
