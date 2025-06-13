import json
import os
import time

import torch
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
from torchaudio.transforms import AmplitudeToDB


def plot_spectrogram(spectrogram):

    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram.squeeze(0).numpy(), aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (frames)')
    plt.ylabel('Frequency (bins)')
    plt.tight_layout()
    plt.show()


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
        start_frame = self.start_times[filename_with_call_index]
        end_frame = self.end_times[filename_with_call_index]
        print("Spectrogram shape before slicing:", spectrogram.shape)
        spectrogram = spectrogram[:, start_frame:end_frame, :]
        print("Spectrogram shape after slicing:", spectrogram.shape)

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
                    start_time = [t for t in start_time if t != ""]
                    start_time = [self.convert_time_to_frames(float(t)) for t in start_time]

            if isinstance(end_time, str):
                if end_time == "[]":
                    end_time = []
                else:
                    end_time = end_time.strip("[]").split(" ")
                    end_time = [t for t in end_time if t != ""]
                    end_time = [self.convert_time_to_frames(float(t)) for t in end_time]

            self.start_times[filename] = start_time
            self.end_times[filename] = end_time

    @staticmethod
    def convert_time_to_frames(time, sample_rate=192000, win_length=2048, hop_length=None):
        if hop_length is None:
            hop_length = win_length // 2
        return int(time * sample_rate / hop_length) - int(win_length / hop_length)

    def create_filenames_with_calls(self):
        for filename in self.filenames:
            idx = 0
            if self.num_calls[filename] > 0:
                for i in range(self.num_calls[filename]):
                    self.filenames_with_calls.append(f"{filename}-{idx}")
                    self.start_times[f"{filename}-{idx}"] = self.start_times[filename][i]
                    self.end_times[f"{filename}-{idx}"] = self.end_times[filename][i]
                    idx += 1


def plot_dataset_item(dataset: BatCallDataSet, idx: int):
    """
    Plot the full spectrogram for item idx with the call region highlighted,
    and then plot just the extracted slice.

    :param dataset: your BatCallDataSet instance
    :param idx: index into dataset
    """
    # 1. Figure out which file & call this is
    key = dataset.filenames_with_calls[idx]          # e.g. "20210712_233400-0"
    base_fname = key.split("-")[0]                   # e.g. "20210712_233400"
    spec_path = os.path.join(dataset.spectrogram_dir,
                             f"spectrogram_{base_fname}.pt")

    # 2. Load the full spectrogram
    full_spec = torch.load(spec_path)                # [1,H,W] or [H,W]
    if full_spec.dim() == 3:
        full_spec = full_spec.squeeze(0)             # [H,W]

    # 3. Get the slice indices
    start = int(dataset.start_times[key])
    end = int(dataset.end_times[key])

    # transform = AmplitudeToDB(stype="power", top_db=80)
    # spec_db = transform(full_spec.unsqueeze(0)).squeeze(0)

    # 4. Plot the full spectrogram
    plt.figure(figsize=(12, 4))
    # plt.imshow(spec_db.numpy(), aspect="auto", origin="lower", cmap="magma")
    plt.imshow(full_spec.numpy(), aspect="auto", origin="lower")
    plt.axvline(start, color="r", linestyle="--", label="start")
    plt.axvline(end,   color="r", linestyle="-.", label="end")
    plt.title(f"Full spectrogram: {base_fname} (item {idx})")
    plt.xlabel("Time frames")
    plt.ylabel("Frequency bins")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 5. Plot just the slice
    # slice_spec = spec_db[:, start:end]
    slice_spec = full_spec[:, start:end]
    plt.figure(figsize=(8, 4))
    plt.imshow(slice_spec.numpy(), aspect="auto", origin="lower")
    plt.title(f"Sliced spectrogram: frames {start}–{end}")
    plt.xlabel("Time frames (slice)")
    plt.ylabel("Frequency bins")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    spectrogram_dir = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/spectrograms"
    labels_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/Beta/train_dataset_info.xlsx"

    dataset = BatCallDataSet(spectrogram_dir, labels_path, "Filename", "label")
    print(f"Total samples: {len(dataset)}")

    # Access a sample
    spectrogram, label = dataset[2]
    print(f"Spectrogram shape: {spectrogram.shape}, Label: {label}")

    # Plot calls from specific file
    filename = "20220617_224400T #0002_1c1b59d86384197e22d3eec015a33ede"
    calls = []
    for call in dataset.filenames_with_calls:
        if call.startswith(filename):
            calls.append(call)

    # Plot all calls for the specified file
    for call in calls:
        time.sleep(0.1)
        plot_dataset_item(dataset, dataset.filenames_with_calls.index(call))

    # Plot some dataset items
    # for i in range(6, 10):
    #     plot_dataset_item(dataset, i)
