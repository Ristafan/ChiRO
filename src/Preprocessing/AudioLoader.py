import glob
from pathlib import Path

import pandas as pd

from src.Logging.Logger import Logger
import torch
import torchaudio
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from src.Training.TrainingParams import DEVICE


class AudioLoader:
    def __init__(self):
        self.waveform = None
        self.sample_rate = None
        self.data = []
        self.device = torch.device(DEVICE)
        print(f"AudioLoader initialized for device: {self.device}", flush=True)

    def get_data(self):
        return self.data

    def load_wav_file(self, filepath):
        """Loads a single wav file."""
        waveform, sample_rate = torchaudio.load(filepath)
        waveform = waveform.to(self.device)
        return waveform, sample_rate

    def load_folder(self, data_path):
        """Load all WAV files from the folder and store them in the list."""
        file_paths = [entry.path for entry in os.scandir(data_path) if entry.name.lower().endswith('.wav')]

        with ThreadPoolExecutor() as executor:
            for waveform, sample_rate in tqdm(executor.map(self.load_wav_file, file_paths), total=len(file_paths), desc='Loading Audio Files'):
                self.data.append(waveform)

    def load_audio_from_exel(self, file_path):
        """Load audio files from an Excel file."""
        data = pd.read_excel(file_path)
        filepaths = data['Filepath'].tolist()

        with ThreadPoolExecutor() as executor:
            for filepath in tqdm(filepaths, desc='Loading Audio Files'):
                normalized_path = str(Path(filepath))  # Normalize path
                waveform, sample_rate = self.load_wav_file(normalized_path)
                self.data.append(waveform)

    @staticmethod
    def get_file_names_from_folder(data_path):
        return [filename for filename in os.listdir(data_path) if filename.lower().endswith('.wav')]

    @staticmethod
    def get_file_names_from_excel(file_path):
        data = pd.read_excel(file_path)
        return data['Filename'].tolist()


if __name__ == '__main__':
    data_loader = AudioLoader()
    data_loader.load_audio_from_exel('C:/Users/MartinFaehnrich/Documents/ChiRO/data/DataAlpha/dataset_info/train_dataset_info.xlsx')
