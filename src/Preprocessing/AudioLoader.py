from src.Logging.Logger import Logger
import torch
import torchaudio
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


class AudioLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.waveform = None
        self.sample_rate = None
        self.data = []

    def get_data(self):
        return self.data

    def load_wav_file(self, filename):
        """Loads a single wav file."""
        filepath = os.path.join(self.data_path, filename)
        return torchaudio.load(filepath)

    def load_folder(self):
        """Load all WAV files from the folder and store them in the list."""
        file_paths = [entry.path for entry in os.scandir(self.data_path) if entry.name.lower().endswith('.wav')]

        with ThreadPoolExecutor() as executor:
            for waveform, sample_rate in tqdm(executor.map(self.load_wav_file, file_paths), total=len(file_paths), desc='Loading Audio Files'):
                Logger().log_debug(f'Loaded file {os.path.basename(file_paths[len(self.data)])}')
                self.data.append(waveform)

        Logger().log_debug(f'Data loaded from folder {self.data_path}')

    def get_file_names(self):
        return [filename for filename in os.listdir(self.data_path) if filename.lower().endswith('.wav')]


if __name__ == '__main__':
    data_loader = AudioLoader('C:/Users/MartinFaehnrich/Documents/ChiRO/data/ExampleData')
    data_loader.load_folder()
    print(data_loader.get_file_names())

