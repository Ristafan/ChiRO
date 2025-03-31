from src.Logging.Logger import Logger
import torch
import torchaudio
import os
from tqdm import tqdm


class AudioLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.file_path = None
        self.waveform = None
        self.sample_rate = None
        self.data = torch.tensor([])

    def get_data(self):
        return self.data

    def load_wav_file(self):
        self.waveform, self.sample_rate = torchaudio.load(self.file_path)

    def load_folder(self):
        self.data = []

        for file in tqdm(self.get_file_names(), desc='Loading Audio Files'):
            self.file_path = os.path.join(self.data_path, file)
            self.load_wav_file()
            Logger().log_debug(f'Loaded file {file}')
            self.data.append(self.waveform)

        Logger().log_debug(f'Data loaded from folder {self.data_path}')

    def get_file_names(self):
        return [filename for filename in os.listdir(self.data_path) if filename.endswith('.WAV')]


if __name__ == '__main__':
    data_loader = AudioLoader('C:/Users/MartinFaehnrich/Documents/ChiRO/data/ExampleData')
    data_loader.load_folder()
    print(data_loader.get_file_names())

