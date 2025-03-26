from src.Logging.Logger import Logger
import torch
import torchaudio
import os


class DataLoader:
    def __init__(self, data_path, file_path=None):
        self.data_path = data_path
        self.file_path = None
        self.waveform = None
        self.sample_rate = None
        self.data = torch.tensor([])
        self.file_names = [filename for filename in os.listdir(data_path) if filename.endswith('.WAV')]

    def get_data(self):
        return self.data

    def load_wav_file(self):
        self.waveform, self.sample_rate = torchaudio.load(self.file_path)

    def load_folder(self):
        for file in self.file_names:
            self.file_path = os.path.join(self.data_path, file)
            self.load_wav_file()
            Logger().log_debug(f'Loaded file {file}')
            self.data = torch.cat((self.data, self.waveform), 0)

        Logger().log_debug(f'Data loaded from folder {self.data_path}')


if __name__ == '__main__':
    data_loader = DataLoader('C:/Users/MartinFaehnrich/Documents/ChiRO/data/ExampleData')
    data_loader.load_folder()
    print(data_loader.data)

