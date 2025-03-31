from src.Logging.Logger import Logger

import torch
import os


class SpectrogramLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.file_path = None
        self.spectrogram = None
        self.data = torch.tensor([])

    def get_data(self):
        return self.data

    def load_spectrogram(self, file_path):
        self.spectrogram = torch.load(file_path)

    def load_folder(self):
        for file in self.get_file_names():
            self.file_path = os.path.join(self.data_path, file)
            self.load_spectrogram(self.file_path)
            Logger().log_debug(f'Loaded file {file}')
            self.data = torch.cat((self.data, self.spectrogram), 0)

        Logger().log_debug(f'Data loaded from folder {self.data_path}')

    def get_file_names(self):
        return [filename for filename in os.listdir(self.data_path) if filename.endswith('.pt')]


if  __name__ == '__main__':
    data_loader = SpectrogramLoader('C:/Users/MartinFaehnrich/Documents/ChiRO/data/Spectrograms')
    data_loader.load_folder()
    data = data_loader.get_data()
    print(data_loader.data)
    print(data)
    print(data.shape)
    print(data[0].shape)

