from src.Logging.Logger import Logger
import torch
import torchaudio
import matplotlib.pyplot as plt
from src.Preprocessing.AudioLoader import AudioLoader


class Spectrogram:
    def __init__(self, data):
        self.data = data
        self.spectrogram = None

    def short_time_fourier_transform(self):
        self.spectrogram = torch.stft(self.data, n_fft=2048, hop_length=512)
        Logger().log_debug('Short time Fourier transform applied')

    def melscale_filterbank(self):
        transform = torchaudio.transforms.Spectrogram(n_fft=1024, win_length=1024, hop_length=512)
        self.spectrogram = transform(self.data)
        Logger().log_debug('Mel scale filterbank applied')

    def save_spectrogram(self, name):
        torch.save(self.spectrogram, f'C:/Users/MartinFaehnrich/Documents/ChiRO/data/Spectrograms/spectrogram_{name}.pt')
        Logger().log_debug('Spectrogram saved')

    def plot_spectrogram(self):
        magnitude = torch.abs(self.spectrogram)
        log_magnitude = torch.log1p(magnitude)  # Use log1p for numerical stability
        plt.figure()
        plt.imshow(log_magnitude.squeeze(0).numpy(), aspect='auto', origin='lower')
        plt.colorbar()
        plt.title('Spectrogram')
        plt.show()


if __name__ == '__main__':
    dt = AudioLoader('C:/Users/MartinFaehnrich/Documents/ChiRO/data/ExampleData')
    dt.load_folder()
    print(dt.get_data().shape)
    spectrogram = Spectrogram(dt.get_data()[0])
    spectrogram.melscale_filterbank()
    #spectrogram.save_spectrogram("1")
    spectrogram.plot_spectrogram()

    spectrogram = Spectrogram(dt.get_data()[1])
    spectrogram.melscale_filterbank()
    #spectrogram.save_spectrogram("2")
    #spectrogram.plot_spectrogram()
