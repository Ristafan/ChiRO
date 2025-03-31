from src.Logging.Logger import Logger
import torch
import torchaudio.transforms as T
import torchaudio.functional as F
import matplotlib.pyplot as plt
from src.Preprocessing.AudioLoader import AudioLoader
from tqdm import tqdm


class SpectrogramProcessor:
    def __init__(self, waveform, sample_rate=96000):
        """
        Initialize with a waveform tensor.
        :param waveform: Tensor of shape (channels, time)
        :param sample_rate: Sample rate of the waveform (default: 96kHz)
        """
        self.waveform = waveform
        self.sample_rate = sample_rate
        self.spectrogram = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def apply_highpass_filter(self, cutoff_freq=16000):
        """
        Applies a high-pass filter to remove frequencies below 16 kHz.
        Useful for isolating bat echolocation calls.
        """
        self.waveform = F.highpass_biquad(self.waveform, sample_rate=self.sample_rate, cutoff_freq=cutoff_freq)

    def compute_spectrogram(self, n_fft=4096, hop_length=256, win_length=4096):
        """
        Computes a high-resolution spectrogram optimized for bat echolocation calls.
        """
        transform = T.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=2.0).to(self.device)
        self.waveform = self.waveform.to(self.device)
        self.spectrogram = transform(self.waveform)
        return self.spectrogram.to('cpu')

    def compute_mel_spectrogram(self, n_mels=256, n_fft=4096, hop_length=256):
        """
        Computes a Mel spectrogram optimized for high-frequency bat calls.
        Uses a high number of Mel bins to capture fine details.
        """
        transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=1000,  # Ignore low frequencies (<1 kHz) as bat calls are usually higher
            f_max=self.sample_rate // 2  # Use Nyquist frequency (48 kHz for 96 kHz recordings)
        ).to(self.device)

        self.waveform = self.waveform.to(self.device)
        self.spectrogram = transform(self.waveform)
        return self.spectrogram.to('cpu')

    def denoise_spectrogram(self):
        """
        Removes noise by subtracting the mean amplitude from each frequency bin.
        Assumes spectrogram is already computed.
        """
        if self.spectrogram is None:
            raise ValueError("Spectrogram has not been computed yet.")

        # Compute the mean amplitude per frequency bin and subtract it
        mean_per_freq = self.spectrogram.mean(dim=-1, keepdim=True)
        self.spectrogram = self.spectrogram - mean_per_freq
        return self.spectrogram

    def save_spectrogram(self, name, save_path="C:/Users/MartinFaehnrich/Documents/ChiRO/data/Spectrograms/"):
        """Saves the computed spectrogram as a .pt file."""
        if self.spectrogram is None:
            raise ValueError("Spectrogram has not been computed yet.")
        torch.save(self.spectrogram, f"{save_path}spectrogram_{name}.pt")

    @staticmethod
    def load_spectrogram(filepath):
        """Loads a spectrogram from a .pt file."""
        return torch.load(filepath)

    def plot_spectrogram(self, log_scale=True):
        """Plots the spectrogram."""
        if self.spectrogram is None:
            raise ValueError("Spectrogram has not been computed yet.")

        # Convert to decibels if using log scale
        spec_to_plot = self.spectrogram.numpy()
        if log_scale:
            spec_to_plot = 10 * torch.log10(self.spectrogram + 1e-10).numpy()  # Avoid log(0) errors

        plt.figure(figsize=(10, 5))
        plt.imshow(spec_to_plot, aspect='auto', origin='lower', cmap='magma', extent=[0, spec_to_plot.shape[1], 16, 48])
        plt.colorbar(label="Power (dB)" if log_scale else "Amplitude")
        plt.xlabel("Time Frames")
        plt.ylabel("Frequency (kHz)")
        plt.title("Mel Spectrogram (Log Scale)" if log_scale else "Mel Spectrogram")
        plt.show()


if __name__ == '__main__':
    dt = AudioLoader('C:/Users/MartinFaehnrich/Documents/ChiRO/data/ExampleData')
    dt.load_folder()
    waveforms = dt.get_data()
    names = dt.get_file_names()

    for i in tqdm(range(len(waveforms)), desc="Creating Spectrograms"):
        sp = SpectrogramProcessor(waveforms[i])
        sp.apply_highpass_filter()
        # sp.compute_spectrogram()
        sp.compute_mel_spectrogram()
        sp.denoise_spectrogram()
        # sp.save_spectrogram(f'{names[i]}', 'C:/Users/MartinFaehnrich/Documents/ChiRO/data/Spectrograms/')
        sp.plot_spectrogram(False)
