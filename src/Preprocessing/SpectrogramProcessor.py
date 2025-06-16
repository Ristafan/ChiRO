import os

import numpy as np
import torchaudio
from scipy.ndimage import median_filter
from torchaudio.transforms import AmplitudeToDB
import numpy as np

from src.Logging.Logger import Logger
import torch
import torchaudio.transforms as T
import torchaudio.functional as F
import matplotlib.pyplot as plt
from src.Preprocessing.AudioLoader import AudioLoader
from tqdm import tqdm


class SpectrogramProcessor:
    def __init__(self, waveform, sample_rate=192000):
        """
        Initialize with a waveform tensor.
        :param waveform: Tensor of shape (channels, time)
        :param sample_rate: Sample rate of the waveform (default: 96kHz)
        """
        self.waveform = waveform
        self.sample_rate = sample_rate
        self.spectrogram = None

    def apply_highpass_filter(self, cutoff_freq=16000):
        """
        Applies a high-pass filter to remove frequencies below 16 kHz.
        Useful for isolating bat echolocation calls.
        """
        self.waveform = F.highpass_biquad(self.waveform, sample_rate=self.sample_rate, cutoff_freq=cutoff_freq)

    def compute_spectrogram(self, n_fft=4096, hop_length=None, win_length=2048):
        """
        Computes a high-resolution spectrogram optimized for bat echolocation calls.
        Reduced default n_fft and win_length.
        """
        transform = T.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=2.0)
        self.spectrogram = transform(self.waveform)
        return self.spectrogram

    def compute_mel_spectrogram(self, n_mels=256, n_fft=1024, hop_length=256, win_length=1024):
        """
        Computes a Mel spectrogram optimized for high-frequency bat calls.
        Uses a high number of Mel bins to capture fine details.
        Reduced default n_mels, n_fft, and win_length.
        """
        transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            win_length=win_length,
            mel_scale="htk",
            f_min=0,
            f_max=self.sample_rate // 2  # Use Nyquist frequency (48 kHz for 96 kHz recordings)
        )

        spectrogram_gpu = transform(self.waveform)
        self.spectrogram = spectrogram_gpu.cpu()
        return self.spectrogram

    def scale_to_db(self, top_db=80.0):
        transform = AmplitudeToDB(stype="power", top_db=top_db)
        self.spectrogram = transform(self.spectrogram.unsqueeze(0)).squeeze(0)

    def denoise_spectrogram_mean_subtraction(self):
        """
        Removes noise by subtracting the mean amplitude from each frequency bin.
        Assumes spectrogram is already computed and is on the CPU.
        """
        if self.spectrogram is None:
            raise ValueError("Spectrogram has not been computed yet.")

        # Compute the mean amplitude per frequency bin and subtract it
        mean_per_freq = self.spectrogram.mean(dim=-1, keepdim=True)
        self.spectrogram = self.spectrogram - mean_per_freq
        return self.spectrogram

    def denoise_spectrogram_median_filter(self, kernel_size=(3, 3)):
        """
        Applies a 2D median filter to the spectrogram. This is an image processing
        technique effective at removing impulsive noise ('salt-and-pepper' like noise)
        and smoothing out isolated bright or dark speckles on the spectrogram,
        while generally preserving edges (like the distinct contours of bat calls).
        Correctly handles both 2D and 3D (batch) spectrograms by iterating over the batch dimension.

        Note: This method is significantly more efficient with NumPy arrays due to
        `scipy.ndimage.median_filter`. If your spectrogram is a PyTorch tensor,
        it will be temporarily converted to NumPy for processing and then converted back.

        Args:
            kernel_size (tuple): A 2-element tuple (height, width) specifying the
                                 dimensions of the median filter window.
                                 It's highly recommended that kernel dimensions are odd (e.g., (3,3), (5,5)).
        """
        if self.spectrogram is None:
            raise ValueError("Spectrogram has not been computed yet.")
        if len(kernel_size) != 2:
            raise ValueError("kernel_size must be a 2-element tuple (height, width).")
        if not all(k % 2 == 1 for k in kernel_size):
            print(f"Warning: It is recommended to use odd kernel sizes for median filter. Current: {kernel_size}")

        if isinstance(self.spectrogram, torch.Tensor):
            print("Converting PyTorch tensor to NumPy array for median filtering (scipy.ndimage.median_filter).")
            # Move tensor to CPU and convert to NumPy array
            np_spectrogram = self.spectrogram.cpu().numpy()

            if np_spectrogram.ndim == 3:
                # If 3D (batch, freq, time), apply filter to each 2D slice
                denoised_slices = []
                for i in range(np_spectrogram.shape[0]):
                    denoised_slices.append(median_filter(np_spectrogram[i], size=kernel_size))
                denoised_np_spectrogram = np.stack(denoised_slices, axis=0)
            elif np_spectrogram.ndim == 2:
                # If 2D (freq, time), apply filter directly
                denoised_np_spectrogram = median_filter(np_spectrogram, size=kernel_size)
            else:
                raise ValueError("Spectrogram must be 2D or 3D to apply median filter.")

            # Convert back to PyTorch tensor and move to original device
            self.spectrogram = torch.from_numpy(denoised_np_spectrogram).to(self.spectrogram.device)

        elif isinstance(self.spectrogram, np.ndarray):
            if self.spectrogram.ndim == 3:
                # If 3D (batch, freq, time), apply filter to each 2D slice
                denoised_slices = []
                for i in range(self.spectrogram.shape[0]):
                    denoised_slices.append(median_filter(self.spectrogram[i], size=kernel_size))
                self.spectrogram = np.stack(denoised_slices, axis=0)
            elif self.spectrogram.ndim == 2:
                # If 2D (freq, time), apply filter directly
                self.spectrogram = median_filter(self.spectrogram, size=kernel_size)
            else:
                raise ValueError("Spectrogram must be 2D or 3D to apply median filter.")
        else:
            raise TypeError("Unsupported spectrogram type. Must be numpy.ndarray or torch.Tensor.")

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
        spec_cpu = self.spectrogram.to('cpu').squeeze()
        spec_to_plot = spec_cpu.numpy()

        if log_scale:
            spec_to_plot = 10 * torch.log10(spec_cpu + 1e-10).numpy()  # Avoid log(0) errors

        plt.figure(figsize=(10, 5))
        plt.imshow(spec_to_plot, aspect='auto', origin='lower', cmap='magma', extent=[0, spec_to_plot.shape[1], 16, 48])
        plt.colorbar(label="Power (dB)" if log_scale else "Amplitude")
        plt.xlabel("Time Frames")
        plt.ylabel("Frequency (kHz)")
        plt.title("Spectrogram (Log Scale)" if log_scale else "Spectrogram")
        plt.show()

    def plot_new(self):
        spectrogram_db = T.AmplitudeToDB(stype="power", top_db=80.0)(self.spectrogram) # Apply dynamic range

        spectrogram_db = spectrogram_db.squeeze(0)  # Remove the channel dimension
        spectrogram_db = spectrogram_db.cpu()  # Move to CPU for plotting

        plt.figure(figsize=(16, 10))
        plt.imshow(spectrogram_db.numpy(), aspect='auto', origin='lower',
                   extent=[0, self.waveform.shape[-1] / self.sample_rate, 0, self.sample_rate / 2])
        plt.colorbar(format='%+2.0f dB', label='Power/Amplitude (dB)')
        plt.title('Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.ylim(0, 96000) # Set the frequency limit as per the settings
        plt.show()


if __name__ == '__main__':
    waveform, sample_rate = AudioLoader().load_wav_file('C:/Users/MartinFaehnrich/Documents/ChiRO/data/ExampleData/train/20220630_221300T #0002_645fbdba0181f0367e9570e949180e4b.wav')

    s = SpectrogramProcessor(waveform)
    s.apply_highpass_filter()
    s.compute_spectrogram()
    s.plot_new()
    # s.compute_mel_spectrogram()
    print(s.spectrogram.shape)
    s.denoise_spectrogram_mean_subtraction()
    # s.denoise_spectrogram_median_filter(kernel_size=(5, 5))
    s.scale_to_db()
    s.plot_new()

#    for i in tqdm(range(len(waveforms)), desc="Creating Spectrograms"):
#        sp = SpectrogramProcessor(waveforms[i])
#        sp.apply_highpass_filter()
#        sp.compute_spectrogram()
#        # sp.compute_mel_spectrogram()
#        sp.denoise_spectrogram()
#        # sp.save_spectrogram(f'{names[i]}', 'C:/Users/MartinFaehnrich/Documents/ChiRO/data/Spectrograms/')
#        sp.plot_spectrogram(False)
