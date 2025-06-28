import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import get_window, stft

# Load audio
file_path = "D:/Bachelorarbeit/AgroscopeData/LabelledSequences/Eptesicus_nilssonii/20220612_223600T__0005_2d01bd6bc485c0150083eb3a754ee189.WAV"

# PARAMETERS
SAMPLE_RATE = 192000
N_FFT = 4096
WIN_LENGTH = 2048
HOP_LENGTH = WIN_LENGTH // 2
start_time_ms = 3550  # in milliseconds
duration_ms = 100     # segment duration in milliseconds

# Load audio file
fs, signal = wavfile.read(file_path)

# Convert to mono if stereo
if signal.ndim > 1:
    signal = signal.mean(axis=1)

# Resample if needed
if fs != SAMPLE_RATE:
    raise ValueError(f"Expected sample rate {SAMPLE_RATE}, but got {fs}")

# Time vector in ms
time_ms = np.arange(len(signal)) / fs * 1000

# Segment indices
start_idx = int(start_time_ms / 1000 * fs)
end_idx = int((start_time_ms + duration_ms) / 1000 * fs)
segment = signal[start_idx:end_idx]
segment_time_ms = np.arange(len(segment)) / fs * 1000 + start_time_ms

# === Plot 1: Full waveform with bars ===
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(time_ms, signal, color='gray')
plt.axvline(start_time_ms, color='red', linestyle='--')
plt.axvline(start_time_ms + duration_ms, color='red', linestyle='--')
plt.title('Full Waveform')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')

# === Plot 2: Zoomed segment ===
plt.subplot(4, 1, 2)
plt.plot(segment_time_ms, segment, color='blue')
plt.title('Segmented Waveform')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')

# === Plot 3: FFT of segment ===
window = get_window('hann', WIN_LENGTH)
f, t, Zxx = stft(segment, fs=fs, nperseg=WIN_LENGTH, noverlap=WIN_LENGTH - HOP_LENGTH, window=window, nfft=N_FFT)
power = 10 * np.log10(2 * np.abs(Zxx)**2 + 1e-12)  # subplot 3

plt.subplot(4, 1, 3)
quad = plt.pcolormesh(t * 1000, f, power, shading='gouraud')
quad.set_rasterized(True)
plt.title('FFT of Segment (Hann window)')
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Power')
plt.ylim(0, fs / 2)

# === Plot 4: Full spectrogram with only segment filled ===
f_full, t_full, Zxx_full = stft(signal, fs=fs, nperseg=WIN_LENGTH, noverlap=WIN_LENGTH - HOP_LENGTH, window=window, nfft=N_FFT)
power_full = 10 * np.log10(2 * np.abs(Zxx_full)**2 + 1e-12)  # subplot 4
t_ms_full = t_full * 1000

# Masking out the full spectrogram except the segment
#mask = (t_ms_full >= start_time_ms) & (t_ms_full <= start_time_ms + duration_ms)
#masked_power = np.zeros_like(power_full)
#masked_power[:, mask] = power_full[:, mask]

plt.subplot(4, 1, 4)
quad = plt.pcolormesh(t_ms_full, f_full, power_full, shading='gouraud')
quad.set_rasterized(True)
plt.title('Spectrogram Highlighting Segment Only')
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Power')
plt.axvline(start_time_ms, color='red', linestyle='--', label='Segment Start')
plt.axvline(start_time_ms + duration_ms, color='red', linestyle='--', label='Segment End')
plt.ylim(0, fs / 2)

plt.tight_layout()
#plt.draw()
plt.savefig("STFT_Comparison.pdf", dpi=400)
#plt.show()
