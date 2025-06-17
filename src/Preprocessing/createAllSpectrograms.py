import glob

from tqdm import tqdm

from src.Preprocessing.AudioLoader import AudioLoader
from src.Preprocessing.SpectrogramProcessor import SpectrogramProcessor
from src.Training.TrainingParams import HIGHPASS_CUTOFF_FREQ, N_FFT, HOP_LENGTH, WIN_LENGTH

if __name__ == "__main__":
    folders = glob.glob("/cluster/raid/home/f60047174/data/LabelledSequences")

    for folder in folders:
        print(f"Processing folder: {folder}")
        audio_loader = AudioLoader()
        waveforms = audio_loader.get_data()
        names = audio_loader.get_file_names_from_folder(folder)

        # Create Spectrograms
        for i in tqdm(range(len(waveforms)), desc="Creating Spectrograms", unit="file"):
            sp = SpectrogramProcessor(waveforms[i])
            sp.apply_highpass_filter(HIGHPASS_CUTOFF_FREQ)
            sp.compute_spectrogram(N_FFT, HOP_LENGTH, WIN_LENGTH)
            sp.denoise_spectrogram_mean_subtraction()
            sp.scale_to_db()
            sp.save_spectrogram(f'{names[i]}', "/cluster/raid/home/f60047174/data/spectrograms/")
