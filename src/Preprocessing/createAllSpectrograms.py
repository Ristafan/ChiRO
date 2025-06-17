import glob

from tqdm import tqdm

from src.Preprocessing.AudioLoader import AudioLoader
from src.Preprocessing.SpectrogramProcessor import SpectrogramProcessor
from src.Training.TrainingParams import HIGHPASS_CUTOFF_FREQ, N_FFT, HOP_LENGTH, WIN_LENGTH

if __name__ == "__main__":
    folders = glob.glob("D:/Bachelorarbeit/AgroscopeData/LabelledSequences/*")
    # normalize paths to Windows format
    folders = [folder.replace("/", "\\") for folder in folders]

    for folder in folders:
        print(f"Processing folder: {folder}")
        audio_loader = AudioLoader()
        audio_loader.load_folder(folder)
        waveforms = audio_loader.get_data()
        names = audio_loader.get_file_names_from_folder(folder)

        # Create Spectrograms
        for i in tqdm(range(len(waveforms)), desc="Creating Spectrograms", unit="file"):
            sp = SpectrogramProcessor(waveforms[i])
            sp.apply_highpass_filter(HIGHPASS_CUTOFF_FREQ)
            sp.compute_spectrogram(N_FFT, HOP_LENGTH, WIN_LENGTH)
            sp.denoise_spectrogram_mean_subtraction()
            sp.scale_to_db()
            sp.save_spectrogram(f'{names[i]}', "D:/Bachelorarbeit/AgroscopeData/spectrograms/")

        # Clear the data to free memory
        audio_loader.data.clear()
        print(f"Finished processing folder: {folder}")
