import os
import time

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.DataSetSplit.DatasetSplitter import DatasetSplitter
from src.DataSetSplit.TrainingClasses import eptesicus_species, myotis_species, nyctalus_species, pipistrellus_species, \
    Chiroptera_generally, bat_species_fixed
from src.Preprocessing.AudioLoader import AudioLoader
from src.Preprocessing.BatFileDataSet import BatFileDataSet
from src.Preprocessing.SpectrogramProcessor import SpectrogramProcessor
from src.Training.TrainingParams import SEED, USE_MIN_FILES_PER_CLASS, SPLIT_METHOD, SPLIT_RATIOS, TOTAL_FILES_PER_CLASS


class Preprocessor:
    def __init__(self, files_and_labels_path, spectrograms_path, root_files_path):
        self.files_and_labels_path = files_and_labels_path
        self.spectrograms_path = spectrograms_path
        self.root_files_path = root_files_path

    def create_data_splits(self, original_labels_path):
        splitter = DatasetSplitter(
            excel_path=original_labels_path,
            root_path=self.root_files_path,
            seed=SEED,
            class_sample_limit=TOTAL_FILES_PER_CLASS,
            use_min_class_count=USE_MIN_FILES_PER_CLASS,
            balance_by_location=SPLIT_METHOD,
            split_ratios=SPLIT_RATIOS,
        )

        splitter.load_data()
        splitter.merge_labels([bat_species_fixed])
        num_classes = splitter.create_splits()
        splitter.export_splits_to_excel(os.path.dirname(self.files_and_labels_path))

        return num_classes

    def create_spectrograms_stft(self, highpass_cutoff_freq=16000, n_fft=4096, hop_length=None, win_length=2048, denois_option="mean_subtraction"):
        # Load Audio Files and create spectrograms
        audio_loader = AudioLoader()
        audio_loader.load_audio_from_exel(self.files_and_labels_path)
        waveforms = audio_loader.get_data()
        names = audio_loader.get_file_names_from_excel(self.files_and_labels_path)

        # Create Spectrograms
        for i in tqdm(range(len(waveforms)), desc="Creating Spectrograms", unit="file"):
            sp = SpectrogramProcessor(waveforms[i])
            sp.apply_highpass_filter(highpass_cutoff_freq)
            sp.compute_spectrogram(n_fft, hop_length, win_length)

            if denois_option == "mean_subtraction":
                sp.denoise_spectrogram_mean_subtraction()
            elif denois_option == "medain_filter":
                sp.denoise_spectrogram_median_filter()

            sp.scale_to_db()
            sp.save_spectrogram(f'{names[i]}', self.spectrograms_path + '/')

    def create_bat_file_dataset(self):
        return BatFileDataSet(self.spectrograms_path, self.files_and_labels_path, "Filename", "label")

    def create_bat_call_dataset(self):
        return BatFileDataSet(self.spectrograms_path, self.files_and_labels_path, "Filename", "label")
