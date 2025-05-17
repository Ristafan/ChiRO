import os

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.DataSetSplit.SplitSets import SplitSet
from src.DataSetSplit.TrainingClasses import eptesicus_species, myotis_species, nyctalus_species, pipistrellus_species, \
    Chiroptera_generally
from src.Preprocessing.AudioLoader import AudioLoader
from src.Preprocessing.BatCallDataset import BatCallDataset
from src.Preprocessing.SpectrogramProcessor import SpectrogramProcessor


class Preprocessor:
    def __init__(self, files_and_labels_path, spectrograms_path, root_files_path):
        self.files_and_labels_path = files_and_labels_path
        self.spectrograms_path = spectrograms_path
        self.root_files_path = root_files_path

    def create_data_splits(self, original_labels_path, use_min_files_per_class=False, total_files_per_class=100, ignored_labels=None, merge_labels=None, split_method="balanced", train_ratio=0.7, test_ratio=0.2, seed=42):
        assert ignored_labels is None or isinstance(ignored_labels, list), "ignored_labels must be a list or None"
        assert merge_labels is None or isinstance(merge_labels, list), "merge_labels must be a list or None"
        assert split_method in ["balanced", "targeted"], "split_method must be either 'balanced' or 'targeted'"
        assert isinstance(train_ratio, float) and isinstance(test_ratio, float), "train_ratio and test_ratio must be floats"

        split_set = SplitSet(self.root_files_path, original_labels_path)
        split_set.select_split_seed(seed)
        split_set.set_ignored_labels(ignored_labels)
        split_set.read_data("File", "Verification 1", "location")
        split_set.select_split_method(split_method)
        split_set.select_split_ratio(train_ratio, test_ratio, 1 - train_ratio - test_ratio)

        if use_min_files_per_class:
            # Using minimum files per class
            split_set.create_splits(use_min_files_per_class=use_min_files_per_class, merge_labels=merge_labels)
        else:
            # Using total files per class
            split_set.create_splits(total_files_per_class=total_files_per_class, merge_labels=merge_labels)

        # Save the splits to Excel files
        split_set.export_to_excel(os.path.dirname(self.files_and_labels_path))

    def create_spectrograms(self):
        # Load Audio Files and create spectrograms
        audio_loader = AudioLoader()
        audio_loader.load_audio_from_exel(self.files_and_labels_path)
        waveforms = audio_loader.get_data()
        names = audio_loader.get_file_names_from_excel(self.files_and_labels_path)

        # Create Spectrograms
        for i in tqdm(range(len(waveforms)), desc="Creating Spectrograms"):
            sp = SpectrogramProcessor(waveforms[i])
            sp.apply_highpass_filter()
            sp.compute_spectrogram()
            sp.denoise_spectrogram()
            sp.save_spectrogram(f'{names[i]}', self.spectrograms_path + '/')

    def create_bat_call_dataset(self):
        return BatCallDataset(self.spectrograms_path, self.files_and_labels_path, "Filename", "label")
