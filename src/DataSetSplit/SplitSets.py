import pandas as pd
import os
import random as rd
import shutil as sh
from collections import defaultdict
from tqdm import tqdm

from TrainingClasses import bat_species, eptesicus_species, myotis_species, nyctalus_species, pipistrellus_species, Chiroptera_generally


class SplitSet:
    def __init__(self, data_source_path, labels_path, data_target_path):
        self.data_source_path = data_source_path
        self.data_target_path = data_target_path
        self.labels_path = labels_path

        self.filenames = []
        self.class_labels = {}
        self.criteria_labels = {}
        self.original_class_labels = {}
        self.original_criteria_labels = {}
        self.combined_labels_criteria = defaultdict(lambda: defaultdict(list))

        self.split_method = "balanced"  # Default to balanced
        self.split_ratio = [0.8, 0.2, 0]
        self.split_seed = None

        self.train_set = []
        self.val_set = []
        self.test_set = []

    def select_split_method(self, split_method):
        # Methods include "random", "balanced"
        if split_method not in ["random", "balanced"]:
            raise ValueError("Invalid split method. Choose from 'random' or 'balanced'.")
        self.split_method = split_method

    def select_split_ratio(self, split_train, split_val, split_test):
        assert sum([split_train, split_val, split_test]) == 1, "Split ratios must sum to 1"
        self.split_ratio = [split_train, split_val, split_test]

    def select_split_seed(self, split_seed):
        if isinstance(split_seed, int) and 0 <= split_seed <= 1000:
            self.split_seed = split_seed
        else:
            raise ValueError("Split seed must be an integer between 0 and 1000.")

    def read_data(self, filename_column, label_column, criterion_column):
        data = pd.read_excel(self.labels_path)
        self.filenames = data[filename_column].tolist()
        self.class_labels = {row[filename_column]: row[label_column] for _, row in data.iterrows()}
        self.criteria_labels = {row[filename_column]: row[criterion_column] for _, row in data.iterrows()}
        self.original_class_labels = self.class_labels.copy()
        self.original_criteria_labels = self.criteria_labels.copy()

    def merge_class_labels(self, to_merge, new_label):
        for filename, label in self.class_labels.items():
            if label in to_merge:
                self.class_labels[filename] = new_label

    def merge_criteria_labels(self, to_merge, new_label):
        for filename, label in self.criteria_labels.items():
            if label in to_merge:
                self.criteria_labels[filename] = new_label

    def _prepare_data_for_splitting(self):
        """Groups filenames by their class and criterion."""
        self.combined_labels_criteria.clear()
        for filename in self.filenames:
            class_label = self.class_labels[filename]
            criterion_label = self.criteria_labels[filename]
            self.combined_labels_criteria[class_label][criterion_label].append(filename)

    def _split_data_by_ratio(self, data_list, ratios, shuffle=True):
        """Splits a list into parts based on the given ratios."""
        if shuffle:
            rd.shuffle(data_list)
        n = len(data_list)
        train_idx = int(n * ratios[0])
        val_idx = train_idx + int(n * ratios[1])
        train_set = data_list[:train_idx]
        val_set = data_list[train_idx:val_idx]
        test_set = data_list[val_idx:]
        return train_set, val_set, test_set

    def create_splits(self, total_files_per_class=None, merge_labels=None, merge_criteria=None):
        if self.split_seed is not None:
            rd.seed(self.split_seed)

        if merge_labels:
            for merge_group in merge_labels:
                new_label = merge_group[0].split(" ")[0] if isinstance(merge_group, list) and merge_group else f"merged_class_{rd.randint(0, 100)}"
                self.merge_class_labels(merge_group, new_label)

        if merge_criteria:
            for merge_group in merge_criteria:
                new_label = merge_group[0].split(" ")[0] if isinstance(merge_group, list) and merge_group else f"merged_criterion_{rd.randint(0, 100)}"
                self.merge_criteria_labels(merge_group, new_label)

        self._prepare_data_for_splitting()

        all_train_files = []
        all_val_files = []
        all_test_files = []

        for class_label, criteria_dict in self.combined_labels_criteria.items():
            class_files = []
            for criterion, files in criteria_dict.items():
                class_files.extend(files)

            if not class_files:
                print(f"Warning: No files found for class '{class_label}'.")
                continue

            if total_files_per_class is not None:
                num_files_to_select = min(total_files_per_class, len(class_files))
                if len(class_files) > num_files_to_select:
                    class_files = rd.sample(class_files, num_files_to_select)

            train_files, val_files, test_files = self._split_data_by_ratio(class_files, self.split_ratio)

            all_train_files.extend(train_files)
            all_val_files.extend(val_files)
            all_test_files.extend(test_files)

        self.train_set = all_train_files
        self.val_set = all_val_files
        self.test_set = all_test_files

        print(f"\nFinal set sizes - Train: {len(self.train_set)}, Val: {len(self.val_set)}, Test: {len(self.test_set)}")

    def export_to_excel(self, output_dir):
        """
        Export dataset information to three separate Excel files (train, val, test),
        each with relevant columns.
        """
        os.makedirs(output_dir, exist_ok=True)

        def _create_dataframe(file_list, set_name):
            data = []
            for filename in file_list:
                data.append({
                    'Filename': filename,
                    'Filepath': os.path.join(self.data_source_path, self.original_class_labels.get(filename, 'unknown'), f'{filename}.WAV'),
                    'Criterion': self.criteria_labels.get(filename, 'unknown'),
                    'Class': self.class_labels.get(filename, 'unknown'),
                    'original_class': self.original_class_labels.get(filename, 'unknown'),
                })
            df = pd.DataFrame(data)
            filepath = os.path.join(output_dir, f"{set_name}_dataset_info.xlsx")
            df.to_excel(filepath, index=False)
            print(f"{set_name.capitalize()} set information exported to {filepath}")
            return df

        train_df = _create_dataframe(self.train_set, "train")
        val_df = _create_dataframe(self.val_set, "val")
        test_df = _create_dataframe(self.test_set, "test")


if __name__ == "__main__":
    # Example usage
    data_source_path = "D:/Bachelorarbeit/AgroscopeData/LabelledSequences"
    labels_path = "D:/Bachelorarbeit/AgroscopeData/LabelledSequencesMerged.xlsx"
    data_target_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/DataAlphaBeta"

    split_set = SplitSet(data_source_path, labels_path, data_target_path)
    split_set.read_data("File", "Verification 1", "location")
    split_set.select_split_method("balanced")
    split_set.select_split_ratio(0.7, 0.15, 0.15)
    split_set.select_split_seed(42)  # For reproducibility
    split_set.create_splits(total_files_per_class=15000, merge_labels=[eptesicus_species, myotis_species, nyctalus_species, pipistrellus_species, Chiroptera_generally])
    split_set.export_to_excel(os.path.join(data_target_path, "dataset_info"))