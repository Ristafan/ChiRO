import pandas as pd
import os
import random as rd
from collections import defaultdict

from src.DataSetSplit.TrainingClasses import bat_species, eptesicus_species, myotis_species, nyctalus_species, pipistrellus_species, Chiroptera_generally


class SplitSet:
    def __init__(self, data_source_path, labels_path):
        self.data_source_path = data_source_path
        self.labels_path = labels_path

        self.filenames = []
        self.class_labels = {}
        self.criteria_labels = {}
        self.original_class_labels = {}
        self.original_criteria_labels = {}
        self.combined_labels_criteria = defaultdict(lambda: defaultdict(list))
        self.class_to_numeric_label = {}
        self.ignored_labels = set()

        self.split_method = "balanced"  # Default to balanced
        self.split_ratio = [0.8, 0.2, 0]
        self.split_seed = None

        self.train_set = []
        self.val_set = []
        self.test_set = []

    def set_ignored_labels(self, labels_to_ignore):
        """Sets a list of labels to be ignored during data loading."""
        if isinstance(labels_to_ignore, list):
            self.ignored_labels.update(labels_to_ignore)
        else:
            print("Warning: Input for set_ignored_labels should be a list.")

    def select_split_method(self, split_method):
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
        temp_filenames = []
        temp_class_labels = {}
        temp_criteria_labels = {}

        for _, row in data.iterrows():
            filename = row[filename_column]
            label = row[label_column]
            criterion = row[criterion_column]
            if label not in self.ignored_labels:
                temp_filenames.append(filename)
                temp_class_labels[filename] = label
                temp_criteria_labels[filename] = criterion

        self.filenames = temp_filenames
        self.class_labels = temp_class_labels
        self.criteria_labels = temp_criteria_labels
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
        self.combined_labels_criteria.clear()
        for filename in self.filenames:
            class_label = self.class_labels[filename]
            criterion_label = self.criteria_labels[filename]
            self.combined_labels_criteria[class_label][criterion_label].append(filename)

        # Create numeric labels for each class
        unique_classes = sorted(self.combined_labels_criteria.keys())
        self.class_to_numeric_label = {label: i for i, label in enumerate(unique_classes)}

    def _split_data_by_ratio(self, data_list, ratios, shuffle=True):
        if shuffle:
            rd.shuffle(data_list)
        n = len(data_list)
        train_idx = int(n * ratios[0])
        val_idx = train_idx + int(n * ratios[1])
        train_set = data_list[:train_idx]
        val_set = data_list[train_idx:val_idx]
        test_set = data_list[val_idx:]
        return train_set, val_set, test_set

    def create_splits(self, total_files_per_class=None, use_min_files_per_class=False, merge_labels=None, merge_criteria=None):
        if self.split_seed is not None:
            rd.seed(self.split_seed)

        print(merge_labels)
        if merge_labels:
            for merge_group in merge_labels:
                new_label = merge_group[0].split("_")[0] if isinstance(merge_group, list) and merge_group else f"merged_class_{rd.randint(0, 100)}"
                print(f"Merging labels: {merge_group} into new label: {new_label}")
                self.merge_class_labels(merge_group, new_label)

        if merge_criteria:
            for merge_group in merge_criteria:
                new_label = merge_group[0].split(" ")[0] if isinstance(merge_group, list) and merge_group else f"merged_criterion_{rd.randint(0, 100)}"
                self.merge_criteria_labels(merge_group, new_label)

        self._prepare_data_for_splitting()

        class_file_counts = {
            class_label: sum(len(files) for files in criteria_dict.values())
            for class_label, criteria_dict in self.combined_labels_criteria.items()
        }

        target_files_per_class = total_files_per_class
        if use_min_files_per_class:
            min_count = min(class_file_counts.values()) if class_file_counts else 0
            target_files_per_class = min_count
            print(f"Using a maximum of {target_files_per_class} files per class based on the smallest class.")
        elif total_files_per_class is not None:
            for class_label, count in class_file_counts.items():
                if count < total_files_per_class:
                    print(f"Warning: Class '{class_label}' has only {count} files, which is less than the target of {total_files_per_class}.")

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

            num_files_to_select = len(class_files)
            if target_files_per_class is not None:
                num_files_to_select = min(target_files_per_class, len(class_files))

            if num_files_to_select < len(class_files):
                class_files = rd.sample(class_files, num_files_to_select)

            train_files, val_files, test_files = self._split_data_by_ratio(class_files, self.split_ratio)

            all_train_files.extend(train_files)
            all_val_files.extend(val_files)
            all_test_files.extend(test_files)

        self.train_set = all_train_files
        self.val_set = all_val_files
        self.test_set = all_test_files

        print(f"\nFinal set sizes - Train: {len(self.train_set)}, Val: {len(self.val_set)}, Test: {len(self.test_set)}")

        # Return label of "Env sounds" if present
        return self.class_to_numeric_label.get("Env_sounds", -1)  # -1 for unknown

    def export_to_excel(self, output_dir):
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
                    'label': self.class_to_numeric_label.get(self.class_labels.get(filename, 'unknown'), -1) # -1 for unknown
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
    data_target_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/DataBeta"

    # Server paths
    #data_source_path = "/local/scratch/faehnrich/AgroscopeData/LabelledSequences"
    #labels_path = "/local/scratch/faehnrich/AgroscopeData/LabelledSequencesMerged.xlsx"
    #data_target_path = "/local/scratch/faehnrich/AgroscopeData/Training/DataAlphaBeta"

    split_set = SplitSet(data_source_path, labels_path)
    split_set.set_ignored_labels(["Env sounds"])
    split_set.read_data("File", "Verification 1", "location")
    split_set.select_split_method("balanced")
    split_set.select_split_ratio(0.7, 0.15, 0.15)

    # Example 1: Using minimum files per class
    split_set.create_splits(use_min_files_per_class=True, merge_labels=[eptesicus_species, myotis_species, nyctalus_species, pipistrellus_species, Chiroptera_generally])
    split_set.export_to_excel(os.path.join(data_target_path, "dataset_info_min"))

    # Example 2: Setting a target number of files per class
    # split_set_target = SplitSet(data_source_path, labels_path, data_target_path)
    # split_set_target.read_data("File", "Verification 1", "location")
    # split_set_target.select_split_method("balanced")
    # split_set_target.select_split_ratio(0.7, 0.15, 0.15)
    # split_set_target.select_split_seed(42)  # For reproducibility
    # split_set_target.create_splits(total_files_per_class=1000, merge_labels=[eptesicus_species, myotis_species, nyctalus_species, pipistrellus_species, Chiroptera_generally])
    # split_set_target.export_to_excel(os.path.join(data_target_path, "dataset_info_target"))