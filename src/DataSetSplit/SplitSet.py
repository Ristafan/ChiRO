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

        self.total_files = 0
        self.filenames = []
        self.classes = []
        self.criteria = []
        self.split_method = 0
        self.split_ratio = [0.8, 0.2, 0]
        self.split_seed = 0

        self.class_labels = {}
        self.criteria_labels = {}
        self.original_labels = {}
        self.combined_labels = {}

        self.num_files_per_class = {}
        self.num_files_per_criterion = {}

        self.train_set = []
        self.val_set = []
        self.test_set = []

    def select_split_method(self, split_method):
        # Methods include "random", "balanced", "all"
        self.split_method = split_method

    def select_split_ratio(self, split_train, split_val, split_test):
        assert split_train + split_val + split_test == 1, "Split ratios must sum to 1"
        self.split_ratio = [split_train, split_val, split_test]

    def select_split_seed(self):
        self.split_seed = rd.randint(0, 1000)

    def read_data(self, filename_column, label_column, criterion_column):
        data = pd.read_excel(self.labels_path)
        self.filenames = data[filename_column].tolist()
        self.class_labels = {row[filename_column]: row[label_column] for _, row in data.iterrows()}
        self.criteria_labels = {row[filename_column]: row[criterion_column] for _, row in data.iterrows()}

    def merge_class_labels(self, to_merge, merge_index=0):
        # Merge class labels based on the provided list
        for filename in self.filenames:
            if self.class_labels[filename] in to_merge:
                self.class_labels[filename] = f'new_class_{str(merge_index)}'

        # Update the TrainingClasses list
        self.classes = list(set(self.class_labels.values()))

    def merge_criteria_labels(self, to_merge, merge_index=0):
        # Merge criteria labels based on the provided list
        for filename in self.filenames:
            if self.criteria_labels[filename] in to_merge:
                self.criteria_labels[filename] = f'new_criterion_{str(merge_index)}'

        # Update the criteria list
        self.criteria = list(set(self.criteria_labels.values()))

    def get_number_of_distinct_criteria(self):
        self.criteria = []
        for criterion in self.criteria_labels.values():
            if criterion not in self.criteria:
                self.criteria.append(criterion)

    def combine_labels_criteria(self):
        # Create dictionary to store combined labels
        self.combined_labels = {}
        for cl in self.classes:
            self.combined_labels[cl] = {}
            for cr in self.criteria:
                self.combined_labels[cl][cr] = []

        # Populate combined labels
        for filename in self.filenames:
            class_label = self.class_labels[filename]
            criterion_label = self.criteria_labels[filename]
            self.combined_labels[class_label][criterion_label].append(filename)

    def _add_files_balanced(self, class_label, files_by_criterion, available_criteria, num_files, target_set):
        """Helper method to add files to a set while maintaining balance across criteria"""
        # If no files are needed, return immediately
        if num_files <= 0:
            return

        files_added = 0

        # First, try to distribute files evenly across criteria
        files_per_criterion = max(1, num_files // len(available_criteria)) if available_criteria else 0

        for cr in available_criteria:
            # Get the number of files to select from this criterion
            to_select = min(files_per_criterion, len(files_by_criterion[cr]))

            # Select random files from this criterion
            if to_select > 0:
                selected = rd.sample(files_by_criterion[cr], to_select)
                target_set.extend(selected)

                # Remove selected files from the pool
                for file in selected:
                    files_by_criterion[cr].remove(file)

                files_added += to_select

        # If we need more files, take them from any available criterion
        # TODO: Fix this part so that not all files are selected from the wholes set
        remaining = num_files - files_added
        if remaining > 0:
            # Update available criteria
            available_criteria = [cr for cr in self.criteria if len(files_by_criterion[cr]) > 0]

            while remaining > 0 and available_criteria:
                # Select a random criterion
                criterion = rd.choice(available_criteria)

                if len(files_by_criterion[criterion]) > 0:
                    # Select a random file from this criterion
                    selected_file = rd.choice(files_by_criterion[criterion])
                    target_set.append(selected_file)

                    # Remove selected file from the pool
                    files_by_criterion[criterion].remove(selected_file)
                    remaining -= 1

                # Update available criteria
                available_criteria = [cr for cr in self.criteria if len(files_by_criterion[cr]) > 0]

    def create_balanced_split(self):
        """
        Create a balanced split where each class has exactly the specified number of files,
        and files are evenly distributed across criteria when possible.
        """
        # Calculate how many files to select per class based on the split ratios
        train_files_per_class = int(self.total_files * self.split_ratio[0])
        val_files_per_class = int(self.total_files * self.split_ratio[1])
        test_files_per_class = self.total_files - train_files_per_class - val_files_per_class

        print(f"Files per class - Train: {train_files_per_class}, Val: {val_files_per_class}, Test: {test_files_per_class}")

        # For each class, select the specified number of files for each set
        for cl in self.classes:
            # Create a pool of all files for this class, grouped by criterion
            files_by_criterion = {}
            for cr in self.criteria:
                files_by_criterion[cr] = self.combined_labels[cl][cr].copy()

            # Get available criteria (those with at least one file)
            available_criteria = [cr for cr in self.criteria if len(files_by_criterion[cr]) > 0]

            # Calculate total files in this class
            total_files_in_class = sum(len(files) for files in files_by_criterion.values())

            if total_files_in_class < self.total_files // len(self.combined_labels.keys()):
                print(f"Warning: Class {cl} has fewer files ({total_files_in_class}) than requested ({self.total_files // len(self.combined_labels.keys())}).")

            # Add files to train set
            self._add_files_balanced(cl, files_by_criterion, available_criteria, train_files_per_class, self.train_set)

            # Update available criteria for val set
            available_criteria = [cr for cr in self.criteria if len(files_by_criterion[cr]) > 0]

            # Add files to validation set (only if val_files_per_class > 0)
            self._add_files_balanced(cl, files_by_criterion, available_criteria, val_files_per_class, self.val_set)

            # Update available criteria for test set
            available_criteria = [cr for cr in self.criteria if len(files_by_criterion[cr]) > 0]

            # Add files to test set (only if test_files_per_class > 0)
            self._add_files_balanced(cl, files_by_criterion, available_criteria, test_files_per_class, self.test_set)

    def create_random_split(self):  # This method contains errors
        """
        Create a random split where each class has exactly the specified number of files,
        with files selected randomly regardless of criterion.
        """
        # Calculate how many files to select per class based on the split ratios
        train_files_per_class = int(self.total_files * self.split_ratio[0])
        val_files_per_class = int(self.total_files * self.split_ratio[1])
        test_files_per_class = self.total_files - train_files_per_class - val_files_per_class

        print(f"Files per class - Train: {train_files_per_class}, Val: {val_files_per_class}, Test: {test_files_per_class}")

        # For each class, select the specified number of files for each set
        for cl in self.classes:
            # Create a pool of all files for this class
            all_files = []
            for cr in self.criteria:
                all_files.extend(self.combined_labels[cl][cr])

            if len(all_files) < self.total_files:
                print(f"Warning: Class {cl} has fewer files ({len(all_files)}) than requested ({self.total_files}).")
                continue

            # Shuffle the files
            rd.shuffle(all_files)

            # Add files to each set (respecting the counts)
            current_index = 0

            # Add files to train set
            if train_files_per_class > 0:
                self.train_set.extend(all_files[current_index:current_index + train_files_per_class])
                current_index += train_files_per_class

            # Add files to validation set
            if val_files_per_class > 0:
                self.val_set.extend(all_files[current_index:current_index + val_files_per_class])
                current_index += val_files_per_class

            # Add files to test set
            if test_files_per_class > 0:
                self.test_set.extend(all_files[current_index:current_index + test_files_per_class])

    def create_splits(self, total_files, merge_labels=None, merge_criteria=None):
        # Store the total_files parameter
        self.total_files = total_files

        # Clear any existing splits
        self.train_set = []
        self.val_set = []
        self.test_set = []

        # Get the number of distinct TrainingClasses and criteria
        self.get_number_of_distinct_criteria()

        # Merge class labels or criteria if required
        if merge_labels is not None:
            self.original_labels = self.class_labels.copy()
            for merge in merge_labels:
                self.merge_class_labels(merge, merge[0].split(" ")[0])

        if merge_criteria is not None:
            self.original_labels = self.criteria_labels.copy()
            for merge in merge_criteria:
                self.merge_criteria_labels(merge, merge[0].split(" ")[0])

        # Count files per class and criterion (for informational purposes)
        self.num_files_per_class = {label: sum(1 for filename in self.filenames if self.class_labels[filename] == label) for label in self.classes}
        self.num_files_per_criterion = {label: sum(1 for filename in self.filenames if self.criteria_labels[filename] == label) for label in self.criteria}

        print()
        print("Available files per class:", self.num_files_per_class)
        print("Available files per criterion:", self.num_files_per_criterion)
        print()

        # Combine labels and criteria into a single dictionary
        self.combine_labels_criteria()

        # Create the splits according to the selected method
        if self.split_method == "balanced":
            self.create_balanced_split()
        elif self.split_method == "random":
            self.create_random_split()

        # Print final set sizes
        print()
        print(f"Final set sizes - Train: {len(self.train_set)}, Val: {len(self.val_set)}, Test: {len(self.test_set)}")
        print()

    def export_to_excel(self, output_dir):
        """
        Export dataset information to three separate Excel files (train, val, test),
        each with three columns:
        - filename
        - class label
        - criteria

        Args:
            output_dir: Directory where the Excel files will be saved
        """
        self.classes = list(set(self.class_labels.values()))
        print(f"Distinct classes: {self.classes}")  # Debugging step

        class_num_labels = {}
        for num_label, class_label in enumerate(self.classes):
            class_num_labels[class_label] = num_label

        # Make sure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create DataFrames for each set
        train_data = []
        val_data = []
        test_data = []

        # Populate train set data
        for filename in self.train_set:
            train_data.append({
                'Filename': filename,
                'Filepath': os.path.join(self.data_source_path, self.original_labels[filename], f'{filename}.WAV'),
                'Criterion': self.criteria_labels[filename],
                'Class': self.class_labels[filename],
                'original_class': self.original_labels[filename],
                'label': class_num_labels[self.class_labels[filename]]
            })

        # Populate validation set data
        for filename in self.val_set:
            val_data.append({
                'Filename': filename,
                'Filepath': os.path.join(self.data_source_path, self.original_labels[filename], f'{filename}.WAV'),
                'Criterion': self.criteria_labels[filename],
                'Class': self.class_labels[filename],
                'original_class': self.original_labels[filename],
                'label': class_num_labels[self.class_labels[filename]]
            })

        # Populate test set data
        for filename in self.test_set:
            test_data.append({
                'Filename': filename,
                'Filepath': os.path.join(self.data_source_path, self.original_labels[filename], f'{filename}.WAV'),
                'Criterion': self.criteria_labels[filename],
                'Class': self.class_labels[filename],
                'original_class': self.original_labels[filename],
                'label': class_num_labels[self.class_labels[filename]]
            })

        # Convert to DataFrames
        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)
        test_df = pd.DataFrame(test_data)

        # Save to Excel files
        train_path = os.path.join(output_dir, "train_dataset_info.xlsx")
        val_path = os.path.join(output_dir, "val_dataset_info.xlsx")
        test_path = os.path.join(output_dir, "test_dataset_info.xlsx")

        train_df.to_excel(train_path, index=False)
        val_df.to_excel(val_path, index=False)
        test_df.to_excel(test_path, index=False)

        print(f"Train set information exported to {train_path}")
        print(f"Validation set information exported to {val_path}")
        print(f"Test set information exported to {test_path}")


if __name__ == "__main__":
    # Example usage
    data_source_path = "D:/Bachelorarbeit/AgroscopeData/LabelledSequences"
    labels_path = "D:/Bachelorarbeit/AgroscopeData/LabelledSequencesMerged.xlsx"
    data_target_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/DataAlphaBeta"

    split_set = SplitSet(data_source_path, labels_path, data_target_path)
    split_set.read_data("File", "Verification 1", "location")
    split_set.select_split_method("balanced")
    split_set.select_split_ratio(0.7, 0.15, 0.15)
    split_set.create_splits(20, merge_labels=[eptesicus_species, myotis_species, nyctalus_species, pipistrellus_species, Chiroptera_generally],)
    split_set.export_to_excel(os.path.join(data_target_path, "dataset_info"))
