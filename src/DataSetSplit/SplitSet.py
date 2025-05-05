import pandas as pd
import os
import random as rd
import shutil as sh
from collections import defaultdict
from tqdm import tqdm


class SplitSet:
    def __init__(self, data_source_path, labels_path, data_target_path):
        self.data_source_path = data_source_path
        self.data_target_path = data_target_path
        self.labels_path = labels_path

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

    def merge_class_labels(self, to_merge):
        self.original_labels = self.class_labels.copy()
        # Merge class labels based on the provided list
        for filename in self.filenames:
            if self.class_labels[filename] in to_merge:
                self.class_labels[filename] = 'new_class'

        # Update the classes list
        self.classes = list(set(self.class_labels.values()))

    def merge_criteria_labels(self, to_merge):
        # Merge criteria labels based on the provided list
        for filename in self.filenames:
            if self.criteria_labels[filename] in to_merge:
                self.criteria_labels[filename] = 'new_criterion'

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

    def export_to_excel(self, output_dir, enumerate_classes=True):
        """
        Export dataset information to three separate Excel files (train, val, test),
        each with three columns:
        - filename
        - class label
        - criteria

        Args:
            output_dir: Directory where the Excel files will be saved
        """
        class_num_labels = {}
        if enumerate_classes:
            num_label = 0
            for i in self.classes:
                class_num_labels[i] = num_label
                num_label += 1

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
                'Class': self.class_labels[filename],
                'Criterion': self.criteria_labels[filename],
                'label': class_num_labels[self.class_labels[filename]] if enumerate_classes else self.class_labels[filename]
            })

        # Populate validation set data
        for filename in self.val_set:
            val_data.append({
                'Filename': filename,
                'Class': self.class_labels[filename],
                'Criterion': self.criteria_labels[filename],
                'label': class_num_labels[self.class_labels[filename]] if enumerate_classes else self.class_labels[filename]
            })

        # Populate test set data
        for filename in self.test_set:
            test_data.append({
                'Filename': filename,
                'Class': self.class_labels[filename],
                'Criterion': self.criteria_labels[filename],
                'label': class_num_labels[self.class_labels[filename]] if enumerate_classes else self.class_labels[filename]
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

        # Report counts
        print(f"Records exported - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

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
        train_files_per_class = int(self.files_per_class * self.split_ratio[0])
        val_files_per_class = int(self.files_per_class * self.split_ratio[1])
        test_files_per_class = self.files_per_class - train_files_per_class - val_files_per_class

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

            if total_files_in_class < self.files_per_class:
                print(f"Warning: Class {cl} has fewer files ({total_files_in_class}) than requested ({self.files_per_class}).")
                continue

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

    def create_random_split(self):
        """
        Create a random split where each class has exactly the specified number of files,
        with files selected randomly regardless of criterion.
        """
        # Calculate how many files to select per class based on the split ratios
        train_files_per_class = int(self.files_per_class * self.split_ratio[0])
        val_files_per_class = int(self.files_per_class * self.split_ratio[1])
        test_files_per_class = self.files_per_class - train_files_per_class - val_files_per_class

        print(f"Files per class - Train: {train_files_per_class}, Val: {val_files_per_class}, Test: {test_files_per_class}")

        # For each class, select the specified number of files for each set
        for cl in self.classes:
            # Create a pool of all files for this class
            all_files = []
            for cr in self.criteria:
                all_files.extend(self.combined_labels[cl][cr])

            if len(all_files) < self.files_per_class:
                print(f"Warning: Class {cl} has fewer files ({len(all_files)}) than requested ({self.files_per_class}).")
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

    def create_splits(self, files_per_class, merge_labels=None, merge_criteria=None):
        # Store the files_per_class parameter
        self.files_per_class = files_per_class

        # Clear any existing splits
        self.train_set = []
        self.val_set = []
        self.test_set = []

        # Get the number of distinct classes and criteria
        self.get_number_of_distinct_criteria()

        # Merge class labels or criteria if required
        if merge_labels is not None:
            self.merge_class_labels(merge_labels)
        if merge_criteria is not None:
            self.merge_criteria_labels(merge_criteria)

        # Count files per class and criterion (for informational purposes)
        self.num_files_per_class = {label: sum(1 for filename in self.filenames if self.class_labels[filename] == label) for label in self.classes}
        self.num_files_per_criterion = {label: sum(1 for filename in self.filenames if self.criteria_labels[filename] == label) for label in self.criteria}

        print("Available files per class:", self.num_files_per_class)
        print("Available files per criterion:", self.num_files_per_criterion)

        # Combine labels and criteria into a single dictionary
        self.combine_labels_criteria()

        # Create the splits according to the selected method
        if self.split_method == "balanced":
            self.create_balanced_split()
        elif self.split_method == "random":
            self.create_random_split()

        # Print final set sizes
        print(f"Final set sizes - Train: {len(self.train_set)}, Val: {len(self.val_set)}, Test: {len(self.test_set)}")

    def export_to_excel(self, output_dir, enumerate_classes=True):
        """
        Export dataset information to three separate Excel files (train, val, test),
        each with three columns:
        - filename
        - class label
        - criteria

        Args:
            output_dir: Directory where the Excel files will be saved
        """
        class_num_labels = {}
        if enumerate_classes:
            num_label = 0
            for i in self.classes:
                class_num_labels[i] = num_label
                num_label += 1

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
                'Filepath': os.path.join(self.data_source_path, self.original_labels[filename], f'{filename}.wav'),
                'Class': self.class_labels[filename],
                'Criterion': self.criteria_labels[filename],
                'label': class_num_labels[self.class_labels[filename]] if enumerate_classes else self.class_labels[filename]
            })

        # Populate validation set data
        for filename in self.val_set:
            val_data.append({
                'Filename': filename,
                'Filepath': os.path.join(self.data_source_path, self.original_labels[filename], f'{filename}.wav'),
                'Class': self.class_labels[filename],
                'Criterion': self.criteria_labels[filename],
                'label': class_num_labels[self.class_labels[filename]] if enumerate_classes else self.class_labels[filename]
            })

        # Populate test set data
        for filename in self.test_set:
            test_data.append({
                'Filename': filename,
                'Filepath': os.path.join(self.data_source_path, self.original_labels[filename], f'{filename}.wav'),
                'Class': self.class_labels[filename],
                'Criterion': self.criteria_labels[filename],
                'label': class_num_labels[self.class_labels[filename]] if enumerate_classes else self.class_labels[filename]
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

    def move_files(self, target_path):
        # Check if target path exists, if not create it
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        # Move files to the target path
        for filename in tqdm(self.train_set, desc="Moving train files"):
            source = os.path.join(self.data_source_path, self.original_labels[filename], f'{filename}.wav')
            target = os.path.join(target_path, "train", f'{filename}.wav')
            os.makedirs(os.path.dirname(target), exist_ok=True)
            sh.copyfile(source, target)

        for filename in tqdm(self.val_set, desc="Moving val files"):
            source = os.path.join(self.data_source_path, self.original_labels[filename], f'{filename}.wav')
            target = os.path.join(target_path, "val", f'{filename}.wav')
            os.makedirs(os.path.dirname(target), exist_ok=True)
            sh.copyfile(source, target)

        for filename in tqdm(self.test_set, desc="Moving test files"):
            source = os.path.join(self.data_source_path, self.original_labels[filename], f'{filename}.wav')
            target = os.path.join(target_path, "test", f'{filename}.wav')
            os.makedirs(os.path.dirname(target), exist_ok=True)
            sh.copyfile(source, target)

    def count_files_in_folder(self, folder_path):
        """
        Count files in a folder grouped by class and criterion.

        Args:
            folder_path: Path to the folder containing files to count

        Returns:
            Prints the count of files by class and criterion
        """
        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"Error: Folder {folder_path} does not exist")
            return

        # Initialize counters
        class_counts = defaultdict(int)
        criterion_counts = defaultdict(int)
        class_criterion_counts = defaultdict(lambda: defaultdict(int))

        # Get files in the folder
        files = []
        for root, _, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.endswith('.wav'):
                    # Extract just the filename without extension
                    base_name = os.path.splitext(filename)[0]
                    files.append(base_name)

        # Count files by class and criterion
        for filename in files:
            if filename in self.class_labels:
                class_label = self.class_labels[filename]
                class_counts[class_label] += 1

                if filename in self.criteria_labels:
                    criterion_label = self.criteria_labels[filename]
                    criterion_counts[criterion_label] += 1
                    class_criterion_counts[class_label][criterion_label] += 1

        # Print results
        print(f"\nFile counts in {folder_path}:")
        print("=" * 50)

        print("\nCounts by Class:")
        print("-" * 50)
        for cls, count in sorted(class_counts.items()):
            print(f"{cls}: {count} files")

        print("\nCounts by Criterion:")
        print("-" * 50)
        for criterion, count in sorted(criterion_counts.items()):
            print(f"{criterion}: {count} files")

        print("\nCounts by Class and Criterion:")
        print("-" * 50)
        for cls in sorted(class_criterion_counts.keys()):
            print(f"\nClass: {cls}")
            for criterion, count in sorted(class_criterion_counts[cls].items()):
                print(f"  - {criterion}: {count} files")

        print("\nTotal files:", len(files))
        return {
            "class_counts": dict(class_counts),
            "criterion_counts": dict(criterion_counts),
            "class_criterion_counts": {k: dict(v) for k, v in class_criterion_counts.items()},
            "total": len(files)
        }


if __name__ == "__main__":
    # Example usage
    bat_species = [
        "Barbastella barbastellus",
        "Chiroptera",
        "Eptesicus nilssonii",
        "Eptesicus serotinus",
        "Hypsugo savii",
        "Miniopterus schreibersii",
        "Myotis bechsteinii or Myotis sp.",
        "Myotis brandtii or Myotis mystacinus",
        "Myotis capaccinii or Myotis daubentonii",
        "Myotis daubentonii",
        "Myotis daubentonii or Myotis capaccinii",
        "Myotis emarginatus",
        "Myotis myotis",
        "Myotis nattereri",
        "Myotis sp.",
        "Nyctalus leisleri",
        "Nyctalus noctula",
        "Pipistrellus kuhlii",
        "Pipistrellus kuhlii or Pipistrellus nathusii",
        "Pipistrellus nathusii or Pipistrellus kuhlii",
        "Pipistrellus nathusii",
        "Pipistrellus pipistrellus",
        "Pipistrellus pipistrellus or Pipistrellus kuhlii",
        "Pipistrellus pygmaeus",
        "Plecotus austriacus",
        "Vespertilio murinus"
    ]

    data_source_path = "D:/Bachelorarbeit/AgroscopeData/LabelledSequences"
    labels_path = "D:/Bachelorarbeit/AgroscopeData/LabelledSequencesMerged.xlsx"
    data_target_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/DataAlpha"

    split_set = SplitSet(data_source_path, labels_path, data_target_path)
    split_set.read_data("File", "Verification 1", "location")
    split_set.select_split_method("balanced")
    split_set.select_split_ratio(0.7, 0.15, 0.15)
    split_set.create_splits(40, merge_labels=bat_species)
    # split_set.move_files(data_target_path)
    split_set.export_to_excel(os.path.join(data_target_path, "dataset_info"), enumerate_classes=True)
    split_set.export_to_excel(os.path.join(data_target_path, "dataset_info"))

    # Count files in the train, validation, and test folders
    # print("\nCounting files in the target folders:")
    # split_set.count_files_in_folder(os.path.join(data_target_path, "train"))
    # split_set.count_files_in_folder(os.path.join(data_target_path, "val"))
    # split_set.count_files_in_folder(os.path.join(data_target_path, "test"))
