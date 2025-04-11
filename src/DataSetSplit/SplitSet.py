import pandas as pd
import os
import random as rd
import shutil as sh


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

    def create_balanced_split(self, num_elements, set_type):
        for element in range(num_elements):
            for cl in self.classes:
                for cr in self.criteria:
                    if len(self.combined_labels[cl][cr]) > 0:
                        selected_file = rd.choice(self.combined_labels[cl][cr])
                        set_type.append(selected_file)
                        self.combined_labels[cl][cr].remove(selected_file)

                        self.num_files_per_class[cl] -= 1
                        self.num_files_per_criterion[cr] -= 1

                    else:
                        # select a file from another criterion from the same class
                        for cr2 in self.criteria:
                            if cr2 != cr and len(self.combined_labels[cl][cr2]) > 0:
                                selected_file = rd.choice(self.combined_labels[cl][cr2])
                                set_type.append(selected_file)
                                self.combined_labels[cl][cr2].remove(selected_file)

                                self.num_files_per_class[cl] -= 1
                                self.num_files_per_criterion[cr2] -= 1
                                break

                            # if no other criterion is available, raise warning
                            if cr2 == self.criteria[-1]:
                                print(f"Warning: No more files available for class {cl}.")
                                break

    def create_random_split(self, num_elements, set_type):
        for element in range(num_elements):
            for cl in self.classes:
            # Check if there are files available for the class
                random_criterion = rd.choice(self.criteria)
                if len(self.combined_labels[cl][random_criterion]) > 0:
                    selected_file = rd.choice(self.combined_labels[cl][random_criterion])
                    set_type.append(selected_file)
                    self.combined_labels[cl][random_criterion].remove(selected_file)

                    self.num_files_per_class[cl] -= 1
                    self.num_files_per_criterion[random_criterion] -= 1
                else:
                    # select a file from another random criterion from the same class
                    for i in range(len(self.criteria) - 1):
                        random_criterion = rd.choice(self.criteria)
                        if len(self.combined_labels[cl][random_criterion]) > 0:
                            selected_file = rd.choice(self.combined_labels[cl][random_criterion])
                            set_type.append(selected_file)
                            self.combined_labels[cl][random_criterion].remove(selected_file)

                            self.num_files_per_class[cl] -= 1
                            self.num_files_per_criterion[random_criterion] -= 1
                            break

                    # if no other criterion is available, raise warning
                    if random_criterion == self.criteria[-1]:
                        print(f"Warning: No more files available for class {cl}.")
                        break

    def create_splits(self, files_per_class, merge_labels=None, merge_criteria=None):
        # Get the number of distinct classes and criteria
        self.get_number_of_distinct_criteria()

        # Merge class labels or criteria if required
        if merge_labels is not None:
            self.merge_class_labels(merge_labels)
        if merge_criteria is not None:
            self.merge_criteria_labels(merge_criteria)

        # Split the data according to the selected method and criteria
        self.num_files_per_class = {label: sum(1 for filename in self.filenames if self.class_labels[filename] == label) for label in self.classes}
        self.num_files_per_criterion = {label: sum(1 for filename in self.filenames if self.criteria_labels[filename] == label) for label in self.criteria}

        num_train = int(files_per_class * self.split_ratio[0])
        num_val = int(files_per_class * self.split_ratio[1])
        num_test = int(files_per_class * self.split_ratio[2])

        # Combine labels and criteria into a single dictionary
        self.combine_labels_criteria()

        # Select iteratively from each class and criterion a random file
        if self.split_method == "balanced":
            self.create_balanced_split(num_train, self.train_set)
            self.create_balanced_split(num_val, self.val_set)
            self.create_balanced_split(num_test, self.test_set)

        elif self.split_method == "random":
            self.create_random_split(num_train, self.train_set)
            self.create_random_split(num_val, self.val_set)
            self.create_random_split(num_test, self.test_set)

    def move_files(self, target_path):
        # Check if target path exists, if not create it
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        # Move files to the target path
        for filename in self.train_set:
            source = os.path.join(self.data_source_path, self.original_labels[filename], f'{filename}.wav')
            target = os.path.join(target_path, "train", f'{filename}.wav')
            os.makedirs(os.path.dirname(target), exist_ok=True)
            sh.copyfile(source, target)

        for filename in self.val_set:
            source = os.path.join(self.data_source_path, self.original_labels[filename], f'{filename}.wav')
            target = os.path.join(target_path, "val", f'{filename}.wav')
            os.makedirs(os.path.dirname(target), exist_ok=True)
            sh.copyfile(source, target)

        for filename in self.test_set:
            source = os.path.join(self.data_source_path, self.original_labels[filename], f'{filename}.wav')
            target = os.path.join(target_path, "test", f'{filename}.wav')
            os.makedirs(os.path.dirname(target), exist_ok=True)
            sh.copyfile(source, target)


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
    data_target_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/Test_SplitData"

    split_set = SplitSet(data_source_path, labels_path, data_target_path)
    split_set.read_data("File", "Verification 1", "location")
    split_set.select_split_method("balanced")
    split_set.select_split_ratio(0.8, 0.1, 0.1)
    split_set.create_splits(30, merge_labels=bat_species)
    split_set.move_files(data_target_path)
