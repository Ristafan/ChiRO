import pandas as pd
import os
import random as rd


class SplitSet:
    def __init__(self, data_path, labels_path):
        self.data_path = data_path
        self.labels_path = labels_path

        self.filenames = []
        self.classes = []
        self.criteria = []
        self.split_criterion = ""
        self.split_method = 0
        self.split_ratio = [0.8, 0.2, 0]
        self.split_seed = 0

        self.class_labels = {}
        self.criteria_labels = {}
        self.combined_labels = {}

        self.train_set = []
        self.val_set = []
        self.test_set = []

    def select_classes(self, classes):
        self.classes = classes

    def select_split_criterion(self, split_criterion):
        self.split_criterion = split_criterion

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
        self.class_labels = {row[filename_column]: int(row[label_column]) for _, row in data.iterrows()}
        self.criteria_labels = {row[filename_column]: int(row[criterion_column]) for _, row in data.iterrows()}

    def get_filenames(self):
        return self.filenames

    def get_class_labels(self):
        return self.class_labels

    def get_criteria_labels(self):
        return self.criteria_labels

    def get_combined_labels(self):
        return self.combined_labels

    def get_number_of_distinct_criteria(self):
        for criterion in self.criteria_labels.values():
            if criterion not in self.criteria:
               self.criteria.append(criterion)

    def combine_labels_criteria(self):
        self.get_number_of_distinct_criteria()

        # Create dictionary to store combined labels
        self.combined_labels = {}
        for cl in (self.classes):
            self.combined_labels[cl] = {}
            for cr in self.criteria:
                self.combined_labels[cl][cr] = []

        # Populate combined labels
        for filename in self.filenames:
            class_label = self.class_labels[filename]
            criterion_label = self.criteria_labels[filename]
            self.combined_labels[class_label][criterion_label].append(filename)

    def create_balanced_split(self, num_elements, set_type, num_files_per_class, num_files_per_criterion):
        for element in range(num_elements):
            for cl in self.classes:
                for cr in self.criteria:
                    if len(self.combined_labels[cl][cr]) > 0:
                        selected_file = rd.choice(self.combined_labels[cl][cr])
                        set_type.append(selected_file)
                        self.combined_labels[cl][cr].remove(selected_file)

                        num_files_per_class[cl] -= 1
                        num_files_per_criterion[cr] -= 1

                    else:
                        # select a file from another criterion from the same class
                        for cr2 in self.criteria:
                            if cr2 != cr and len(self.combined_labels[cl][cr2]) > 0:
                                selected_file = rd.choice(self.combined_labels[cl][cr2])
                                set_type.append(selected_file)
                                self.combined_labels[cl][cr2].remove(selected_file)

                                num_files_per_class[cl] -= 1
                                num_files_per_criterion[cr2] -= 1
                                break

    def create_splits(self):
        # Split the data according to the selected method and criteria
        num_files_per_class = {label: sum(1 for filename in self.filenames if self.class_labels[filename] == label) for label in self.classes}
        num_files_per_criterion = {label: sum(1 for filename in self.filenames if self.criteria_labels[filename] == label) for label in self.criteria}

        num_train = int(len(self.filenames) * self.split_ratio[0])
        num_val = int(len(self.filenames) * self.split_ratio[1])
        num_test = len(self.filenames) - num_train - num_val

        # Combine labels and criteria into a single dictionary
        self.combine_labels_criteria()

        # Select iteratively from each class and criterion a random file
        if self.split_method == "balanced":
            self.create_balanced_split(num_train, self.train_set, num_files_per_class, num_files_per_criterion)
            self.create_balanced_split(num_val, self.val_set, num_files_per_class, num_files_per_criterion)
            self.create_balanced_split(num_test, self.test_set, num_files_per_class, num_files_per_criterion)

        elif self.split_method == "random":


















