from src.Logging.Logger import Logger
import pandas as pd
import torch


class LabelsLoader:
    def __init__(self, path, filename_column="Filename", text_column="Label", additional_column=None):
        self.path = path
        self.filename_column = filename_column
        self.text_column = text_column
        self.additional_column = additional_column

        self.filenames = []
        self.labels = {}
        self.additional = {}

    def load_labels_excel(self):
        data = pd.read_excel(self.path)
        self.filenames = data[self.filename_column].tolist()
        self.labels = {row[self.filename_column]: int(row[self.text_column]) for _, row in data.iterrows()}

        if self.additional_column:
            self.additional = {row[self.filename_column]: row[self.additional_column] for _, row in data.iterrows()}

    def get_labels(self):
        return self.labels

    def get_filenames(self):
        return self.filenames

    def get_additional(self):
        return self.additional


if __name__ == '__main__':
    labels_loader = LabelsLoader('C:/Users/MartinFaehnrich/Documents/ChiRO/data/Labels/labels.xlsx',
                                 'Filename', 'Label')
    labels_loader.load_labels_excel()
    print(labels_loader.get_labels())
    print(labels_loader.get_filenames())
