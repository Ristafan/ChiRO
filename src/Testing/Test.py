import os.path

from src.Architectures.AlphaV1 import AlphaV1
from src.Logging.Logger import Logger

import torch

from src.Preprocessing.AudioLoader import AudioLoader
from src.Preprocessing.BatCallDataSet import BatCallDataset
from src.Preprocessing.LabelsLoader import LabelsLoader
from src.Preprocessing.SpectrogramProcessor import SpectrogramProcessor


class Test:
    def __init__(self, model, model_path, data_path, labels_file_path, spectrogram_processed=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_path = data_path
        self.labels_file_path = labels_file_path
        self.model_path = model_path
        self.model = model
        self.spectrogram_processed = spectrogram_processed

    def load_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            Logger().log_debug("Model loaded successfully.")
        except FileNotFoundError:
            Logger().log_debug("Model file not found.")
            raise ValueError("Model file not found")

    def run_test(self):
        """
        Run the test on the model.
        :return: None
        """
        # Load the model
        self.load_model()

        if not self.spectrogram_processed:
            # Load the data
            data_loader = AudioLoader(self.data_path)
            data_loader.load_folder()
            data = data_loader.get_data()

            if not os.path.exists(self.data_path + '/TestSpectrograms/'):
                os.makedirs(self.data_path + '/TestSpectrograms/')

            for i in range(len(data)):
                # Process the data
                spectrogram_processor = SpectrogramProcessor(data[i])
                spectrogram_processor.apply_highpass_filter()
                spectrogram_processor.compute_spectrogram()
                spectrogram_processor.denoise_spectrogram()
                spectrogram_processor.save_spectrogram(f'{data_loader.get_file_names()[i]}', self.data_path + '/TestSpectrograms/')

        # Load the labels
        labels_loader = LabelsLoader(self.labels_file_path,'Filename', 'Label')
        labels_loader.load_labels_excel()

        # Load the spectrogram
        dataset = BatCallDataset(self.data_path + '/TestSpectrograms/', labels_loader)

        # Make predictions
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(len(dataset)):
                spectrogram, _ = dataset[i]
                spectrogram = spectrogram.unsqueeze(0).to(self.device)
                output = self.model(spectrogram)
                predictions.append(output)

        Logger().log_debug("Test completed successfully.")

        # Evaluate the model
        accuracy, precision, recall, f1_score = self.evaluate(predictions, dataset.get_labels())
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1_score:.2f}")

    @staticmethod
    def evaluate(predictions, labels):
        correct = 0
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        total = len(labels)

        for i in range(total):
            if torch.argmax(predictions[i]) == labels[i]:
                correct += 1

                if labels[i] == 1:
                    true_positives += 1

                if labels[i] == 0:
                    true_negatives += 1

            if labels[i] == 1:
                false_negatives += 1

            if labels[i] == 0:
                false_positives += 1

        accuracy = correct / total

        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0.001
            Logger().log_debug("Precision is 0, setting to 0.001 to avoid division by zero.")

        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0.001
            Logger().log_debug("Recall is 0, setting to 0.001 to avoid division by zero.")

        f1_score = 2 * (precision * recall) / (precision + recall)

        Logger().log_debug(f"Model Accuracy: {accuracy * 100:.2f}%")
        Logger().log_debug(f"Precision: {precision:.2f}")
        Logger().log_debug(f"Recall: {recall:.2f}")
        Logger().log_debug(f"F1 Score: {f1_score:.2f}")

        return accuracy, precision, recall, f1_score


if __name__ == "__main__":
    # Example usage
    model = AlphaV1()  # Replace with your model
    model_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/src/Models/Alpha/alphaV1.pth"
    data_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/ExampleData"
    labels_file_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/Labels/labels_exampledata.xlsx"
    test = Test(model, model_path, data_path, labels_file_path, spectrogram_processed=False)
    test.run_test()



