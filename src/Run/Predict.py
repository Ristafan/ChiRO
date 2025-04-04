from src.Logging.Logger import Logger
from src.Preprocessing.AudioLoader import AudioLoader
from src.Preprocessing.SpectrogramProcessor import SpectrogramProcessor
from src.Architectures.AlphaV1 import AlphaV1

import torch
import os


class Predictor():
    def __init__(self, model, model_path, data_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_path = data_path
        self.model_path = model_path
        self.model = model

    def load_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            Logger().log_debug("Model loaded successfully.")
        except FileNotFoundError:
            Logger().log_debug("Model file not found.")
            raise ValueError("Model file not found")

    def predict_audio_file(self, filename):
        """
        Predicts the class of a single audio file.
        :param filename: Path to the audio file.
        :return: Predicted class.
        """
        # Load the audio file
        audio_loader = AudioLoader(self.data_path)
        waveform, _ = audio_loader.load_wav_file(filename)

        # Process the waveform
        spectrogram_processor = SpectrogramProcessor(waveform)
        spectrogram_processor.apply_highpass_filter()
        spectrogram_processor.compute_spectrogram()
        spectrogram_processor.denoise_spectrogram()
        spectrogram = spectrogram_processor.spectrogram

        # Load the model
        self.load_model()

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            spectrogram = spectrogram.to(self.device)
            output = self.model(spectrogram.unsqueeze(0))
            print(output)
            _, predicted_class = torch.max(output, 1)

        return predicted_class.item()

    def predict_folder(self):
        """
        Predicts the classes of all audio files in a folder.
        :return: List of predicted classes.
        """
        predictions = []
        for filename in os.listdir(self.data_path):
            if filename.lower().endswith('.wav'):
                prediction = self.predict_audio_file(filename)
                predictions.append(prediction)
        return predictions


if __name__ == "__main__":
    # Example usage
    model = AlphaV1()  # Replace with your model
    model_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/src/Models/Alpha/alphaV1.pth"
    data_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/ExampleData"
    predictor = Predictor(model, model_path, data_path)
    #prediction = predictor.predict_audio_file("20220705_220800T_555b4a69c5e3062b73b1fb2448b5addb.wav")
    #print(f"Predicted class: {prediction}")

    predictions = predictor.predict_folder()
    print(f"Predicted classes: {predictions}")
