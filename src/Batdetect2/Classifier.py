import torch
import os

from model import Net2DFast
from parameters import TARGET_SAMPLERATE_HZ, FFT_WIN_LENGTH_S, FFT_OVERLAP, RESIZE_FACTOR, \
    SPEC_DIVIDE_FACTOR, SPEC_HEIGHT, SCALE_RAW_AUDIO, NMS_KERNEL_SIZE, MAX_FREQ_HZ, MIN_FREQ_HZ, NMS_TOP_K_PER_SEC, \
    SPEC_SCALE, DENOISE_SPEC_AVG, MAX_SCALE_SPEC
from src.Batdetect2.types import ProcessingConfiguration
from detector_utils import process_file


class Predictor:
    def __init__(self, model, model_path, data_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_path = data_path
        self.model_path = model_path
        self.model = model
        self.load_model()

    def load_model(self):
        """
        Load the model and its parameters from the specified path.
        """
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError("Model file not found.")

        # Load model parameters and weights
        net_params = torch.load(self.model_path, map_location=self.device)
        params = net_params["params"]

        # Initialize the model based on the model name
        if params["model_name"] == "Net2DFast":
            self.model = Net2DFast(
                params["num_filters"],
                num_classes=17,
                emb_dim=params["emb_dim"],
                ip_height=params["ip_height"],
                resize_factor=params["resize_factor"],
            )
        else:
            raise ValueError("Unknown model.")

        # Load weights into the model
        self.model.load_state_dict(net_params["state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict_folder(self):
        """
        Predicts the classes of all audio files in a folder.
        :return: List of predicted classes.
        """
        DETECTION_THRESHOLD = 0.5

        processing_configuration = ProcessingConfiguration(
            {
                "detection_threshold": DETECTION_THRESHOLD,
                "class_names": ["bat1", "bat2", "bat3", "bat4", "bat5", "bat6", "bat7", "bat8", "bat9", "bat10", "bat11", "bat12", "bat13", "bat14", "bat15", "bat16", "bat17", "bat18"],
                "spec_slices": False,
                "chunk_size": 3,
                "spec_features": False,
                "cnn_features": False,
                "quiet": True,
                "target_samp_rate": TARGET_SAMPLERATE_HZ,
                "fft_win_length": FFT_WIN_LENGTH_S,
                "fft_overlap": FFT_OVERLAP,
                "resize_factor": RESIZE_FACTOR,
                "spec_divide_factor": SPEC_DIVIDE_FACTOR,
                "spec_height": SPEC_HEIGHT,
                "scale_raw_audio": SCALE_RAW_AUDIO,
                "time_expansion": 1,
                "top_n": 3,
                "return_raw_preds": True,
                "max_duration": None,
                "nms_kernel_size": NMS_KERNEL_SIZE,
                "max_freq": MAX_FREQ_HZ,
                "min_freq": MIN_FREQ_HZ,
                "nms_top_k_per_sec": NMS_TOP_K_PER_SEC,
                "spec_scale": SPEC_SCALE,
                "denoise_spec_avg": DENOISE_SPEC_AVG,
                "max_scale_spec": MAX_SCALE_SPEC,
            }
        )

        predictions = []
        for filename in os.listdir(self.data_path):
            if filename.lower().endswith('.wav'):
                prediction = process_file(os.path.join(self.data_path, filename), self.model, processing_configuration, self.device)
                predictions.append(prediction)

        if not predictions:
            raise ValueError("No .wav files found in the specified folder.")

        for prediction in predictions:
            print(f"\nProcessing file")
            det_probs = prediction["det_probs"]
            x_pos = prediction["x_pos"]
            y_pos = prediction["y_pos"]
            bb_widths = prediction["bb_widths"]
            bb_heights = prediction["bb_heights"]
            start_times = prediction["start_times"]
            end_times = prediction["end_times"]
            low_freqs = prediction["low_freqs"]
            high_freqs = prediction["high_freqs"]
            class_probs = prediction["class_probs"]
            bb_width = prediction["bb_width"]
            bb_height = prediction["bb_height"]

            print(len(det_probs), "detections found in file")
            print(len(x_pos), "x positions found in file")
            print(len(y_pos), "y positions found in file")
            print(len(bb_widths), "bounding box widths found in file")
            print(len(bb_heights), "bounding box heights found in file")
            print(len(start_times), "start times found in file")
            print(len(end_times), "end times found in file")
            print(len(low_freqs), "low frequencies found in file")
            print(len(class_probs), "class probabilities found in file")

            print()


            #print(f"Detection probabilities: {det_probs}")
            #print(f"X positions: {x_pos}")
            #print(f"Y positions: {y_pos}")
            #print(f"Bounding box widths: {bb_widths}")
            #print(f"Bounding box heights: {bb_heights}")
            #print(f"Start times: {start_times}")
            #print(f"End times: {end_times}")
            #print(f"Low frequencies: {low_freqs}")
            #print(f"High frequencies: {high_freqs}")
            #print(f"Class probabilities: {class_probs}")
            #print(f"Bounding box width: {bb_width}")
            #print(f"Bounding box height: {bb_height}")

        return predictions


if __name__ == "__main__":
    # Example usage
    model = Net2DFast(num_filts=64)

    model_path = "Net2DFast_UK_same.pth.tar"
    data_path = "G:/Andere Computer/Mein Laptop/Bachelorarbeit/Code/BatsCode/batdetect2-main/example_data/audio"  # Replace with your data path
    predictor = Predictor(model, model_path, data_path)

    predictions = predictor.predict_folder()
    print(f"Predicted classes: {predictions}")
