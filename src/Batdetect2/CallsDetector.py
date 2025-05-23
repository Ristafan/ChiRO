import pandas as pd
import torch
import os

from tqdm import tqdm

from src.Batdetect2.Net2DFast import Net2DFast
from src.Batdetect2.parameters import TARGET_SAMPLERATE_HZ, FFT_WIN_LENGTH_S, FFT_OVERLAP, RESIZE_FACTOR, \
    SPEC_DIVIDE_FACTOR, SPEC_HEIGHT, SCALE_RAW_AUDIO, NMS_KERNEL_SIZE, MAX_FREQ_HZ, MIN_FREQ_HZ, NMS_TOP_K_PER_SEC, \
    SPEC_SCALE, DENOISE_SPEC_AVG, MAX_SCALE_SPEC
from src.Batdetect2.types import ProcessingConfiguration
from src.Batdetect2.detector_utils import process_file


class CallsDetector:
    def __init__(self, model, model_path, labels_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels_path = labels_path
        self.model_path = model_path
        self.model = model
        self.load_model()

        self.filenames = []
        self.filepaths = []

        self.num_calls = {}
        self.start_times = {}
        self.end_times = {}

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

    def load_filenames_and_filepaths(self, filename_column="Filename", filepath_column="Filepath"):
        data = pd.read_excel(self.labels_path)
        self.filenames = data[filename_column].tolist()
        self.filepaths = data[filepath_column].tolist()

    def save_predictions(self):
        # Load the original data
        df = pd.read_excel(self.labels_path)

        # Convert lists to strings for easier DataFrame creation
        for key in self.start_times:
            self.start_times[key] = str(self.start_times[key])
            self.end_times[key] = str(self.end_times[key])

        # Create DataFrames from each dictionary using 'Filepath' as the key
        df_start = pd.DataFrame.from_dict(self.start_times, orient='index', columns=['start_time'])
        df_end = pd.DataFrame.from_dict(self.end_times, orient='index', columns=['end_time'])
        df_calls = pd.DataFrame.from_dict(self.num_calls, orient='index', columns=['num_calls'])

        # Ensure index is labeled 'Filepath' to merge correctly
        df_start.index.name = 'Filepath'
        df_end.index.name = 'Filepath'
        df_calls.index.name = 'Filepath'

        # Merge with original data based on 'Filepath'
        df = df.merge(df_start, on='Filepath', how='left')
        df = df.merge(df_end, on='Filepath', how='left')
        df = df.merge(df_calls, on='Filepath', how='left')

        # Save to new Excel file
        df.to_excel(self.labels_path, index=False)

    def predict_set(self):

        DETECTION_THRESHOLD = 0.5
        TARGET_SAMPLERATE_HZ = 192000

        processing_configuration = ProcessingConfiguration(
            {
                "detection_threshold": DETECTION_THRESHOLD,
                "class_names": ["bat1", "bat2", "bat3", "bat4", "bat5", "bat6", "bat7", "bat8", "bat9", "bat10",
                                "bat11", "bat12", "bat13", "bat14", "bat15", "bat16", "bat17", "bat18"],
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

        for file in tqdm(self.filepaths, desc="Detecting Calls", unit="file"):
            if file.lower().endswith('.wav'):
                prediction = process_file(file, self.model, processing_configuration, self.device)
                self.start_times[file] = prediction["start_times"]
                self.end_times[file] = prediction["end_times"]
                self.num_calls[file] = len(prediction["start_times"])

            # det_probs = file[0]["det_probs"]
            # x_pos = file[0]["x_pos"]
            # y_pos = file[0]["y_pos"]
            # bb_widths = file[0]["bb_widths"]
            # bb_heights = file[0]["bb_heights"]
            # low_freqs = file[0]["low_freqs"]
            # high_freqs = file[0]["high_freqs"]
            # class_probs = file[0]["class_probs"]
            # bb_width = file[0]["bb_width"]
            # bb_height = file[0]["bb_height"]


if __name__ == "__main__":
    # Example usage
    model = Net2DFast(num_filts=64)

    model_path = "Net2DFast_UK_same.pth.tar"
    labels_path = "C:/Users/MartinFaehnrich/Documents/ChiRO/data/ExampleData/dataset_info/train_dataset_info.xlsx"
    predictor = CallsDetector(model, model_path, labels_path)
    predictor.load_filenames_and_filepaths()
    predictor.predict_set()
    predictor.save_predictions()
