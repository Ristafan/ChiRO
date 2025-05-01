import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd


class TensorMetadataCollector:
    def __init__(self, tensor_dir, tensor_extension=".pt"):
        self.tensor_dir = tensor_dir
        self.tensor_extension = tensor_extension
        self.dimensions = []
        self.value_ranges = []
        self.file_names = []

    def collect_metadata(self):
        for file in os.listdir(self.tensor_dir):
            if file.endswith(self.tensor_extension):
                tensor = torch.load(os.path.join(self.tensor_dir, file))

                self.dimensions.append(tensor.shape)
                values = tensor.cpu().flatten().numpy()
                self.value_ranges.append({
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                    "percentile_25": float(np.percentile(values, 25)),
                    "percentile_75": float(np.percentile(values, 75))
                })
                self.file_names.append(file)

    def summarize(self):
        dims_array = np.array(self.dimensions)

        # Gather value stats across all tensors
        all_values = {
            "min": [v["min"] for v in self.value_ranges],
            "max": [v["max"] for v in self.value_ranges],
            "mean": [v["mean"] for v in self.value_ranges],
            "std": [v["std"] for v in self.value_ranges],
            "median": [v["median"] for v in self.value_ranges],
            "percentile_25": [v["percentile_25"] for v in self.value_ranges],
            "percentile_75": [v["percentile_75"] for v in self.value_ranges],
        }

        value_summary = {
            k: {
                "mean": float(np.mean(v)),
                "min": float(np.min(v)),
                "max": float(np.max(v))
            }
            for k, v in all_values.items()
        }

        summary = {
            "channels": {
                "min": int(np.min(dims_array[:, 0])),
                "max": int(np.max(dims_array[:, 0])),
                "mean": float(np.mean(dims_array[:, 0]))
            },
            "freq_bins": {
                "min": int(np.min(dims_array[:, 1])),
                "max": int(np.max(dims_array[:, 1])),
                "mean": float(np.mean(dims_array[:, 1]))
            },
            "time_steps": {
                "min": int(np.min(dims_array[:, 2])),
                "max": int(np.max(dims_array[:, 2])),
                "mean": float(np.mean(dims_array[:, 2]))
            },
            "value_stats": value_summary,
            "num_files": len(self.dimensions)
        }

        return summary

    def plot_distributions(self):
        dims_array = np.array(self.dimensions)

        plt.figure(figsize=(18, 5))

        plt.subplot(1, 3, 1)
        plt.hist(dims_array[:, 0], bins=30, color='red', alpha=0.7)
        plt.title('Distribution of Channels')
        plt.xlabel('Channels')
        plt.ylabel('Count')

        plt.subplot(1, 3, 2)
        plt.hist(dims_array[:, 1], bins=30, color='blue', alpha=0.7)
        plt.title('Distribution of Frequency Bins (Height)')
        plt.xlabel('Frequency Bins')
        plt.ylabel('Count')

        plt.subplot(1, 3, 3)
        plt.hist(dims_array[:, 2], bins=30, color='green', alpha=0.7)
        plt.title('Distribution of Time Steps (Width)')
        plt.xlabel('Time Steps')
        plt.ylabel('Count')

        plt.tight_layout()
        plt.show()

    def save_metadata(self, output_path="tensor_metadata.json"):
        summary = self.summarize()

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"[INFO] Metadata saved to {output_path}")

    def save_detailed_dimensions(self, output_path="tensor_dimensions.csv"):
        # Save detailed dimension info to a CSV for inspection
        dims_array = np.array(self.dimensions)
        df = pd.DataFrame({
            "filename": self.file_names,
            "channels": dims_array[:, 0],
            "freq_bins": dims_array[:, 1],
            "time_steps": dims_array[:, 2],
            "min_value": [v["min"] for v in self.value_ranges],
            "max_value": [v["max"] for v in self.value_ranges],
            "mean_value": [v["mean"] for v in self.value_ranges],
            "std_value": [v["std"] for v in self.value_ranges],
            "median_value": [v["median"] for v in self.value_ranges],
            "percentile_25": [v["percentile_25"] for v in self.value_ranges],
            "percentile_75": [v["percentile_75"] for v in self.value_ranges]
        })
        df.to_csv(output_path, index=False)
        print(f"[INFO] Detailed tensor dimensions saved to {output_path}")


if __name__ == '__main__':
    # Example usage
    tensor_dir = 'C:/Users/MartinFaehnrich/Documents/ChiRO/data/Spectrograms'
    collector = TensorMetadataCollector(tensor_dir)
    collector.collect_metadata()
    collector.plot_distributions()
    collector.save_metadata(output_path="summary_metadata.json")
    collector.save_detailed_dimensions(output_path="detailed_dimensions.csv")
