import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# Set this to the root directory containing run_* folders
ROOT_DIR = "D:/Bachelorarbeit/AlphaRuns"


def load_all_epoch_data(root_dir: str) -> pd.DataFrame:
    rows = []

    for run_dir in tqdm(os.listdir(root_dir), desc="Loading epoch data"):
        if not run_dir.startswith("run_"):
            continue

        config_path = os.path.join(root_dir, run_dir, "config.json")
        metrics_path = os.path.join(root_dir, run_dir, "metrics.json")

        if not os.path.exists(config_path) or not os.path.exists(metrics_path):
            continue

        with open(config_path, "r") as f:
            config = json.load(f)

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        model = config.get("model", "Unknown")
        model_name = config.get("model_name", run_dir)

        for m in metrics.values():
            rows.append({
                "run": run_dir,
                "model": model,
                "model_name": model_name,
                "epoch": m.get("epoch", -1),
                "train_loss": m.get("train_loss", float("nan")) if m.get("train_loss", float("nan")) < 1.3 else float("nan"),
                "val_loss": m.get("val_loss", float("nan")) if m.get("val_loss", float("nan")) < 1.3 else float("nan"),
                "train_accuracy": m.get("train_accuracy", float("nan")),
                "val_accuracy": m.get("val_accuracy", float("nan")),
                "learning_rate": m.get("learning_rate", float("nan"))
            })

    return pd.DataFrame(rows)


def load_runs(root_dir):
    data = []
    for run_dir in tqdm(os.listdir(root_dir), desc="Loading runs"):
        if not run_dir.startswith("run_"):
            continue
        run_path = os.path.join(root_dir, run_dir)
        config_path = os.path.join(run_path, "config.json")
        metrics_path = os.path.join(run_path, "metrics.json")

        if not os.path.isfile(config_path) or not os.path.isfile(metrics_path):
            continue

        with open(config_path, "r") as f:
            config = json.load(f)
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        model = config.get("model", "unknown")
        model_name = config.get("model_name", run_dir)
        model_archtecture = config.get("model_architecture", "unknown")

        # Get best epoch by highest val_accuracy
        # Find best epoch by highest val_accuracy
        best_epoch_data = max(
            (m for m in metrics.values() if "val_accuracy" in m),
            key=lambda x: x["val_accuracy"],
            default=None
)

        if best_epoch_data:
            data.append({
                "run": run_dir,
                "model": model,
                "model_architecture": model_archtecture,
                "model_name": model_name,
                "val_accuracy": best_epoch_data.get("val_accuracy", float("nan")),
                "train_accuracy": best_epoch_data.get("train_accuracy", float("nan")),
                "val_loss": best_epoch_data.get("val_loss", float("nan")),
                "train_loss": best_epoch_data.get("train_loss", float("nan")),
                "epoch": best_epoch_data.get("epoch", -1)
            })

    return pd.DataFrame(data)


def plot_metrics(df):
    subtitles = {"model": "Model Type", "model_architecture": "Model Architecture"}

    sns.set(style="whitegrid", font_scale=1.2)

    """
    # Plot validation accuracy by model
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="model", y="val_accuracy", palette="Set2")
    sns.stripplot(data=df, x="model", y="val_accuracy", color=".2", jitter=0.2, ax=plt.gca(), size=4) # Overlay stripplot
    #sns.swarmplot(data=df, x="model", y="val_accuracy", color=".25")
    plt.title("Best Validation Accuracy per Model")
    plt.ylabel("Validation Accuracy")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.show()

    # Plot validation loss by model
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="model", y="val_loss", palette="Set2")
    sns.swarmplot(data=df, x="model", y="val_loss", color=".25")
    plt.title("Best Validation Loss per Model")
    plt.ylabel("Validation Loss")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.show()
    """
    # Plot train vs val accuracy
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="train_accuracy", y="val_accuracy", hue="model_architecture", style="model", s=100)
    plt.title("Train vs Validation Accuracy")
    plt.xlabel("Train Accuracy")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.savefig('TrainVsValidationAccuracy.pdf', bbox_inches='tight', dpi=600)
    plt.show()


def plot_loss_over_epochs(df_epochs):
    sns.set(style="whitegrid", font_scale=1.2)

    losses = df_epochs.get("train_loss")
    for loss in losses:
        if loss > 1.0:
            print(loss)

    # Train loss plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_epochs, x="epoch", y="train_loss", hue="model", style="run", alpha=0.7)
    plt.title("Train Loss over Epochs per Model")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.ylim(0, 1.2)  # Adjust y-axis limits as needed
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    """
    # Validation loss plot
    #if df_epochs["val_loss"].notna().any():
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_epochs[df_epochs["val_loss"].notna()],
                 x="epoch", y="val_loss", hue="model", style="run", alpha=0.7)
    plt.title("Validation Loss over Epochs per Model")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    """


def plot_accuracy_over_epochs(df_epochs):
    sns.set(style="whitegrid", font_scale=1.2)

    # Train accuracy
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_epochs, x="epoch", y="train_accuracy", hue="model", style="run", alpha=0.7)
    plt.title("Train Accuracy over Epochs per Model")
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    """
    # Validation accuracy
    if df_epochs["val_accuracy"].notna().any():
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_epochs[df_epochs["val_accuracy"].notna()],
                     x="epoch", y="val_accuracy", hue="model", style="run", alpha=0.7)
        plt.title("Validation Accuracy over Epochs per Model")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy (%)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        """


def list_top_10_runs_sorted_by_accuracy(df):
    """
    List runs sorted by validation accuracy.
    """
    if df.empty:
        print("No runs available to sort.")
        return []

    ind = 0

    sorted_runs = df.sort_values(by="val_accuracy", ascending=False)
    for index, row in sorted_runs.iterrows():
        print(f"Run: {row['run']}, Model: {row['model']}, "
              f"Val Accuracy: {row['val_accuracy']:.4f}, "
              f"Train Accuracy: {row['train_accuracy']:.4f}, "
              f"Val Loss: {row['val_loss']:.4f}, "
              f"Train Loss: {row['train_loss']:.4f}, "
              f"Epoch: {row['epoch']}"
              f", Model Architecture: {row['model_architecture']}, "
              f"Model Params: {row['model_name']}")

        if ind >= 9:
            break
        ind += 1

    return sorted_runs


if __name__ == "__main__":
    df_runs = load_runs(ROOT_DIR)
    df_epochs = load_all_epoch_data(ROOT_DIR)

    if df_runs.empty:
        print("No valid run directories found.")
    else:
        print(f"Loaded {len(df_runs)} runs from {df_runs['model'].nunique()} models.")

    if df_epochs.empty:
        print("No valid epoch data found.")
    else:
        print(f"Loaded {len(df_epochs)} epochs from {df_epochs['model'].nunique()} models.")

    plot_metrics(df_runs)
    #plot_loss_over_epochs(df_epochs)
    plot_accuracy_over_epochs(df_epochs)
    sorted_runs = list_top_10_runs_sorted_by_accuracy(df_runs)
