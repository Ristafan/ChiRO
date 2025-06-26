import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

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
        model_architecture = config.get("model_architecture", "Unknown")


        # ["AlphaV2", "AlphaSelfAttention", "AlphaAttention", "AlphaSelfAttentionPositional", "AlphaV2_1D", "AlphaV2_1D_1", "AlphaSectionDynamic", "AlphaStandard"]
        if model_architecture in ["AlphaV2_1D", "AlphaV2_1D_1"]:
            continue
        #if model not in ["AlphaSectionDynamic"]:
        #    continue

        for m in metrics.values():
            rows.append({
                "run": run_dir,
                "model": model,
                "model_name": model_name,
                "model_architecture": m.get("model_architecture", model_architecture),
                "epoch": m.get("epoch", -1),
                "train_loss": m.get("train_loss", float("nan")) if m.get("train_loss", float("nan")) < 1.3 else float("nan"),
                "val_loss": m.get("val_loss", float("nan")) if m.get("val_loss", float("nan")) < 1.3 else float("nan"),
                "train_accuracy": m.get("train_accuracy", float("nan")),
                "val_accuracy": m.get("val_accuracy", float("nan")) * 100,
                "learning_rate": m.get("learning_rate", float("nan")),
                "good_sections": m.get("good_sections", 0),
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

        # ["AlphaV2", "AlphaSelfAttention", "AlphaAttention", "AlphaSelfAttentionPositional", "AlphaV2_1D", "AlphaV2_1D_1", "AlphaSectionDynamic", "AlphaStandard"]
        if model_archtecture in ["AlphaV2_1D", "AlphaV2_1D_1"]:
            continue
        #if model not in ["AlphaSectionDynamic"]:
        #    continue

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
                "val_accuracy": best_epoch_data.get("val_accuracy", float("nan")) * 100,
                "train_accuracy": best_epoch_data.get("train_accuracy", float("nan")),
                "val_loss": best_epoch_data.get("val_loss", float("nan")),
                "train_loss": best_epoch_data.get("train_loss", float("nan")),
                "epoch": best_epoch_data.get("epoch", -1)
            })

    return pd.DataFrame(data)


def plot_metrics(df):
    sns.set(style="whitegrid", font_scale=1.2)

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


def plot_merged_metrics(df):
    sns.set(style="whitegrid", font_scale=1.2)

    df_corrected = df.copy()

    # Define label mappings
    model_architecture_labels = {
        "AlphaV2": "Standard",
        "AlphaSelfAttention": "Self-Attention",
        "AlphaAttention": "Attention",
        "AlphaSelfAttentionPositional": "Self-Attention",
        "AlphaV2_1D": "Standard 1D",
        "AlphaSectionDynamic": "Standard",
        "AlphaStandard": "Standard"
    }

    model_labels = {
        "AlphaV2": "Standard",
        "AlphaSelfAttention": "Self-Attention",
        "AlphaAttention": "Attention",
        "AlphaSelfAttentionPositional": "Self-Attention",
        "AlphaV2_1D": "Standard 1D",
        "AlphaSectionDynamic": "Section Dynamic",
        "AlphaStandard": "Standard"
    }

    # Apply label mappings
    df_corrected["model_architecture_mapped"] = df_corrected["model_architecture"].map(model_architecture_labels)
    df_corrected["model_mapped"] = df_corrected["model"].map(model_labels)

    plt.figure(figsize=(10, 6))
    scatterplot = sns.scatterplot(
        data=df_corrected,
        x="train_accuracy",
        y="val_accuracy",
        style="model_architecture_mapped",
        hue="model_mapped",
        s=100,
        color=[0, 1, 2, 3]
    )
    font = {'size': 30}
    plt.rc('font', **font)
    plt.xlabel("Train Accuracy")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    plt.tight_layout()

    # Get the current legend handles and labels
    handles, labels = scatterplot.get_legend_handles_labels()

    # Find the index where the 'hue' legend starts and the 'style' legend starts
    # The first label is typically 'model_architecture_mapped' (hue title)
    # The next is 'model_mapped' (style title)
    hue_title_index = labels.index("model_architecture_mapped")
    style_title_index = labels.index("model_mapped")

    # Change the sublegend titles
    labels[hue_title_index] = "Model Architecture"
    labels[style_title_index] = "Training Method"

    # Recreate the legend with updated titles
    plt.legend(handles=handles, labels=labels, loc='lower right')

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
    model_labels = {
        "AlphaV2": "Standard 2D",
        "AlphaSelfAttention": "Self-Attention",
        "AlphaAttention": "Attention",
        "AlphaSelfAttentionPositional": "Self-Attention",
        "AlphaSectionDynamic": "Section Dynamic",
        "AlphaStandard": "Standard"
    }
    df_corrected = df_epochs.copy()
    df_corrected["model_mapped"] = df_corrected["model"].map(model_labels)

    sns.set(style="whitegrid", font_scale=1.2)

    # Train accuracy
    run_counts = df_corrected.groupby('model_mapped')['run'].nunique()

    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(
        data=df_corrected,
        x="epoch",
        y="train_accuracy",
        hue="model",
        style="run",
        alpha=0.7,
        legend=False  # Disable the automatic legend
    )
    font = {'size': 30}
    plt.rc('font', **font)
    sns.lineplot(data=df_corrected, x="epoch", y="train_accuracy", hue="model_mapped", style="run", alpha=0.7, color=[0, 1, 2, 3])
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy")
    plt.xlim(1, df_corrected["epoch"].max() + 1)
    handles, labels = ax.get_legend_handles_labels()

    hue_labels = []
    hue_handles = []
    # We need to find the correct handles and labels. The first entry is the hue title,
    # and the rest are the items. We can iterate through the unique models to find them.
    for model in df_corrected['model_mapped'].unique():
        # Find the index of the model in the labels list
        try:
            label_index = labels.index(model)
            # Add the handle and the modified label
            hue_handles.append(handles[label_index])
            hue_labels.append(f"{model} ({run_counts[model]} runs)")
        except ValueError:
            pass

    # Manually create the legend with only the 'hue' handles and labels
    plt.legend(hue_handles, hue_labels, title="Training Method", loc='center right')
    plt.savefig('TrainAccuracyOverEpochs.pdf', bbox_inches='tight', dpi=600)
    plt.show()

    # Validation accuracy
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(
        data=df_corrected,
        x="epoch",
        y="val_accuracy",
        hue="model",
        style="run",
        alpha=0.7,
        legend=False  # Disable the automatic legend
    )
    font = {'size': 30}
    plt.rc('font', **font)
    sns.lineplot(data=df_corrected, x="epoch", y="val_accuracy", hue="model_mapped", style="run", alpha=0.7, color=[0, 1, 2, 3])
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.xlim(1, df_corrected["epoch"].max() + 1)
    handles, labels = ax.get_legend_handles_labels()

    hue_labels = []
    hue_handles = []
    # We need to find the correct handles and labels. The first entry is the hue title,
    # and the rest are the items. We can iterate through the unique models to find them.
    for model in df_corrected['model_mapped'].unique():
        # Find the index of the model in the labels list
        try:
            label_index = labels.index(model)
            # Add the handle and the modified label
            hue_handles.append(handles[label_index])
            hue_labels.append(f"{model} ({run_counts[model]} runs)")
        except ValueError:
            pass

    # Manually create the legend with only the 'hue' handles and labels
    plt.legend(hue_handles, hue_labels, title="Training Method", loc='center right')
    plt.savefig('ValidationAccuracyOverEpochs.pdf', bbox_inches='tight', dpi=600)
    plt.show()


def plot_good_sections_over_epochs(df_epochs):
    df_corrected = df_epochs.copy()
    sns.set(style="whitegrid", font_scale=1.2)

    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(
        data=df_corrected,
        x="epoch",
        y="good_sections",
        hue="model_architecture",
        style="run",
        alpha=0.7,
        legend=False  # Disable the automatic legend
    )

    sns.lineplot(data=df_corrected, x="epoch", y="good_sections", hue="model_architecture", style="run", alpha=0.7)
    plt.title("Train Accuracy over Epochs per Model")
    plt.xlabel("Epoch")
    plt.ylabel("TNumber of good sections")
    plt.xlim(1, df_corrected["epoch"].max() + 1)
    handles, labels = ax.get_legend_handles_labels()
    # Only select the unique model architectures
    unique_models = df_corrected['model_architecture'].unique()
    hue_labels = []
    hue_handles = []
    # We need to find the correct handles and labels. The first entry is the hue title,
    # and the rest are the items. We can iterate through the unique models to find them.
    for model in unique_models:
        # Find the index of the model in the labels list
        try:
            label_index = labels.index(model)
            # Add the handle and the modified label
            hue_handles.append(handles[label_index])
            hue_labels.append(model)
        except ValueError:
            pass

    # Manually create the legend with only the 'hue' handles and labels
    plt.legend(handles, hue_labels, title="Method", loc='upper right')
    plt.show()


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
    # Set this to the root directory containing run_* folders
    ROOT_DIR = "D:/Bachelorarbeit/AlphaRuns"

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

    #plot_metrics(df_runs)
    plot_merged_metrics(df_runs)
    #plot_loss_over_epochs(df_epochs)
    plot_accuracy_over_epochs(df_epochs)
    #plot_good_sections_over_epochs(df_epochs)
    #sorted_runs = list_top_10_runs_sorted_by_accuracy(df_runs)
