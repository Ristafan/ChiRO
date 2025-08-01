import glob
import os

import numpy as np
import torch

from torch.utils.data import DataLoader

from src.Architectures.BinaryClassification.AlphaV2_1D import AlphaV2_1D
from src.Architectures.BinaryClassification.AlphaV2_1D_1 import AlphaV2_1D_1
from src.Architectures.GenusClassification.BetaV3 import BetaV3
from src.DataSetSplit.TrainingClasses import eptesicus_species, myotis_species, nyctalus_species, pipistrellus_species, \
    Chiroptera_generally, bat_species_fixed
from src.Preprocessing.Preprocessor import Preprocessor
from src.Testing.ConfusionMatrix import plot_confusion_matrix
from src.Training.Train_BSectionDynamic import collate_fn
from src.Training.TrainingParams import TrainingParams
from src.utils import load_path_config


def evaluate_model_and_plot_confusion_matrix(model, model_name, model_path, label_names, test_loader=None, *args, **kwargs):
    # Determine the device to run the model on (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model with the same parameters used during training.
    # Here, batch_norm is set to False as per the user's training code snippet.
    if model == BetaV3:
        model = model(batch_norm=True, num_genera=6).to(device)
    #elif model == AlphaSelfAttentionPositionalNet:
    #    model_params = kwargs.get('model_params', {})
    #    attention_heads = model_params.get('attention_heads', 4)
    #    batch_norm = model_params.get('batch_norm', True)
    #    dropout_rate = model_params.get('dropout_rate', 0.1)
    #    final_pooling = model_params.get('final_pooling', 'avg')
    #    max_sequence_length = model_params.get('max_sequence_length', 750)
    #    use_learned_positional_encoding = model_params.get('use_learned_positional_encoding', False)
#
    #    model = model(attention_heads=attention_heads, batch_norm=batch_norm,
    #                  dropout_rate=dropout_rate, final_pooling=final_pooling,
    #                  max_sequence_length=max_sequence_length,
    #                  use_learned_positional_encoding=use_learned_positional_encoding).to(device)
    else:
        model = model(batch_norm=True).to(device)

    # Load the trained model weights.
    # Ensure the 'model_path' points to your actual saved model file.
    # Use map_location to ensure it loads correctly regardless of original device
    model.load_state_dict(torch.load(model_path), strict=False)
    print(f"Model weights loaded successfully from {model_path}")

    # Set the model to evaluation mode. This disables dropout and batch normalization
    # updates, ensuring consistent predictions.
    model.eval()

    all_predictions = []      # To store all predicted labels
    all_actual_labels = []    # To store all true labels

    # Check if a test_loader was provided
    if test_loader is None:
        print("Error: No test_loader provided. Cannot perform evaluation.")
        print("Please provide a DataLoader instance containing your test dataset.")
        return # Exit the function if no loader is provided

    print(f"Starting evaluation with {len(test_loader.dataset)} samples from the provided test_loader.")

    # Disable gradient calculations during evaluation to save memory and speed up inference.
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            # Move data to the appropriate device (CPU or GPU)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Perform a forward pass to get model outputs (logits)
            outputs = model(inputs)
            # Get the predicted class by finding the index with the maximum logit value
            _, predicted = torch.max(outputs.data, 1)

            # Extend the lists with predictions and true labels, converting them to NumPy arrays
            # and moving them to CPU first if they are on GPU.
            all_predictions.extend(predicted.cpu().numpy())
            all_actual_labels.extend(labels.cpu().numpy())

    # Convert the collected lists to NumPy arrays for confusion matrix calculation
    final_predictions = np.array(all_predictions)
    final_actual_labels = np.array(all_actual_labels)

    # Plot the unnormalized confusion matrix
    print("\nGenerating Unnormalized Confusion Matrix:")
    plot_confusion_matrix(final_predictions, final_actual_labels, label_names, model_name,
                          normalize=False, title='Confusion Matrix')

    # Plot the normalized confusion matrix
    print("\nGenerating Normalized Confusion Matrix:")
    plot_confusion_matrix(final_predictions, final_actual_labels, label_names, model_name,
                          normalize=True, title='Normalized Confusion Matrix')

    # Store results in a json file
    results = {
        "model_name": model_name,
        "predictions": final_predictions.tolist(),
        "actual_labels": final_actual_labels.tolist(),
        "label_names": label_names,
    }
    results_file = os.path.join(f"C:/Users/MartinFaehnrich/Documents/ChiRO/src/Testing/{model_name}_results.json")
    with open(results_file, 'w') as f:
        import json
        json.dump(results, f, indent=4)


def parse_model_name(model_name):
    base, *parts = model_name.split("_")
    if len(parts) != 13:
        raise ValueError(f"Unexpected format with {len(parts)} parts: {model_name}")

    def parse_float(s):
        return float(s.replace("-", "."))

    return {
        "optimizer": parts[0],
        "batch_size": int(parts[1]),
        "num_epochs": int(parts[2]),
        "learning_rate": parse_float(parts[3]),
        "dropout_rate": parse_float(parts[4]),
        "loss_filter_threshold_percentage": parse_float(parts[5]),
        "window_size": parse_float(parts[6]),
        "overlap_size": parse_float(parts[7]),
        "attention_heads": int(parts[8]),
        "batch_norm": parts[9] == "True",
        "early_stopping": parts[10] == "True",
        "patience": int(parts[11]),
        "unknown_or_suffix": parts[12],  # you can discard or rename as needed
    }


if __name__ == "__main__":
    config = load_path_config()
    alpha = True
    beta = False

    if alpha:
        test_files_and_labels_path = config['dataset']['test_files_and_labels_path_alpha']
        original_files_and_labels_path = config['dataset']['original_files_and_labels_path']
        root_files_path = config['dataset']['files_path_root']
        spectrograms_path = config['spectrogram']['spectrograms_dir']
        model_path = config['model']['alpha']
        runs_dir_beta = config['logs']['runs_dir_beta']

        training_params = TrainingParams()
        training_params.device = "cpu"
        training_params.model = "AlphaSectionDynamic"
        training_params.merge_labels = [bat_species_fixed]
        labels = ['Bat Call', 'Env Sounds']

        # Load Audio Files, Labels and create spectrograms
        preprocessor = Preprocessor(test_files_and_labels_path, spectrograms_path, root_files_path)

        #preprocessor.create_data_splits(original_files_and_labels_path, training_params.merge_labels, training_params.ignored_labels, training_params.seed, training_params.total_files_per_class, training_params.use_min_files_per_class, training_params.split_method, training_params.split_ratios)
        #preprocessor.create_spectrograms_stft(test_files_and_labels_path)

        test_dataset = preprocessor.create_bat_file_dataset(test_files_and_labels_path)
        test_loader = DataLoader(test_dataset, batch_size=training_params.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=1, pin_memory=True)

        # BIG EVALUATION
        model_params = {}
        all_models = glob.glob('C:/Users/MartinFaehnrich/Documents/ChiRO/src/Testing/ModelsToEvaluate/*.pth')
        basename_models = [os.path.basename(model)[:-4] for model in all_models]
        #all_models = ["D:/Bachelorarbeit/models/AlphaStandard_SGD_2_2_0.001_0.4_0.85_0.23_0.12_2_True_True_6.pth"]

        # This was evaluated last: AlphaAttention_Adam_4_6_0-0001_0-0_0-85_0-2_0-1_1_True_False_5

        for model in all_models:
            model_name = os.path.basename(model)#[:-4]
            print(f"Evaluating model: {model_name}")
            # Dynamically load the model class based on its name
            #if 'AlphaV2_1D_1' in model_name:
            #    model_class = AlphaV2_1D_1
            #elif 'AlphaV2_1D' in model_name:
            #    model_class = AlphaV2_1D
            #elif 'AlphaV2' in model_name:
            #    model_class = AlphaV2
            #elif 'AlphaStandard' in model_name:
            #    model_class = AlphaV2
            #elif 'AlphaAttention' in model_name:
            #    model_class = AlphaV1_Attention
            #elif 'AlphaSelfAttentionPositional' in model_name:
            #    model_class = AlphaSelfAttentionPositionalNet
            #elif 'AlphaSelfAttention' in model_name:
            #    model_class = SelfAttentionNet
            #else:
            #    print(f"Unknown model architecture for {model_name}, skipping.")
            #    continue
            model_class = AlphaV2_1D
            print(model_class)

            model_path = model  # Use the full path to the model file
            # Call the evaluation function
            try:
                evaluate_model_and_plot_confusion_matrix(model_class, model_name, model_path, labels, test_loader, kwargs=model_params)
            except Exception as e:
                print(30* "-")
                print(f"ERROR evaluating model {model_name}: {e}")
                print(30* "-")

            try:
                model_class = AlphaV2_1D_1
                evaluate_model_and_plot_confusion_matrix(model_class, model_name, model_path, labels, test_loader, kwargs=model_params)
            except Exception as e:
                print(30* "-")
                print(f"ERROR evaluating model {model_name}: {e}")
                print(30* "-")

        #model = AlphaV2

        # Path to the trained model weights
        #model_path = 'D:/Bachelorarbeit/models/AlphaStandard_Adam_1_8_0_001_0_1_0_75_0_27_0_13_2_True_False_3.pth'

        # Call the evaluation function
        #evaluate_model_and_plot_confusion_matrix(model, model_name, model_path, labels, test_loader)

    if beta:
        test_files_and_labels_path = config['dataset']['test_files_and_labels_path_beta']
        original_files_and_labels_path = config['dataset']['original_files_and_labels_path']
        root_files_path = config['dataset']['files_path_root']
        spectrograms_path = config['spectrogram']['spectrograms_dir']
        model_path = config['model']['beta']
        runs_dir_beta = config['logs']['runs_dir_beta']

        training_params = TrainingParams()
        training_params.model = "BetaSectionDynamic"
        training_params.dataset_name = "GenusBatCalls"
        training_params.ignored_labels = ["Env_sounds"]
        training_params.merge_labels = [eptesicus_species, myotis_species, nyctalus_species, pipistrellus_species, Chiroptera_generally]
        training_params.num_classes = 6
        labels = ['Chiroptera', 'Eptesicus', 'Myotis', 'Nyctalus', 'Pipistrellus', 'Vespertilio']

        # Load Audio Files, Labels and create spectrograms
        preprocessor = Preprocessor(test_files_and_labels_path, spectrograms_path, root_files_path)

        #preprocessor.create_data_splits(original_files_and_labels_path, training_params.merge_labels, training_params.ignored_labels, training_params.seed, training_params.total_files_per_class, training_params.use_min_files_per_class, training_params.split_method, training_params.split_ratios)
        #preprocessor.create_spectrograms_stft(test_files_and_labels_path)

        test_dataset = preprocessor.create_bat_file_dataset(test_files_and_labels_path)
        test_loader = DataLoader(test_dataset, batch_size=training_params.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=1, pin_memory=True)

        model = BetaV3

        # Path to the trained model weights
        model_path = 'D:/Bachelorarbeit/models/BetaSectionDynamic-PretrainedAlpha_BetaV3_Adam_6_10_1e-05_0_0_0_7_0_23_0_12_1_True_True_2.pth'

        # Call the evaluation function
        evaluate_model_and_plot_confusion_matrix(model, model_path, labels, test_loader)

