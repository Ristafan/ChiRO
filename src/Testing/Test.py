import os

import numpy as np
import torch
from src.Architectures.AlphaV2 import AlphaV2
from torch.utils.data import DataLoader
from src.Preprocessing.BatFileDataSet import BatFileDataSet
from src.Preprocessing.Preprocessor import Preprocessor
from src.Testing.ConfusionMatrix import plot_confusion_matrix


def evaluate_model_and_plot_confusion_matrix(model, model_path, test_loader=None):
    """
    Evaluates the AlphaV2 model by loading its weights, running inference on a
    provided dataset (via a DataLoader), and then plotting a confusion matrix
    based on the results.

    Args:
        model (callable): A callable that returns an instance of the model to be evaluated.
        model_path (str): Path to the saved model state dictionary (.pth file).
                          Defaults to 'trained_alpha_v2.pth'.
        test_loader (torch.utils.data.DataLoader): A DataLoader containing the
                                                    test dataset. This is expected
                                                    to yield (inputs, labels).
    """
    # Determine the device to run the model on (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model with the same parameters used during training.
    # Here, batch_norm is set to False as per the user's training code snippet.
    model = model(batch_norm=False).to(device)

    # Load the trained model weights.
    # Ensure the 'model_path' points to your actual saved model file.
    if os.path.exists(model_path):
        # Use map_location to ensure it loads correctly regardless of original device
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model weights loaded successfully from {model_path}")
    else:
        print(f"Warning: Model weights not found at '{model_path}'. "
              "Using randomly initialized weights for demonstration. "
              "Please ensure you have a trained model saved at this path "
              "for meaningful evaluation results.")
        # If no model exists, the script will still run, but predictions will be random.

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

    # Define the names for your classes. Make sure these match your label encoding.
    # Assuming 0: Noise, 1: Bat Call based on '2 classes: bat call or noise'
    label_names = ['Noise', 'Bat Call']

    # Plot the unnormalized confusion matrix
    print("\nGenerating Unnormalized Confusion Matrix:")
    plot_confusion_matrix(final_predictions, final_actual_labels, label_names,
                          normalize=False, title='Confusion Matrix (AlphaV2)')

    # Plot the normalized confusion matrix
    print("\nGenerating Normalized Confusion Matrix:")
    plot_confusion_matrix(final_predictions, final_actual_labels, label_names,
                          normalize=True, title='Normalized Confusion Matrix (AlphaV2)')


if __name__ == "__main__":
    # Load your test dataset (replace with your actual dataset path)
    preprocessor = Preprocessor("C:/Users/MartinFaehnrich/Documents/ChiRO/data/Alpha/test_dataset_info.xlsx",
                                "C:/Users/MartinFaehnrich/Documents/ChiRO/data/spectrograms",
                                "D:/Bachelorarbeit/AgroscopeData/LabelledSequences")
    num_classes = preprocessor.create_data_splits("D:/Bachelorarbeit/AgroscopeData/LabelledSequencesMerged_cleaned_cleaned_cleaned_cleaned.xlsx")
    #preprocessor.create_spectrograms_stft()
    test_dataset = preprocessor.create_bat_file_dataset()
    test_loader = DataLoader(test_dataset, batch_size=2,
                              shuffle=True, num_workers=2)

    # Path to the trained model weights
    model_path = 'C:/Users/MartinFaehnrich/Documents/ChiRO/src/Models/Alpha/alphaMIL_19-40-33.pth'

    # Call the evaluation function
    evaluate_model_and_plot_confusion_matrix(AlphaV2, model_path, test_loader)
