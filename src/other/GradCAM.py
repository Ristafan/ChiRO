import torch
import torch.nn.functional as F
import numpy as np
import cv2 # For image processing (resizing, overlaying)
from PIL import Image # For opening/saving images (spectrograms)
from src.Architectures.AlphaV2 import AlphaV2  # Import your model architecture
from src.Preprocessing.AudioLoader import AudioLoader
from src.Preprocessing.SpectrogramProcessor import SpectrogramProcessor


def apply_grad_cam(model, input_spectrogram, target_layer, target_class_idx):
    """
    Applies Grad-CAM to a given model and input.

    Args:
        model (torch.nn.Module): Your trained CNN model.
        input_spectrogram (torch.Tensor): The input spectrogram (e.g., [1, C, H, W]).
        target_layer (torch.nn.Module): The convolutional layer to extract features from.
        target_class_idx (int): The index of the class you want to visualize (0 for 'no bat call', 1 for 'bat call').

    Returns:
        np.ndarray: The Grad-CAM heatmap, ready for overlaying.
    """
    model.eval() # Set model to evaluation mode

    # Store gradients and activations
    activations = None
    gradients = None

    def save_activation(module, input, output):
        nonlocal activations
        activations = output

    def save_gradient(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    # Register hooks
    # Register hook to save activations of the target layer
    hook_handle_activation = target_layer.register_forward_hook(save_activation)
    # *** CHANGE THIS LINE ***
    hook_handle_gradient = target_layer.register_full_backward_hook(save_gradient)

    # Forward pass
    output = model(input_spectrogram)

    print("Model output (logits):", output)
    predicted_class = torch.argmax(output).item()
    print(f"Model predicted class: {predicted_class}")
    print(f"Target class for Grad-CAM: {target_class_idx}")


    # Get the predicted class probability for the target class
    one_hot_output = torch.zeros_like(output)
    one_hot_output[0][target_class_idx] = 1 # For a single input

    # Zero gradients, then backward pass to get gradients of the target class with respect to the activations
    model.zero_grad()
    output.backward(gradient=one_hot_output, retain_graph=True) # retain_graph=True if you need to do multiple backward passes

    # Remove hooks
    hook_handle_activation.remove()
    hook_handle_gradient.remove()

    print("Activations shape:", output.shape)
    print("Activations min/max:", output.min(), output.max())

    # Get the gradients and activations
    gradients = gradients.cpu().data.numpy()[0] # [C, H_feature, W_feature]
    activations = activations.cpu().data.numpy()[0] # [C, H_feature, W_feature]

    # Compute neuron importance weights (global average pooling of gradients)
    weights = np.mean(gradients, axis=(1, 2)) # [C]

    # Compute the Grad-CAM heatmap
    heatmap = np.zeros(activations.shape[1:], dtype=np.float32) # [H_feature, W_feature]

    for i, w in enumerate(weights):
        heatmap += w * activations[i]

    # Apply ReLU to the heatmap
    heatmap = np.maximum(heatmap, 0)

    # Normalize the heatmap to be between 0 and 1
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)

    # Resize the heatmap to the original spectrogram size
    heatmap = cv2.resize(heatmap, (input_spectrogram.shape[3], input_spectrogram.shape[2])) # W, H

    return heatmap


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}") # Good for debugging

    waveform, sample_rate = AudioLoader().load_wav_file('C:/Users/MartinFaehnrich/Documents/ChiRO/data/ExampleData/train/20220630_221300T #0002_645fbdba0181f0367e9570e949180e4b.wav')

    s = SpectrogramProcessor(waveform)
    s.apply_highpass_filter()
    s.compute_spectrogram()
    print(s.spectrogram.shape)
    s.denoise_spectrogram_mean_subtraction()
    s.scale_to_db()

    spectrogram = s.spectrogram.unsqueeze(0).to(device)  # Add batch dimension and move to device
    target_class_index = 0

    model = AlphaV2()  # Load your model
    model_path = 'D:/Bachelorarbeit/models/AlphaSectionDynamic_AlphaV2_Adam_2_10_0_001_0_0_0_75_1_0_0_3_1_True_True_2.pth'
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    target_layer = model.conv1  # Assuming this is the layer you want to visualize
    heatmap = apply_grad_cam(model, spectrogram, target_layer, target_class_index)

    # --- 3. Visualize the Heatmap and Overlay ---

    # Normalize the original spectrogram to 0-255 for display
    # Spectrograms in dB often have a wide range. We need to map it to 0-255.
    # It's crucial to handle min/max for proper visualization.
    # If your min_db and max_db are known, use those for consistent scaling.
    min_db = spectrogram.min() # Or a fixed value like -80 dB
    max_db = spectrogram.max() # Or a fixed value like -20 dB (if appropriate)

    scaled_spectrogram_temp = None
    # Avoid division by zero or errors if max and min are the same
    if max_db == min_db:
        # If all values are the same, just create a uniform grayscale image
        scaled_spectrogram_temp = np.zeros_like(spectrogram, dtype=np.uint8)
    else:
        scaled_spectrogram_temp = (spectrogram - min_db) / (max_db - min_db)
        scaled_spectrogram_temp = scaled_spectrogram_temp.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU
        spectrogram = (scaled_spectrogram_temp * 255).astype(np.uint8)

    # Convert original spectrogram to 3 channels (BGR) for color overlay
    original_spectrogram_colored = cv2.cvtColor(spectrogram, cv2.COLOR_GRAY2BGR)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    alpha = 0.5
    overlaid_img = cv2.addWeighted(original_spectrogram_colored, 1 - alpha, heatmap_colored, alpha, 0)

    # --- NEW: Apply stretching to both the original and overlaid images ---
    stretch_factor_x = 10 # Stretch factor for the time (width) dimension
    stretch_factor_y = 1  # No stretching in the frequency (height) dimension

    # Get original dimensions (height, width)
    original_height, original_width = original_spectrogram_colored.shape[:2]

    # Calculate new dimensions
    new_width = int(original_width * stretch_factor_x)
    new_height = int(original_height * stretch_factor_y) # Should be same as original height if factor is 1


    # Resize (stretch) the images using cv2.resize
    # INTER_LINEAR is a good default for upscaling/stretching
    stretched_original_spectrogram = cv2.resize(original_spectrogram_colored, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    stretched_heatmap = cv2.resize(heatmap_colored, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    stretched_overlaid_img = cv2.resize(overlaid_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Save the images
    output_dir = "grad_cam_output"
    import os
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "original_spectrogram.png"), stretched_original_spectrogram)
    cv2.imwrite(os.path.join(output_dir, "heatmap_raw.png"), stretched_heatmap)
    cv2.imwrite(os.path.join(output_dir, "grad_cam_overlaid.png"), stretched_overlaid_img)
    print(f"Images saved to: {os.path.abspath(output_dir)}")