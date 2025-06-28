import torch
import onnx
import os # For managing file paths

from src.Architectures.AlphaV2 import AlphaV2

# 1. Initialize your model
model = AlphaV2()
model.eval() # Set the model to evaluation mode for consistent export

# 2. Create a dummy input tensor
# This input should match the expected input shape of your model
dummy_input = torch.zeros([1, 1, 64, 64])

# 3. Define the ONNX output path
onnx_file_path = "alpha_v2.onnx"

print(f"Exporting model to {onnx_file_path}...")


# 4. Export the model to ONNX format
# The 'opset_version' is important for compatibility.
# A common opset version like 11, 12, or 13 is usually good.
# 'input_names' and 'output_names' are optional but good practice for clarity.
# 'dynamic_axes' is crucial if your batch size can vary, or if input dimensions are dynamic.
torch.onnx.export(
    model,
    dummy_input,
    onnx_file_path,
    verbose=False, # Set to True for more detailed output during export
    opset_version=11, # A common and well-supported ONNX opset version
    input_names=['input'], # Name for the input tensor
    output_names=['output'], # Name for the output tensor
    dynamic_axes={
        'input': {0: 'batch_size'}, # Allow variable batch size
        'output': {0: 'batch_size'} # Allow variable batch size for output
    }
)
print(f"Model successfully exported to {onnx_file_path}")

# Instructions for Netron
print("\n--- Next Steps ---")
print("To visualize this ONNX model, it is highly recommended to use Netron.")
print("1. Download Netron from: https://github.com/lutzroeder/netron/releases")
print("2. Install and open Netron.")
print(f"3. Open the file '{os.path.abspath(onnx_file_path)}' in Netron.")
print("Netron provides an interactive visualization of the model's graph, layers, and operations.")



