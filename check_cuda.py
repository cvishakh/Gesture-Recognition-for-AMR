import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available! GPU is being used.")
    print("Current device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Running on CPU.")