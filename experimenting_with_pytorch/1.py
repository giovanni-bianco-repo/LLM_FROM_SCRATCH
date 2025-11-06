import torch
import numpy as np
num_gpus = torch.cuda.device_count() #0
print(num_gpus)


## import torch

print("CUDA available:", torch.cuda.is_available())  # Always False on Mac
print("MPS available:", torch.backends.mps.is_available())  # True on M1/M2/M3 Macs

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
x = torch.rand(3, 3).to(device)
print(x.device)

