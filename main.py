import torch
import sys
import os

if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda.is_available())
