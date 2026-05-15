import sys
import torch
import torchvision
import sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import wandb
import cv2

def main():
    """
    Test script to verify that the core dependencies for the project
    are installed and accessible within the current Python environment.
    """
    print("=" * 40)
    print("Project Odysseus: Environment Check")
    print("=" * 40)

    print(f"{'Python Version:':<20} {sys.version.split(' ')[0]}")
    
    print("-" * 40)
    print("Machine Learning & Data Science Libraries:")
    print("-" * 40)
    print(f"{'PyTorch:':<20} {torch.__version__}")
    print(f"{'Torchvision:':<20} {torchvision.__version__}")
    print(f"{'Scikit-learn:':<20} {sklearn.__version__}")
    print(f"{'Pandas:':<20} {pd.__version__}")
    print(f"{'NumPy:':<20} {np.__version__}")
    
    print("-" * 40)
    print("Visualization Libraries:")
    print("-" * 40)
    print(f"{'Matplotlib:':<20} {matplotlib.__version__}")
    print(f"{'Seaborn:':<20} {sns.__version__}")
    
    print("-" * 40)
    print("Utilities & Tracking:")
    print("-" * 40)
    print(f"{'WandB:':<20} {wandb.__version__}")
    print(f"{'OpenCV (cv2):':<20} {cv2.__version__}")

    print("=" * 40)
    print("All core imports successful!")
    print("=" * 40)

if __name__ == "__main__":
    main()