import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from model import MNISTClassifier

def parse_arguments():
    parser = argparse.ArgumentParser(description="MNIST Classifier Prediction")
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image for prediction",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../models/best_mnist_mlp.pth",
        help="Path to the trained model checkpoint",
    )
    
    return parser.parse_args()

