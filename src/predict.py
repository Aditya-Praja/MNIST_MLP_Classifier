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

def get_device():
    return (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

def preprocess_image(image_path):
    
    image = Image.open(image_path).convert("L")
    
    pixel_values = list(image.getdata())
    average_pixel_value = sum(pixel_values) / len(pixel_values)
    
    if average_pixel_value > 127:
        image = ImageOps.invert(image)
        
    bounding_box = image.getbbox()
    
    if bounding_box is None:
        raise ValueError("The image is completely white or black.")
    
    image = image.crop(bounding_box)
    
    original_width, original_height = image.size
    
    maximum_digit_size = 20
    
    scale = min(
        maximum_digit_size / original_width,
        maximum_digit_size / original_height,
    )
    
    resized_width = max(1, int(original_width * scale))
    resized_height = max(1, int(original_height * scale))
    
    image = image.resize(
        (resized_width, resized_height), 
        Image.RESAMPLE_LANCZOS,
    )
    
    canvas = Image.new("L", (28, 28), color=0)
    
    left = (28 - resized_width) // 2
    top = (28 - resized_height) // 2
    
    canvas.paste(image, (left, top))
    
    image_tensor = transforms.ToTensor()(canvas).unsqueeze(0)
    
    return image_tensor


def load_model(checkpoint_path, device):
    
    checkpoint = torch.load(
        checkpoint_path, 
        map_location=device,
        weights_only=True,
    )
    
    dropout_probability = checkpoint.get("dropout_probability", 0.2)
    
    model = MNISTClassifier(
        dropout_probability=dropout_probability
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    
    return model

def predict(
    model,
    image_tensor,
    device,
):
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
    
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities.max(dim=1).values.item()
        
    return predicted_class, confidence, probabilities[0]

def main(args):
    
    if not Path(args.image_path).exists():
        raise FileNotFoundError(
            f"Image file not found: {args.image_path}"
        )
        
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {args.checkpoint}"
        )
        
    device = get_device()
    print(f"Using device: {device}")
    
    image_tensor = preprocess_image(args.image_path)
    
    model = load_model(args.checkpoint, device)
    
    predicted_class, confidence, probabilities = predict(
        model,
        image_tensor,
        device,
    )
    
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    
    print("Class probabilities:")
    
    for digit, prob in enumerate(probabilities):
        print(f"Digit {digit}: {prob.item():.4f}")
        
if __name__ == "__main__":
    args = parse_arguments()
    main(args)
        
     