"""Quick prediction test — feed a handwritten digit image to the trained MNIST model."""

import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

from src.model_packager import MNISTNet


def predict(image_path: str) -> None:
    """Load a digit image and print the model's prediction."""
    model = MNISTNet()
    model.load_state_dict(torch.load("mnist_model.pt", map_location="cpu", weights_only=True))
    model.eval()

    img = Image.open(image_path).convert("L").resize((28, 28))
    tensor = transforms.ToTensor()(img)  # shape [1, 28, 28], values 0-1

    # MNIST uses white digit on black background.
    # If your image is dark digit on light background, invert it.
    if tensor.mean() > 0.5:
        tensor = 1.0 - tensor

    with torch.no_grad():
        output = model(tensor.unsqueeze(0))  # [1, 1, 28, 28]
        probabilities = torch.softmax(output, dim=1)
        prediction = probabilities.argmax(dim=1).item()
        confidence = probabilities.max().item()

    print(f"Predicted: {prediction} (confidence: {confidence:.1%})")
    print(f"All probabilities: {[f'{p:.3f}' for p in probabilities.squeeze().tolist()]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run test_predict.py <image_path>")
        print("  Image should be a digit (any size, will be resized to 28x28)")
        sys.exit(1)
    predict(sys.argv[1])
