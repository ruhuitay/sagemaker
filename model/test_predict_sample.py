"""Predict a random sample from the MNIST test set to verify the model works."""

import random

import torch
import torchvision.transforms as transforms
from torchvision import datasets

from src.model_packager import MNISTNet


def main():
    model = MNISTNet()
    model.load_state_dict(torch.load("mnist_model.pt", map_location="cpu", weights_only=True))
    model.eval()

    test_data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())

    idx = random.randint(0, len(test_data) - 1)
    image, label = test_data[idx]

    with torch.no_grad():
        output = model(image.unsqueeze(0))
        probabilities = torch.softmax(output, dim=1)
        prediction = probabilities.argmax(dim=1).item()
        confidence = probabilities.max().item()

    print(f"Sample index: {idx}")
    print(f"Actual label: {label}")
    print(f"Predicted:    {prediction} (confidence: {confidence:.1%})")
    print(f"{'✓ Correct!' if prediction == label else '✗ Wrong!'}")


if __name__ == "__main__":
    main()
