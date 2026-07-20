import torch
import torch.nn as nn
from torchvision import datasets, transforms

from src.model_packager import MNISTNet

model = MNISTNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

train_data = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

model.train()
for images, labels in loader:
    optimizer.zero_grad()
    loss_fn(model(images), labels).backward()
    optimizer.step()

torch.save(model.state_dict(), "mnist_model.pt")
