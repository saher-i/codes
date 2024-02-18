import torch
import torch.nn as nn
import torch.optim as optim
from c1 import ResNet18
# Create an instance of ResNet-18
resnet18 = ResNet18()

# Count the number of trainable parameters
trainable_params = sum(p.numel() for p in resnet18.parameters() if p.requires_grad)
print("Number of trainable parameters in ResNet-18: ", trainable_params)

# Define a loss function and the SGD optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)

# Count the number of gradients
resnet18.zero_grad()
inputs = torch.randn(1, 3, 224, 224)  # Example input
outputs = resnet18(inputs)
outputs.backward(torch.randn(1, 1000))  # Example gradient

total_gradients = 0
for p in resnet18.parameters():
    if p.grad is not None:
        total_gradients += p.grad.numel()
print("Number of gradients in ResNet-18: ", total_gradients)
