import torch
import torch.nn as nn
from model import ResNet18

# Assuming ResNet18() is the ResNet-18 model as previously defined

model = ResNet18()

# Count the number of trainable parameters
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {num_trainable_params}')

# Assuming the model has been connected to an optimizer and a backward pass has been made
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# Example input for a forward pass (assuming CIFAR-10 images, with shape 32x32x3)
example_input = torch.randn(1, 3, 32, 32)  # Batch size of 1

# Forward pass
output = model(example_input)

# Example target for computing loss (assuming CIFAR-10, with 10 classes)
target = torch.tensor([1], dtype=torch.long)  # Example target class

# Loss function
criterion = nn.CrossEntropyLoss()

# Compute loss
loss = criterion(output, target)

# Backward pass to compute gradients
loss.backward()

# Check if gradients exist and count them
num_gradients = sum(p.grad.numel() for p in model.parameters() if p.grad is not None)
print(f'Number of gradients: {num_gradients}')

# Confirming the relationship between trainable parameters and gradients
assert num_trainable_params == num_gradients, "The number of trainable parameters should equal the number of gradients"

