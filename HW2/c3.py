import torch
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
import numpy as np

# Transform for the CIFAR10 data
transform = transforms.Compose([transforms.ToTensor()])

# Load CIFAR10 training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# Worker counts to test
worker_counts = [0, 4, 8, 12, 16, 20, 24, 28, 32]

# Record DataLoader times
times = []

for num_workers in worker_counts:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=num_workers)
    
    start_time = time.time()
    for i, data in enumerate(trainloader, 0):
        # Simulate processing of data
        pass
    end_time = time.time()
    
    total_time = end_time - start_time
    times.append(total_time)
    print(f"Number of workers: {num_workers}, Time taken: {total_time:.2f} seconds")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(worker_counts, times, marker='o', linestyle='-', color='b')
plt.title('DataLoader Performance vs. Number of Workers (CIFAR10)')
plt.xlabel('Number of Workers')
plt.ylabel('Total Time Taken (seconds)')
plt.xticks(worker_counts)
plt.grid(True)
plt.show()

# Find the optimal number of workers
optimal_workers = worker_counts[np.argmin(times)]
print(f"The optimal number of workers is: {optimal_workers}")

