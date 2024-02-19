import time
import torchvision.transforms as transforms
import time
import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
from model import BasicBlock, ResNet
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity


def ResNet18(use_bn=True):
    return ResNet(
        block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=10, use_bn=use_bn
    )


def c2(train_dataset, args, workers):
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=workers
    )

    model = ResNet18().to(device)

    if args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )
    elif args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        raise ValueError("Unsupported optimizer provided. Choose 'sgd' or 'adam'.")

    criterion = nn.CrossEntropyLoss()

    # Training phase

    for epoch in range(1, 6):  # run for 5 epochs
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
        ) as prof:
            with record_function("model_training"):
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

        # Save profile
        prof.export_chrome_trace(f"trace_epoch_{epoch}.json")

    print("Profiling complete. Trace files saved.")


def c3(train_dataset, worker_counts):
    # Record DataLoader times
    times = []
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=workers
    )
    for num_workers in worker_counts:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
        ) as prof:

        start_time = time.time()
        for i, data in enumerate(trainloader, 0):
            # Simulate processing of data
            pass
        end_time = time.time()

        total_time = end_time - start_time
        times.append(total_time)
        print(f"Number of workers: {num_workers}, Time taken: {total_time:.2f} seconds")

    return times


# Plotting function
def plot_workers_vs_time(worker_counts, times):
    plt.figure(figsize=(10, 6))
    plt.plot(worker_counts, times, marker="o", linestyle="-", color="b")
    plt.title("DataLoader Performance vs. Number of Workers")
    plt.xlabel("Number of Workers")
    plt.ylabel("Total Time Taken (seconds)")
    plt.xticks(worker_counts)
    plt.grid(True)
    plt.show()


def c5(train_dataset, args, workers, dev):
    total_time = 0

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=workers
    )

    model = ResNet18().to(dev)

    if args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )
    elif args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        raise ValueError("Unsupported optimizer provided. Choose 'sgd' or 'adam'.")

    criterion = nn.CrossEntropyLoss()

    # Training phase

    for epoch in range(1, 6):  # run for 5 epochs
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
        ) as prof:
            with record_function("model_training"):
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

        # Save profile
        prof.export_chrome_trace(f"trace_epoch_{epoch}.json")

    print("Profiling complete. Trace files saved.")


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch CIFAR10 Training with Profiling"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--data-path", type=str, default="./data", help="path to dataset"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="optimizer; 'sgd' or 'adam' (default: 'sgd')",
    )
    parser.add_argument(
        "--plot", action="store_true", default=False, help="Plots the times"
    )
    args = parser.parse_args()

    global device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device : ", device)

    # Data loading
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=args.data_path, train=True, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=args.num_workers
    )

    print(
        "#################################***********Extra Credit**************##################################\n"
    )

    # Run c2
    print("*" * 100)
    print()
    print("Running Part C2")
    c2(train_dataset, args, workers=2)
    print()

    # Run C3
    print("*" * 100)
    print()
    print("Running Part C3")
    worker_counts = [0, 4, 8, 12, 16, 20, 24, 28, 32]
    times = c3(train_dataset, worker_counts)
    print()

    # Plot the times
    if args.plot:
        print("*" * 100)
        print()
        print("Plotting for C3")
        plot_workers_vs_time(worker_counts, times)

    # Run C4
    print("*" * 100)
    print()
    print("Running Part C4")
    c2(train_dataset, args, workers=1)
    c2(train_dataset, args, workers=8)
    print()

    # Running C5
    print("*" * 100)
    print()
    print("Running C5 on CUDA")
    c5(train_dataset, args, workers=8, dev="cuda")
    print()
    print("*" * 100)
    print()
    print("Running C5 on CPU")
    print("device: cpu")
    c5(train_dataset, args, workers=8, dev="cpu")


if __name__ == "__main__":
    main()
