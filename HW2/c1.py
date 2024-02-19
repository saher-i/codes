import time
import torchvision
import torchvision.transforms as transforms
import time
import torch.nn as nn
import torch
import torch.optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from model import BasicBlock, ResNet
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


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
        epoch_start_time = time.perf_counter()

        model.train()
        training_time = 0  # Reset training time for each epoch
        data_loading_time = 0  # Reset data-loading time for each epoch

        start_data_loading_time = time.perf_counter()
        for i, data in enumerate(train_loader, 0):
            # Simulate processing of data
            pass
        end_data_loading_time = time.perf_counter()
        data_loading_time = end_data_loading_time - start_data_loading_time

        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = (
                data.to(device),
                target.to(device),
            )  # pushed data, target to right device
            start_training_time = time.perf_counter()
            optimizer.zero_grad()  # reset
            output = model(data)  # prediction
            loss = criterion(output, target)  # prediction and target differnece
            loss.backward()  # pytorch function inbuilt
            optimizer.step()  # updates gradients
            end_training_time = time.perf_counter()
            training_time += end_training_time - start_training_time

            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()  # accuracy

            # if batch_idx % 100 == 0:
            #     print(
            #         f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
            #         f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
            #         f"Loss: {loss.item():.6f} "
            #         f"Accuracy: {100. * correct / len(data):.2f}%"
            #     )

        epoch_end_time = time.perf_counter()
        total_epoch_time = epoch_end_time - epoch_start_time
        print(
            f"Epoch {epoch} Complete: \n"
            f"\tData-Loading Time: {data_loading_time:.6f} seconds\n"
            f"\tTraining Time: {training_time:.6f} seconds\n"
            f"\tTotal Epoch Time: {total_epoch_time:.6f} seconds\n"
        )


def c3(train_dataset, worker_counts):
    # Record DataLoader times
    times = []

    for num_workers in worker_counts:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=num_workers
        )

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
        epoch_start_time = time.perf_counter()

        model.train()
        training_time = 0  # Reset training time for each epoch
        data_loading_time = 0  # Reset data-loading time for each epoch

        start_data_loading_time = time.perf_counter()
        for i, data in enumerate(train_loader, 0):
            # Simulate processing of data
            pass
        end_data_loading_time = time.perf_counter()
        data_loading_time = end_data_loading_time - start_data_loading_time

        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = (
                data.to(dev),
                target.to(dev),
            )  # pushed data, target to right device
            start_training_time = time.perf_counter()
            optimizer.zero_grad()  # reset
            output = model(data)  # prediction
            loss = criterion(output, target)  # prediction and target differnece
            loss.backward()  # pytorch function inbuilt
            optimizer.step()  # updates gradients
            end_training_time = time.perf_counter()
            training_time += end_training_time - start_training_time

            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()  # accuracy

        epoch_end_time = time.perf_counter()
        total_epoch_time = epoch_end_time - epoch_start_time
        total_time = total_time + total_epoch_time

    print(f"\tAverage Time over 5 Epochs: {total_time/5:.6f} seconds\n")


def gradients_params_count(optim="sgd"):
    model = ResNet18()

    # Count the number of trainable parameters
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_trainable_params}")

    if optim == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Example input for a forward pass (assuming CIFAR-10 images, with shape 32x32x3)
    example_input = torch.randn(1, 3, 32, 32)  # Batch size of 1

    optimizer.zero_grad()
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

    # step take
    optimizer.step()

    # Check if gradients exist and count them
    num_gradients = sum(
        p.grad.numel() for p in model.parameters() if p.grad is not None
    )
    print(f"Number of gradients: {num_gradients}")

    # Confirming the relationship between trainable parameters and gradients
    assert (
        num_trainable_params == num_gradients
    ), "The number of trainable parameters should equal the number of gradients"


def main():
    parser = argparse.ArgumentParser(description="CIFAR10 Training Pytorch")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disabled CUDA training"
    )
    parser.add_argument(
        "--plot", action="store_true", default=False, help="Plots the C3 part times"
    )

    parser.add_argument(
        "--data-path", type=str, default="./data", help="path to dataset"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="set number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="which optimizer? 'sgd' or 'adam' (default: 'sgd')",
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

    # Run c2
    print("*" * 100)
    print()
    print("Running Part C2")
#    c2(train_dataset, args, workers=2)
    print()

    # Run c3
    print("*" * 100)
    print()
    print("Running Part C3")
    worker_counts = [0, 4, 8, 12, 16, 20, 24, 28, 32]
 #   times = c3(train_dataset, worker_counts)
    print()

    # Plot the times
    if args.plot:
        print("*" * 100)
        print()
        print("Plotting for C3")
        plot_workers_vs_time(worker_counts, times)

    # Gradients calculation - Q3, Q4
    print("*" * 100)
    print()
    print("Running Q3 using SGD optimizer")
#    gradients_params_count("sgd")
    print()
    print("Using Adam optimizer for Q4\n")
#    gradients_params_count("adam")

    # Run c4
    print("*" * 100)
    print()
    print("Running Part C4")
#    c2(train_dataset, args, workers=1)
#    c2(train_dataset, args, workers=8)
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
#    c5(train_dataset, args, workers=8, dev="cpu")


if __name__ == "__main__":
    main()
