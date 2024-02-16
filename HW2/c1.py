import time
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
from model import BasicBlock, ResNet


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def main():
    parser = argparse.ArgumentParser(description="CIFAR10 Training Pytorch")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disabled CUDA training"
    )

    parser.add_argument("-c", type=int, default=1, help="Homework part number")

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

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device : ", device)

    # Data loading
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
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

    # if args.c == 3:
    #   print("Results of Data loading time: ")
    #
    #   exit()

    model = ResNet18().to(device)

    if args.optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )
    elif args.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        raise ValueError("Unsupported optimizer provided. Choose 'sgd' or 'adam'.")

    criterion = nn.CrossEntropyLoss()

    # Training phase

    for epoch in range(1, 6):  # run for 5 epochs
        epoch_start_time = time.perf_counter()

        model.train()
        training_time = 0  # Reset training time for each epoch
        data_loading_time = 0  # Reset data-loading time for each epoch

        for batch_idx, (data, target) in enumerate(train_loader):
            
            #Added the following
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,shuffle=True, num_workers=num_workers) 

            start_data_loading_time = time.perf_counter()
            for i, data in enumerate(trainloader, 0):
            # Simulate processing of data
             pass
            end_data_loading_time = time.perf_counter()

            data_loading_time = end_data_loading_time - start_data_loading_time;

            data, target = data.to(device), target.to(device)

            start_training_time = time.perf_counter()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            end_training_time = time.perf_counter()

            training_time += end_training_time - start_training_time

            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                    f"Loss: {loss.item():.6f} "
                    f"Accuracy: {100. * correct / len(data):.2f}%"
                )

        epoch_end_time = time.perf_counter()
        total_epoch_time = epoch_end_time - epoch_start_time
        print(
            f"Epoch {epoch} Complete: \n"
            f"\tData-Loading Time: {data_loading_time:.6f} seconds\n"
            f"\tTraining Time: {training_time:.6f} seconds\n"
            f"\tTotal Epoch Time: {total_epoch_time:.6f} seconds\n"
        )


if __name__ == "__main__":
    main()
