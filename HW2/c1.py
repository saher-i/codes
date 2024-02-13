import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
from torch.optim.lr_scheduler import StepLR
from hpmlc1 import BasicBlock, ResNet
from tqdm import tqdm

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--data-path', type=str, default='./data',
                        help='path to dataset')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help="optimizer; 'sgd' or 'adam' (default: 'sgd')")
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Data loading code
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root=args.data_path, train=True,
                                     download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                              num_workers=args.num_workers)

    model = ResNet18().to(device)

    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        raise ValueError("Unsupported optimizer. Use 'sgd' or 'adam'.")

    criterion = nn.CrossEntropyLoss()

    # Training phase
    model.train()
    for epoch in tqdm(range(1, 6)):  # run for 5 epochs
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f} '
                      f'Accuracy: {100. * correct / len(data):.2f}%')

if __name__ == '__main__':
    main()

