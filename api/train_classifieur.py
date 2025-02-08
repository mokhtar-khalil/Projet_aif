import argparse
import os
from statistics import mean

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import MoviePosterNet  # Replace with your model

# Use GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def train(net, optimizer, loader, epochs=10, writer=None):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'Training loss: {mean(running_loss)}')
        if writer is not None:
            writer.add_scalar('Training Loss', mean(running_loss), epoch)

def test(model, dataloader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x).argmax(1)
            test_corrects += y_hat.eq(y).sum().item()
            total += y.size(0)
    return test_corrects / total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='MoviePoster', help='Experiment name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--nb_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--split_ratio', type=float, default=0.75, help='Train/Test split ratio')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    

    args = parser.parse_args()
    exp_name = args.exp_name
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    split_ratio = args.split_ratio
    data_path = args.data_path

    # Transforms for Movie Posters
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pretrained normalization
    ])

    # Load the full dataset
    full_dataset = ImageFolder(data_path, transform=transform)

    # Split into train and test sets
    train_size = int(split_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Data Loaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    net = MoviePosterNet().to(device)

    # TensorBoard
    writer = SummaryWriter(f'runs/{exp_name}')

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Train and Test
    train(net, optimizer, trainloader, nb_epochs, writer)
    test_acc = test(net, testloader)

    print(f'Test Accuracy: {test_acc}')

    # Save the Model
    os.makedirs('./weights', exist_ok=True)
    torch.save(net.state_dict(), './weights/movie_poster_net.pth')

    # Log Hyperparameters and Metrics
    writer.add_hparams({'lr': lr, 'bsize': batch_size}, {'hparam/accuracy': test_acc}, run_name='MoviePoster')
