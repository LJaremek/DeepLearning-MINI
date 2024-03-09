from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import optim
import torch.nn as nn
import numpy as np
import torch


def calc_data_mean_std(
        root_path: str
        ) -> dict[str, tuple[float, float, float]]:

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=root_path, transform=transform)

    dataloader = DataLoader(dataset, batch_size=64)

    mean = np.array([0.0, 0.0, 0.0])
    std = np.array([0.0, 0.0, 0.0])
    nb_samples = 0

    i = 0
    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0).numpy()
        std += data.std(2).sum(0).numpy()
        nb_samples += batch_samples
        i += 1
        if i == 10:
            break

    mean /= nb_samples
    std /= nb_samples

    return {
        "mean": mean.tolist(),
        "std": std.tolist()
    }


def plot_images(images, labels, dataset, image_size: tuple[int, int]) -> None:
    images = images.view(-1, 3, image_size[0], image_size[1])

    plt.figure(figsize=(2, 2))
    plt.axis("off")

    # Zmiana kolejności wymiarów z (C, H, W) na (H, W, C)
    img = images[0].permute(1, 2, 0)

    label = labels[0].item()
    plt.imshow(img)
    plt.title(f"Label: {dataset.classes[label]}")
    plt.show()


def train_model(
        device: torch.device,
        model: nn.Module,
        transform: transforms.Compose,
        data_path: str,
        batch_size: int,
        epochs: int,
        criterion,
        optimizer: optim.Optimizer,
        printing: bool = True
        ) -> None:

    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()

    for epoch in range(epochs):
        epoch_losses = []
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            # print("L:", [int(el) for el in labels])
            # print("O:", [int(el.argmax(dim=0)) for el in outputs])
            # print("O:", outputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses)/len(epoch_losses)
        if printing:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


def valid_model(
        device: torch.device,
        model: nn.Module,
        transform: transforms.Compose,
        data_path: str,
        batch_size: int,
        criterion,
        printing: bool = True
        ) -> None:

    test_loss = 0
    correct = 0
    total = 0

    dataset = datasets.ImageFolder(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / len(dataloader.dataset)
    accuracy = 100 * correct / total

    if printing:
        print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


def run_model(
        device: torch.device,
        model: nn.Module,
        transform: transforms.Compose,
        train_path: str,
        test_path: str,
        batch_size: int,
        epochs: int,
        criterion,
        optimizer: optim.Optimizer,
        printing: bool = True
        ) -> None:

    train_model(
        device,
        model,
        transform,
        train_path,
        batch_size,
        epochs,
        criterion,
        optimizer,
        printing
        )

    valid_model(
        device,
        model,
        transform,
        test_path,
        batch_size,
        criterion,
        printing
    )
