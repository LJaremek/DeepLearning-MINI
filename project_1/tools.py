from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
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


def evaluate_model(model, loader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluation", unit="batch"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy
