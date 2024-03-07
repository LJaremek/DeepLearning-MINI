from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


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
