from itertools import product
import json

from torchvision import datasets, transforms
from torch import optim
import torch

from models import SimpleCNN, AdvancedCNN
from tools import run_model


with open("data_mean_std.json", "r", -1, "utf-8") as f:
    data_desc = json.loads("".join(f.readlines()))

data_mean = data_desc["mean"]
data_std = data_desc["std"]

IMAGE_SIZE = 32

transform_list = {
    "small":
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=data_mean,
                std=data_std
                )
        ]),
    "big":
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(IMAGE_SIZE, padding=4),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_mean, std=data_std),
            transforms.RandomErasing(
                p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random"
                )
        ])
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print()

dataset = datasets.ImageFolder(root="data/train")
classes_number = len(dataset.classes)

available_models = (
    SimpleCNN,
    AdvancedCNN
)

learning_rates = (
    0.00001,
)

batch_sizes = (
    64,
)

EPOCHS = 20

products = product(
    available_models,
    learning_rates,
    batch_sizes,
    transform_list
)

for model, learning_rate, batch_size, transform_name in products:
    print("-----", model.name, "-----")
    print("PARAMS:")
    print(f"\tLearning rate: {learning_rate}")
    print(f"\tBatch size: {batch_size}")
    print(f"\tTransformations name: {transform_name}")
    model = model(num_classes=classes_number).to(device)

    transform = transform_list[transform_name]

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    run_model(
        device,
        model,
        transform,
        "data/train",
        "data/test",
        batch_size,
        EPOCHS,
        criterion,
        optimizer
    )
