import json

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import optim
import torch

from models import TinyModel, SimpleCNN


with open("data_mean_std.json", "r", -1, "utf-8") as f:
    data_desc = json.loads("".join(f.readlines()))

data_mean = data_desc["mean"]
data_std = data_desc["std"]

IMAGE_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=data_mean,
        std=data_std
        ),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print()

dataset = datasets.ImageFolder(root="data/train", transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

classes_number = len(dataset.classes)


def plot_images(images, labels, dataset):
    images = images.view(-1, 3, IMAGE_SIZE, IMAGE_SIZE)

    plt.figure(figsize=(2, 2))
    plt.axis("off")

    # Zmiana kolejności wymiarów z (C, H, W) na (H, W, C)
    img = images[0].permute(1, 2, 0)

    label = labels[0].item()
    plt.imshow(img)
    plt.title(f"Label: {dataset.classes[label]}")
    plt.show()


available_models = (
    TinyModel,
    SimpleCNN
)

epochs = 10


for model in available_models:
    print("-----", model.name, "-----")
    model = model(num_classes=classes_number).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    model.train()

    for epoch in range(epochs):
        epoch_losses = []
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            if model.name == "TinyModel":
                images = images.view(-1, IMAGE_SIZE*IMAGE_SIZE*3)

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
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    test_dataset = datasets.ImageFolder(root="data/test", transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            if model.name == "TinyModel":
                images = images.view(-1, IMAGE_SIZE*IMAGE_SIZE*3)

            outputs = model(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / len(test_dataloader.dataset)
    accuracy = 100 * correct / total

    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
