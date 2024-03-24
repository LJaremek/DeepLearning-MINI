import json

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch

from models import EfficientNetB0
from tools import calc_data_mean_std


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


try:
    with open("data_mean_std.json", "r", -1, "utf-8") as f:
        data_desc = json.loads("".join(f.readlines()))
except FileNotFoundError:
    data_desc = calc_data_mean_std("data")

data_mean = data_desc["mean"]
data_std = data_desc["std"]


# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=data_mean, std=data_std),
])

# Load datasets
train_dataset = datasets.ImageFolder("data/train", transform=transform)
test_dataset = datasets.ImageFolder("data/test", transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

classes = len(train_dataset.classes)

# Model initialization
model = EfficientNetB0(num_classes=classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("Start training...")
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    with tqdm(train_loader, unit="batch") as tepoch:
        for images, labels in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            tepoch.set_postfix(loss=running_loss/len(train_loader))

    torch.save(
        model.state_dict(),
        f"{classes}c_model_epoch_{epoch+1}.pth"
        )

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    with tqdm(test_loader, unit="batch") as tepoch:
        for images, labels in tepoch:
            tepoch.set_description("Evaluation")
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            tepoch.set_postfix(accuracy=f'{accuracy:.2f}%')

print(f'Accuracy on the test set: {accuracy:.2f}%')
