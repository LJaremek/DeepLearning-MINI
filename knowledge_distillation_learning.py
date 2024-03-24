import json

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
from collections import deque
from EfficientNetB0 import EfficientNetB0
from SimpleEfficientNetB0 import SimpleEfficientNetB0
from tools import calc_data_mean_std


IMAGE_SIZE = 32

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
train_dataset = datasets.ImageFolder('data/train', transform=transform)
test_dataset = datasets.ImageFolder('data/test', transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model initialization
teacher_model = EfficientNetB0(num_classes=len(train_dataset.classes)).to(device)
teacher_model.load_state_dict(torch.load('model_epoch_10.pth', map_location=device))

student_model = SimpleEfficientNetB0(num_classes=len(train_dataset.classes)).to(device)

ce_loss = nn.CrossEntropyLoss()

teacher_model.eval()
student_model.train()

last_test_loss = deque(maxlen=3)
last_test_loss.append(float("inf"))

epochs = 10

learning_rate = 0.0001
weight_decay = 0

optimizer = optim.Adam(
    student_model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
    )

temperature = 2

percent_of_teacher = 0.5
print("Started training")
for epoch in range(0, epochs):
    running_loss = 0.0
    with tqdm(train_loader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")
        for images, labels in tepoch:
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            student_logits = student_model(images)

            with torch.no_grad():
                teacher_logits = teacher_model(images)
            
            # Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / temperature, dim=-1)
            outputs = student_model(images)
            # Calculate the soft targets loss. Scaled by temperature**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (temperature**2)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)
            # Weighted sum of the two losses
            loss = percent_of_teacher * soft_targets_loss + (1-percent_of_teacher) * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            tepoch.set_postfix(loss=running_loss/len(train_loader))

        torch.save(student_model.state_dict(), f"student_model_epoch_{epoch+1}.pth")

# Evaluation
student_model.eval()
correct = 0
total = 0
with torch.no_grad():
    with tqdm(test_loader, unit="batch") as tepoch:
        for images, labels in tepoch:
            tepoch.set_description("Evaluation")
            images, labels = images.to(device), labels.to(device)

            outputs = student_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            tepoch.set_postfix(accuracy=f'{accuracy:.2f}%')

print(f'Accuracy on the test set: {accuracy:.2f}%')