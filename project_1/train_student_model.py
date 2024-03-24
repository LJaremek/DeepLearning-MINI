import json

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch

from models import EfficientNetB0, SimpleEfficientNetB0, SuperSimpleEfficientNetB0
from tools import calc_data_mean_std, evaluate_model


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

classes = len(train_dataset.classes)

# Model initialization
teacher_model = EfficientNetB0(num_classes=classes).to(device)
teacher_path = "project_1\\models_status\\10c_model_epoch_7.pth"
teacher_model.load_state_dict(
    torch.load(teacher_path, map_location=device)
    )

student_model = SuperSimpleEfficientNetB0(num_classes=classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(student_model.parameters(), lr=0.0001)


# Training loop
teacher_model.eval()

temperature = 2
percent_of_teacher = 0.5

print("Started training")
num_epochs = 10
for epoch in range(num_epochs):
    student_model.train()
    running_loss = 0.0
    with tqdm(train_loader, unit="batch") as tepoch:
        for images, labels in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            student_logits = student_model(images)

            with torch.no_grad():
                teacher_logits = teacher_model(images)

            # Soften the student logits by
            # applying softmax first and log() second.
            soft_targets = nn.functional.softmax(
                teacher_logits / temperature, dim=-1
                )

            soft_prob = nn.functional.log_softmax(
                student_logits / temperature, dim=-1
                )

            outputs = student_model(images)

            # Calculate the soft targets loss.
            # Scaled by temperature**2 as suggested by the authors of the paper
            # "Distilling the knowledge in a neural network".
            _sum = torch.sum(soft_targets * (soft_targets.log() - soft_prob))
            soft_targets_loss = _sum / soft_prob.size()[0] * (temperature**2)

            # Calculate the true label loss
            label_loss = criterion(student_logits, labels)

            # Weighted sum of the two losses
            loss = (
                percent_of_teacher * soft_targets_loss +
                (1-percent_of_teacher) * label_loss
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            tepoch.set_postfix(loss=running_loss/len(train_loader))

        accuracy = evaluate_model(test_loader, student_model, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Accuracy: {accuracy:.2f}%")
        torch.save(
            student_model.state_dict(),
            f"{classes}c_student_model_{student_model._get_name}_epoch_{epoch+1}.pth"
            )

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
            tepoch.set_postfix(accuracy=f"{accuracy:.2f}%")

print(f"Accuracy on the test set: {accuracy:.2f}%")
