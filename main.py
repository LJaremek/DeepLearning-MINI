import json

from torchvision import datasets, transforms
from torch.utils.data import DataLoader  # , Subset
import matplotlib.pyplot as plt
from torch import optim
import torch

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

print("Loading data...")
dataset = datasets.ImageFolder(root="data/train_small", transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# selected_classes = ["airplane", "dog"]
# indices = [
#     i
#     for i, (_, label) in enumerate(dataset)
#     if dataset.classes[label] in selected_classes
#     ]

# dataset = Subset(dataset, indices)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

classes_number = len(dataset.classes)


class TinyModel(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            first_hidden_size: int,
            second_hidden_size: int,
            output_size: int
            ) -> None:

        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(input_size, first_hidden_size)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(first_hidden_size, second_hidden_size)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(second_hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        return x


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


model = TinyModel(IMAGE_SIZE*IMAGE_SIZE*3, 2000, 500, classes_number)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

epochs = 10

model.train()

first = True


print("Start training...")
for epoch in range(epochs):
    for images, labels in dataloader:
        images = images.view(-1, IMAGE_SIZE*IMAGE_SIZE*3)

        optimizer.zero_grad()

        outputs = model(images)

        if first:
            print(images.shape, labels.shape)
            print(outputs.shape)
            print()
            first = False

        # print("L:", [int(el) for el in labels])
        # print("O:", [int(el.argmax(dim=0)) for el in outputs])
        # print("O:", outputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("Training is done.")
