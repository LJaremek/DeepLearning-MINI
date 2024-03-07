import torch.nn.functional as F
import torch.nn as nn


class TinyModel(nn.Module):
    name = "TinyModel"

    def __init__(
            self,
            input_size: int = 3072,
            num_classes: int = 10
            ) -> None:

        super(TinyModel, self).__init__()

        self.linear1 = nn.Linear(input_size, 5_000)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(5_000, 3_000)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(3_000, 1_000)
        self.activation3 = nn.ReLU()
        self.linear4 = nn.Linear(1_000, num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.activation3(x)
        x = self.linear4(x)
        return x


class SimpleCNN(nn.Module):
    name = "SimpleCNN"

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        a = self.conv1(x)
        b = F.relu(a)
        x = self.pool(b)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
