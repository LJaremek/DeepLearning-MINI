import torch.nn as nn

from EfficientNetB0 import MBConvBlock


class SqueezeExcitation(nn.Module):

    def __init__(self, in_channels, reduction_ratio):
        super(SqueezeExcitation, self).__init__()
        reduced_channels = in_channels // reduction_ratio
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            nn.ReLU(),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class SimpleEfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super(SimpleEfficientNetB0, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )

        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, 1, 1),
            MBConvBlock(16, 24, 6, 2),
            MBConvBlock(24, 40, 6, 2),
            MBConvBlock(40, 80, 6, 2),
            MBConvBlock(80, 112, 6, 1), 
            MBConvBlock(112, 112, 6, 1),
            MBConvBlock(112, 192, 6, 2), 
            MBConvBlock(192, 320, 6, 1),
        )

        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    # Checking the model
    model = SimpleEfficientNetB0(num_classes=1000)
    print(model)
