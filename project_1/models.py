import torch.nn as nn

from _models_parts import MBConvBlock


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNetB0, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )

        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, 1, 1),  # 1x
            MBConvBlock(16, 24, 6, 2),  # 2x
            MBConvBlock(24, 24, 6, 1),
            MBConvBlock(24, 40, 6, 2),  # 2x
            MBConvBlock(40, 40, 6, 1),
            MBConvBlock(40, 80, 6, 2),  # 3x
            MBConvBlock(80, 80, 6, 1),
            MBConvBlock(80, 80, 6, 1),
            MBConvBlock(80, 112, 6, 1),  # 3x
            MBConvBlock(112, 112, 6, 1),
            MBConvBlock(112, 112, 6, 1),
            MBConvBlock(112, 192, 6, 2),  # 4x
            MBConvBlock(192, 192, 6, 1),
            MBConvBlock(192, 192, 6, 1),
            MBConvBlock(192, 192, 6, 1),
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


class SuperSimpleEfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super(SuperSimpleEfficientNetB0, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )

        self.blocks = nn.Sequential(
            MBConvBlock(32, 24, 1, 1),
            MBConvBlock(24, 80, 6, 2),
            MBConvBlock(80, 112, 6, 1),
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
    

class SuperGigaSimpleEfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super(SuperSimpleEfficientNetB0, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )

        self.blocks = nn.Sequential(
            MBConvBlock(32, 80, 1, 1),
            MBConvBlock(80, 112, 6, 1),
            MBConvBlock(112, 320, 6, 1),
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
    model = EfficientNetB0(num_classes=10)
    print(model)
