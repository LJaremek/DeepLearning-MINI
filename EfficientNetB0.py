import torch.nn.functional as F
import torch.nn as nn


class MBConvBlock(nn.Module):
    def __init__(
            self,
            in_channels, out_channels, expansion_factor, stride, kernel_size=3,
            reduction_ratio=4, dropout_rate=0.2
            ):

        super(MBConvBlock, self).__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expansion_factor
        self.expand = in_channels != hidden_dim
        if self.expand:
            self.expand_conv = nn.Conv2d(
                in_channels, hidden_dim, kernel_size=1, bias=False
                )

            self.bn0 = nn.BatchNorm2d(hidden_dim)

        self.conv1 = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size, stride, kernel_size//2,
            groups=hidden_dim, bias=False
            )

        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.se = SqueezeExcitation(hidden_dim, reduction_ratio)
        self.conv2 = nn.Conv2d(
            hidden_dim, out_channels, kernel_size=1, bias=False
            )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x
        if self.expand:
            x = F.relu6(self.bn0(self.expand_conv(x)))

        x = F.relu6(self.bn1(self.conv1(x)))
        x = self.se(x)
        x = self.bn2(self.conv2(x))
        if self.use_residual:
            x = self.dropout(x)
            x += identity
        return x


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


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNetB0, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )

        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, 1, 1),
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


if __name__ == "__main__":
    # Checking the model
    model = EfficientNetB0(num_classes=1000)
    print(model)
