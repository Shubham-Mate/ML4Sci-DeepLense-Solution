import torch
import torch.nn.functional as F

class LensCNN(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(LensCNN, self).__init__()

        # Convolutional layers
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)  # (3x64x64) → (32x64x64)
        self.bn1 = torch.nn.BatchNorm2d(32)
        
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (32x64x64) → (64x64x64)
        self.bn2 = torch.nn.BatchNorm2d(64)

        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)  # (64x32x32) → (128x32x32)
        self.bn3 = torch.nn.BatchNorm2d(128)

        self.pool = torch.nn.MaxPool2d(2, 2)  # Reduce size by half

        # Global Average Pooling
        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d(1)  # (128x8x8) → (128x1x1)

        # Fully connected layer
        self.fc = torch.nn.Linear(128, num_classes)  # 128 → 2

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 + ReLU + Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 + ReLU + Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv3 + ReLU + Pool

        x = self.global_avg_pool(x)  # Global Average Pooling
        x = torch.flatten(x, 1)  # Flatten to (batch_size, 128)
        x = self.fc(x)  # Fully Connected Layer

        return x