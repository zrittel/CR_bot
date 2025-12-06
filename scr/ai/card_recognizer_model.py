import torch
import torch.nn as nn
import torch.nn.functional as F


class CardCNN(nn.Module):
    """CNN для определения карт колоды Clash Royale"""

    def __init__(self, num_classes=9):  # 9 карт + None
        super().__init__()

        # Вход: (batch, 3, 64, 64) - RGB картинки карт
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 32, 64, 64
        self.pool = nn.MaxPool2d(2, 2)  # 32, 32, 32

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64, 32, 32
        # Pool: 64, 16, 16

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 128, 16, 16
        # Pool: 128, 8, 8

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32, 32, 32
        x = self.pool(F.relu(self.conv2(x)))  # 64, 16, 16
        x = self.pool(F.relu(self.conv3(x)))  # 128, 8, 8
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
