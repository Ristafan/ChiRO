import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaV2(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(AlphaV2, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Global average pooling - works with any input size
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)  # 2 classes: bat call or noise

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        # Second block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        # Third block
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        # Global average pooling - this enables variable-sized inputs
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    model = AlphaV2()
    print(model)
