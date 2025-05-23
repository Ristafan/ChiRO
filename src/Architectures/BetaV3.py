import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaV3(nn.Module):
    def __init__(self, num_genera, dropout_rate=0.3):
        super(BetaV3, self).__init__()
        self.num_genera = num_genera
        self.dropout = nn.Dropout(dropout_rate)

        # Convolutional layers - increased number of filters
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=6, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers - adjusted output size
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_genera) # Output layer for the number of genera

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.batchnorm1(x)

        # Second block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.batchnorm2(x)

        # Third block
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        # Global average pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # No softmax here, as it's included in CrossEntropyLoss

        return x


if __name__ == '__main__':
    # Example usage with 5 different genera
    num_genera = 5
    model = BetaV3(num_genera=num_genera)
    print(model)