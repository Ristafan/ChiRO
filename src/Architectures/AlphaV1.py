import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaV1(nn.Module):
    def __init__(self):
        super(AlphaV1, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global pooling to adapt to varying input size

        self.fc1 = nn.Linear(64, 128)  # Adjusted for the output size from global pooling
        self.fc2 = nn.Linear(128, 2)   # Output layer for binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Apply global average pooling to reduce to (batch_size, 64, 1, 1)
        x = self.global_avg_pool(x)

        # Flatten the tensor to (batch_size, 64)
        x = torch.flatten(x, start_dim=1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    model = AlphaV1()
    print(model)
