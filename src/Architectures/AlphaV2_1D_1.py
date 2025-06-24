import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaV2_1D_1(nn.Module):
    def __init__(self, dropout_rate=0.3, batch_norm=True, global_pooling="avg"):
        super(AlphaV2_1D_1, self).__init__()

        # Convolutional layers
        self.conv1_time = nn.Conv2d(1, 1, kernel_size=(1, 1), stride=1, padding=(1, 0), groups=1)
        self.conv1_freq = nn.Conv2d(1, 32, kernel_size=(1, 1), stride=1, padding=(0, 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(32) if batch_norm else None

        self.conv2_time = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=1, padding=(1, 0), groups=32)
        self.conv2_freq = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=1, padding=(0, 1))
        self.batchnorm2 = nn.BatchNorm2d(64) if batch_norm else None

        self.conv3_time = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=1, padding=(1, 0), groups=64)
        self.conv3_freq = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=1, padding=(0, 1))
        self.dropout = nn.Dropout(dropout_rate)

        # Global average pooling - works with any input size
        if global_pooling == "avg":
            self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.global_avg_pool = nn.AdaptiveMaxPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)  # 2 classes: bat call or noise

    def forward(self, x):
        # First block
        x = self.conv1_time(x)
        x = self.conv1_freq(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.batchnorm1(x) if self.batchnorm1 else x

        # Second block
        x = self.conv2_time(x)
        x = self.conv2_freq(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.batchnorm2(x) if self.batchnorm2 else x

        # Third block
        x = self.conv3_time(x)
        x = self.conv3_freq(x)
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
    model = AlphaV2_1D_1()
    print(model)
