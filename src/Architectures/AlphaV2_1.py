import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaV2_1(nn.Module):
    def __init__(self, dropout_rate=0.3, batch_norm=True):
        super(AlphaV2_1, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(2049, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.batchnorm1 = nn.BatchNorm1d(32) if batch_norm else None

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(64) if batch_norm else None

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(dropout_rate)

        # Global average pooling - works with any input size
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)  # 2 classes: bat call or noise

    def forward(self, x):
        batch_size, channels, freq_bins, time_steps = x.size()
        x = x.view(batch_size, channels * freq_bins, time_steps)

        #x = x.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        #x = x.reshape(batch_size * time_steps, channels, freq_bins)

        # First block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.batchnorm1(x) if self.batchnorm1 else x

        # Second block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.batchnorm2(x) if self.batchnorm2 else x

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
    model = AlphaV2_1()
    print(model)
