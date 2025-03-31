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

        self.global_avg_pool = nn.AdaptiveAvgPool2d((8, 8))  # Ensure fixed-size output
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 2)  # Binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = self.global_avg_pool(x)  # Adaptive pooling to ensure fixed shape
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage:
# Assuming input spectrogram has shape (batch_size, 1, time_steps, frequency_bins)
# Example input: torch.randn(8, 1, 100, 64)  # (batch, channels, time, frequency)


if __name__ == '__main__':
    model = AlphaV1()
    print(model)
