import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaBetaV1(nn.Module):
    def __init__(self, num_genera, dropout_rate=0.3):
        super(AlphaBetaV1, self).__init__()

        # Shared convolutional layers (same as AlphaV2)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Shared fully connected layer
        self.fc1 = nn.Linear(128, 64)

        # Head Alpha: Bat call or noise classification
        self.head_alpha = nn.Linear(64, 2)  # 2 classes: bat call or noise

        # Head Beta: Genus classification
        self.head_beta = nn.Linear(64, num_genera)  # num_genera classes

    def forward(self, x):
        # Shared convolutional layers
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)

        # Shared fully connected layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Head Alpha: Bat call or noise
        out_alpha = self.head_alpha(x)

        # Head Beta: Genus classification
        out_beta = self.head_beta(x)

        return out_alpha, out_beta


if __name__ == '__main__':
    # Example usage:
    # Assuming you have 10 genera (including the 'chiroptera' catch-all class)
    num_genera = 7
    model = AlphaBetaV1(num_genera=num_genera)

    # Example input tensor (batch size of 1, 1 channel, some height, some width)
    #  Replace the height and width with your actual spectrogram dimensions
    input_tensor = torch.randn(1, 1, 128, 256) # Example: 128 height, 256 width

    # Forward pass
    out_alpha, out_beta = model(input_tensor)

    # Print the shapes of the outputs
    print("Output Alpha (Bat Call/Noise):", out_alpha.shape)  # Should be [1, 2]
    print("Output Beta (Genus):", out_beta.shape)    # Should be [1, num_genera]
