import torch
import torch.nn as nn
import torch.nn.functional as F

# --- ResNet Bottleneck Block (using standard Conv2d) ---
class Bottleneck(nn.Module):
    expansion = 4 # Each bottleneck block expands channels by this factor

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        # First 1x1 convolution (reduce dimensions)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 convolution (main feature extraction)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Second 1x1 convolution (expand dimensions back)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity # Residual connection
        out = self.relu(out)

        return out

# --- ResNet50 Architecture for Spectrograms (using standard Conv2d) ---
class ResNet50ForSpectrogram(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50ForSpectrogram, self).__init__()
        self.in_channels = 64 # Initial number of channels after the first conv

        # --- Initial Layer ---
        # Original ResNet uses a 7x7 conv with stride 2.
        # For spectrograms, a 5x5 or 7x7 might be too aggressive for the frequency dimension,
        # especially if it's small. A 3x3 or 5x5 with stride 1 or 2 is often better.
        # We start with 1 input channel (spectrogram).
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- ResNet Stages ---
        # Each stage consists of multiple Bottleneck blocks
        # (3, 4, 6, 3 blocks for ResNet50)
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)  # 3 blocks
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2) # 4 blocks, 1st block has stride 2 for downsampling
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2) # 6 blocks, 1st block has stride 2 for downsampling
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2) # 3 blocks, 1st block has stride 2 for downsampling

        # --- Final Layers ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # Initialize weights (standard practice for CNNs)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        # The first block in a stage might need downsampling for the identity connection
        # if the spatial dimensions change or the channel count changes.
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Final layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    # Example usage for your bat call classification
    num_genera = 7 # As per your BetaV1 num_genera
    # Example input: Batch size 4, 1 channel, 128 freq bins, 256 time steps
    # Adjust this to match your actual average/padded spectrogram snippet size.
    example_input = torch.randn(4, 1, 128, 256)

    print("--- ResNet50 (Standard Convolutions) ---")
    model_resnet_standard = ResNet50ForSpectrogram(num_classes=num_genera)
    print(model_resnet_standard)
    num_params_standard = sum(p.numel() for p in model_resnet_standard.parameters() if p.requires_grad)
    print(f"Number of parameters (Standard Conv): {num_params_standard}")
    output_standard = model_resnet_standard(example_input)
    print(f"Output shape (Standard Conv): {output_standard.shape}\n")

    # To integrate this into your training script:
    # 1. Replace `BetaV1` with `ResNet50ForSpectrogram` in your `if __name__ == '__main__':` block.
    #    model = ResNet50ForSpectrogram(num_classes=num_genera)
    # 2. Make sure `num_genera` matches the actual number of classes from your dataset.
    # 3. Remember the input spectrogram dimensions considerations discussed previously.