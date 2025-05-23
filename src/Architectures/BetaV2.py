import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Helper for Depthwise Separable Convolution ---
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation,
                                   groups=in_channels, bias=bias)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# --- ResNet Bottleneck Block ---
class Bottleneck(nn.Module):
    expansion = 4 # Each bottleneck block expands channels by this factor

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_separable=False):
        super(Bottleneck, self).__init__()
        # Choose the convolution type based on the flag
        Conv = SeparableConv2d if use_separable else nn.Conv2d

        # First 1x1 convolution (reduce dimensions)
        self.conv1 = Conv(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 convolution (main feature extraction)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Second 1x1 convolution (expand dimensions back)
        self.conv3 = Conv(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
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

# --- ResNet50 Architecture for Spectrograms ---
class ResNet50ForSpectrogram(nn.Module):
    def __init__(self, num_classes, use_separable=False):
        super(ResNet50ForSpectrogram, self).__init__()
        self.in_channels = 64 # Initial number of channels after the first conv
        self.use_separable = use_separable # Option to use separable convolutions
        # Note: The initial conv1 should usually remain a standard Conv2d
        # for a 1-channel input to expand features into 64 channels effectively.
        # It's less common to make the very first layer separable.
        # If you truly want the first layer separable, you'd define it here.
        # For typical ResNet adaptations, conv1 is standard.
        # For simplicity and aligning with common practices, we'll keep conv1 standard.
        # If you were to make it separable, you'd replace:
        # self.conv1 = nn.Conv2d(1, 64, ...) with self.conv1 = SeparableConv2d(1, 64, ...)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- ResNet Stages ---
        # Each stage consists of multiple Bottleneck blocks
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # --- Final Layers ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # Initialize weights (standard practice for CNNs)
        for m in self.modules():
            # Check if it's a standard Conv2d layer or a BatchNorm layer
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # No need to explicitly handle SeparableConv2d here because its internal
            # nn.Conv2d layers will be caught by the `isinstance(m, nn.Conv2d)` check
            # when the loop iterates deeper into `SeparableConv2d`'s submodules.

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        # The first block in a stage might need downsampling for the identity connection
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            # Choose the convolution type for downsample path
            DownsampleConv = SeparableConv2d if self.use_separable else nn.Conv2d
            downsample = nn.Sequential(
                DownsampleConv(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, use_separable=self.use_separable))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, use_separable=self.use_separable))

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
    example_input = torch.randn(4, 1, 128, 256)

    print("--- ResNet50 (Standard Convolutions) ---")
    model_resnet_standard = ResNet50ForSpectrogram(num_classes=num_genera, use_separable=False)
    print(model_resnet_standard)
    num_params_standard = sum(p.numel() for p in model_resnet_standard.parameters() if p.requires_grad)
    print(f"Number of parameters (Standard Conv): {num_params_standard}")
    output_standard = model_resnet_standard(example_input)
    print(f"Output shape (Standard Conv): {output_standard.shape}\n")


    print("--- ResNet50 (with Separable Convolutions) ---")
    model_resnet_separable = ResNet50ForSpectrogram(num_classes=num_genera, use_separable=True)
    print(model_resnet_separable)
    num_params_separable = sum(p.numel() for p in model_resnet_separable.parameters() if p.requires_grad)
    print(f"Number of parameters (Separable Conv): {num_params_separable}")
    output_separable = model_resnet_separable(example_input)
    print(f"Output shape (Separable Conv): {output_separable.shape}")