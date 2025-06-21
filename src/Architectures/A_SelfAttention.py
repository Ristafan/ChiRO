from torch import nn
import torch
from torch import nn


class SelfAttentionNet(nn.Module):
    def __init__(self, attention_heads=4, batch_norm=True, dropout_rate=0.1, final_pooling="avg"):
        super().__init__()

        def conv_block(in_channels, out_channels, kernel_size, batch_norm, dropout):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            return nn.Sequential(*layers)

        # Convolutional layers
        self.conv_blocks = nn.Sequential(
            conv_block(1, 32, kernel_size=3, batch_norm=batch_norm, dropout=dropout_rate),
            conv_block(32, 64, kernel_size=3, batch_norm=batch_norm, dropout=dropout_rate),
            conv_block(64, 128, kernel_size=3, batch_norm=batch_norm, dropout=dropout_rate)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=attention_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Pooling
        if final_pooling == "avg":
            self.pooling = nn.AdaptiveAvgPool1d(1)
        else:
            self.pooling = nn.AdaptiveMaxPool1d(1)

        # Classifier
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv_blocks(x)      # [B, 128, F, T]
        x = x.mean(dim=2)            # Collapse freq: [B, 128, T]
        x = x.permute(0, 2, 1)       # [B, T, 128]
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)       # [B, 128, T]
        x = self.pooling(x)          # [B, 128, 1]
        x = x.view(x.size(0), -1)    # [B, 128]
        return self.classifier(x)

