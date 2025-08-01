import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaV1_Attention(nn.Module):
    def __init__(self, dropout_rate=0.3, batch_norm=True, num_classes=2, embed_dim=128, num_heads=8):
        super(AlphaV1_Attention, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(32) if batch_norm else None

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64) if batch_norm else None

        self.conv3 = nn.Conv2d(64, embed_dim, kernel_size=3, stride=1, padding=1)
        self.dropout_cnn = nn.Dropout(dropout_rate)

        # Class Token
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Attention Layer
        self.attention_layer = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=False)

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout_attn = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(embed_dim, 64) # Input is `embed_dim` from the attended class token
        self.fc2 = nn.Linear(64, num_classes) # Output is `num_classes`
        self.dropout_fc = nn.Dropout(dropout_rate)

    def forward(self, x):
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
        x = self.pool(x) # Output shape: [batch_size, embed_dim, H_pooled, W_pooled]
        x = self.dropout_cnn(x) # Apply dropout after CNN features

        # Prepare Features for Attention
        batch_size, channels, H_pooled, W_pooled = x.shape

        # Flatten the spatial dimensions (H_pooled * W_pooled) into a sequence of tokens.
        feature_tokens = x.view(batch_size, channels, -1).permute(0, 2, 1) # [B, N, embed_dim] where N = H_pooled * W_pooled

        # Incorporate Class Token
        class_token_expanded = self.class_token.expand(batch_size, -1, -1)

        # Concatenate the expanded class token with the feature tokens along the sequence dimension.
        x_sequence = torch.cat((class_token_expanded, feature_tokens), dim=1)

        # Apply Layer Normalization
        x_norm = self.norm(x_sequence)

        # Cross-Attention Mechanism
        query = x_norm[:, 0:1, :].permute(1, 0, 2)
        key_value = x_norm[:, 1:, :].permute(1, 0, 2)

        # attn_output will be the attended class token: [1, B, embed_dim]
        attn_output, attn_weights = self.attention_layer(query, key_value, key_value)

        # Extract the attended class token (it's the only item in attn_output).
        class_token_out = attn_output[0]
        class_token_out = self.dropout_attn(class_token_out) # Apply dropout

        # Classifier
        x = F.relu(self.fc1(class_token_out))
        x = self.dropout_fc(x) # Apply dropout after first FC layer
        x = self.fc2(x) # Final output for classification

        return x


if __name__ == '__main__':
    model = AlphaV1_Attention()
    print(model)
