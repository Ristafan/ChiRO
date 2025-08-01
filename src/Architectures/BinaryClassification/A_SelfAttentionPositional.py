import math

import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        if x.dim() == 3:
            seq_len = x.size(0) if not self.position_embeddings.embedding_dim == x.size(2) else x.size(1) # More robust check needed if shapes vary


            # Create a tensor of positions (0, 1, ..., seq_len-1)
            # For batch_first=False:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(1).expand(-1, x.size(1)) # (seq_len, B)
            # For batch_first=True:
            # positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1) # (B, seq_len)

            pos_emb = self.position_embeddings(positions) # (seq_len, B, d_model) or (B, seq_len, d_model)

            return x + pos_emb
        else:
            raise ValueError("Input to LearnedPositionalEncoding must be 3D (S, B, E) or (B, S, E)")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model) for (S, N, E)
        self.register_buffer('pe', pe) # Register as buffer so it's saved with the model but not trained

    def forward(self, x):
        # Add positional encoding to the input embedding
        # Ensure pe's sequence dimension matches x's sequence dimension
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class AlphaSelfAttentionPositionalNet(nn.Module):
    def __init__(self, attention_heads=4, batch_norm=True, dropout_rate=0.1, final_pooling="avg", max_sequence_length=750, use_learned_positional_encoding=False):
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

        self.conv_blocks = nn.Sequential(
            conv_block(1, 32, kernel_size=3, batch_norm=batch_norm, dropout=dropout_rate),
            nn.MaxPool2d(kernel_size=(2, 2)),
            conv_block(32, 64, kernel_size=3, batch_norm=batch_norm, dropout=dropout_rate),
            nn.MaxPool2d(kernel_size=(2, 2)),
            conv_block(64, 128, kernel_size=3, batch_norm=batch_norm, dropout=dropout_rate)
        )

        # Positional Encoding Choice
        self.use_learned_positional_encoding = use_learned_positional_encoding
        if use_learned_positional_encoding:
            self.positional_encoder = LearnedPositionalEncoding(d_model=128, max_len=max_sequence_length)
        else:
            self.positional_encoder = PositionalEncoding(d_model=128, dropout=dropout_rate, max_len=max_sequence_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=attention_heads, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Pooling
        if final_pooling == "avg":
            self.pooling = nn.AdaptiveAvgPool1d(1)
        else:
            self.pooling = nn.AdaptiveMaxPool1d(1)

        # Classifier
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv_blocks(x)      # [B, 128, F', T']
        x = x.mean(dim=2)            # Collapse freq: [B, 128, T']
        x = x.permute(2, 0, 1)       # [T', B, 128]

        # Apply positional encoding
        x = self.positional_encoder(x)

        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)       # [B, 128, T']
        x = self.pooling(x)          # [B, 128, 1]
        x = x.view(x.size(0), -1)    # [B, 128]
        return self.classifier(x)
