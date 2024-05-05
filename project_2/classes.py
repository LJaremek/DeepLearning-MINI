import math

import torch.nn as nn
import torch


# ----------------------------------------
# Simple Model
# ----------------------------------------


class AudioModel(nn.Module):
    name = "AudioModel"

    def __init__(
            self,
            num_tokens: int, dim_model: int, num_heads: int,
            num_classes: int, dim_feedforward: int = 2048,
            num_layers: int = 1, dropout: int = 0.1
            ) -> None:

        super().__init__()
        self.embedding = nn.Embedding(num_tokens, dim_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout
            )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
            )

        self.fc = nn.Linear(dim_model, num_classes)

    def forward(self, src):
        src = self.embedding(src)  # Replace tokens with embeddings
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        output = self.fc(output)
        return output


# ----------------------------------------
# Model
# ----------------------------------------


class ImprovedAudioModel(nn.Module):
    name = "ImprovedAudioModel"

    def __init__(
            self,
            num_tokens: int, num_classes: int, embedding_dim: int = 256,
            rnn_units: int = 128, num_rnn_layers: int = 2, dropout: float = 0.3
            ) -> None:

        super(ImprovedAudioModel, self).__init__()
        self.embedding = nn.Embedding(num_tokens, embedding_dim)

        self.rnn = nn.GRU(
            embedding_dim, rnn_units, num_rnn_layers,
            batch_first=True, dropout=dropout
            )

        self.fc = nn.Linear(rnn_units, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # Taking the last output of sequences from RNN
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ----------------------------------------
# Transformer
# ----------------------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
            )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pe = self.pe[:x.size(0), :x.size(1)]
        x = x + pe
        return self.dropout(x)


class AudioTransformer(nn.Module):
    name = "AudioTransformer"

    def __init__(
            self,
            num_tokens: int,
            num_classes: int,
            dim_model: int = 256,
            num_heads: int = 8,
            num_layers: int = 3,
            dropout: float = 0.1,
            dim_feedforward: int = 512
            ) -> None:

        super(AudioTransformer, self).__init__()
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.positional_encoding = PositionalEncoding(dim_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model, nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
            )

        self.output_fc = nn.Linear(dim_model, num_classes)

    def forward(self, src):
        src = self.embedding(src)  # Convert token IDs to embeddings
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)  # Average pooling over sequence
        output = self.output_fc(output)
        return output
