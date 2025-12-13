import torch
import torch.nn as nn

import config


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=config.HIDDEN_DIM, latent_dim=config.LATENT_DIM, num_layers=config.NUM_LAYERS):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2 if num_layers>1 else 0)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2 if num_layers>1 else 0)
        self.output_fc = nn.Linear(hidden_dim, input_dim)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        _, (hidden, cell) = self.encoder_lstm(x)
        latent = self.encoder_fc(hidden[-1])
        decoded = torch.tanh(self.decoder_fc(latent)).unsqueeze(1).repeat(1, seq_len, 1)
        h0 = decoded.mean(dim=1).unsqueeze(0).repeat(self.num_layers, 1, 1).contiguous()
        c0 = torch.zeros_like(h0)
        output, _ = self.decoder_lstm(decoded, (h0, c0))
        output = self.output_fc(output)
        return output, latent