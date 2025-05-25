import torch
import torch.nn as nn


class EnhancedLSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.0):
        super(EnhancedLSTMForecast, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout_prob if num_layers > 1 else 0)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=min(8, hidden_size // (
            hidden_size // 8 if hidden_size >= 8 else 1)), dropout=dropout_prob,
                                               batch_first=True)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.GELU()

    def forward(self, x):
        projected_x = self.input_projection(x)
        h0 = torch.zeros(self.num_layers, projected_x.size(0), self.hidden_size).to(projected_x.device)
        c0 = torch.zeros(self.num_layers, projected_x.size(0), self.hidden_size).to(projected_x.device)
        lstm_out, _ = self.lstm(projected_x, (h0, c0))

        if self.hidden_size % self.attention.num_heads != 0:
            attn_out = lstm_out
        else:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        x = self.layer_norm1(lstm_out + attn_out)
        final_hidden = x[:, -1, :]
        final_hidden = self.layer_norm2(final_hidden)
        out = self.activation(self.fc1(final_hidden))
        out = self.dropout(out)
        out = self.fc2(out)
        return out
