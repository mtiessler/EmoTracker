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
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout_prob, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.input_projection(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        x = self.layer_norm1(lstm_out + attn_out)
        final_hidden_state = x[:, -1, :]
        final_hidden_state = self.layer_norm2(final_hidden_state)
        out = self.activation(self.fc1(final_hidden_state))
        out = self.dropout(out)
        out = self.fc2(out)
        return out