import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    def __init__(
        self,
        all_char: set[str],
        num_hidden=512,
        num_layers=3,
        drop_prob=0.5,
    ):
        super(LSTM, self).__init__()

        self.all_char = all_char
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.drop_prob = drop_prob

        self.decoder = dict(enumerate(all_char))
        self.encoder = {char: idx for idx, char in self.decoder.items()}

        self.lstm = nn.LSTM(
            len(all_char), num_hidden, num_layers, dropout=drop_prob, batch_first=True
        )

        self.fc_linear = nn.Linear(num_hidden, len(self.all_char))
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: NDArray[any], hidden: tuple):
        lstm_out, hidden = self.lstm(x, hidden)
        dropout = self.dropout(lstm_out).contiguous().view(-1, self.num_hidden)
        return self.fc_linear(dropout), hidden

    def init_hidden(self, batch_size: int):
        return (
            torch.zeros(self.num_layers, batch_size, self.num_hidden).to(device),
            torch.zeros(self.num_layers, batch_size, self.num_hidden).to(device),
        )

    def encode_text(self, text: str) -> NDArray[any]:
        return np.array([self.encoder[char] for char in text])
