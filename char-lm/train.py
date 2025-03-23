import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def one_hot_encoder(encoded_text: NDArray[any], n_unique_char: int):
    one_hot = np.zeros((encoded_text.size, n_unique_char)).astype(np.float32)
    one_hot[np.arange(one_hot.shape[0]), encoded_text.flatten()] = 1.0
    one_hot = one_hot.reshape((*encoded_text.shape, n_unique_char))

    return one_hot


def generate_batches(encoded_text: NDArray[any], sample_per_batch=10, seq_len=50):
    char_per_batch = sample_per_batch * seq_len
    avail_batch = int(len(encoded_text) / char_per_batch)
    encoded_text = encoded_text[: char_per_batch * avail_batch]
    encoded_text = encoded_text.reshape((sample_per_batch, -1))

    for n in range(0, encoded_text.shape[1], seq_len):
        x = encoded_text[:, n : n + seq_len]
        y = np.zeros_like(x)

        try:
            y[:, :-1] = x[:, 1:]
            y[:, -1] = encoded_text[:, n + seq_len]
        except:
            y[:, :-1] = x[:, 1:]
            y[:, -1] = encoded_text[:, 0]

        yield x, y


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


with open("shakespeare.txt") as f:
    text = f.read()

model = LSTM(
    set(text),
    num_hidden=1024,
    num_layers=4,
    drop_prob=0.6,
).to(device)

encoded_text = model.encode_text(text)

optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train_percent = 0.9
train_ind = int(len(encoded_text) * train_percent)
train_data = encoded_text[:train_ind]
val_data = encoded_text[train_ind:]
num_epoch = 20
batch_size = 100
seq_len = 100
i = 0
validation_step_threshold = 25
num_char = max(encoded_text) + 1

model.train()

for epoch in range(num_epoch):
    hidden = model.init_hidden(batch_size)

    for x, y in generate_batches(val_data, batch_size, seq_len):
        i += 1
        model.zero_grad()
        x = one_hot_encoder(x, num_char)
        inputs = torch.tensor(x).to(device)
        targets = torch.tensor(y).long().to(device)
        hidden = tuple([state.data for state in hidden])

        lstm_out, hidden = model.forward(inputs, hidden)
        loss = criterion(lstm_out, targets.view(batch_size * seq_len))
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimiser.step()

        if i % validation_step_threshold == 0:
            model.eval()
            val_hidden = model.init_hidden(batch_size)
            val_losses = []

            for x, y in generate_batches(train_data, batch_size, seq_len):
                x = one_hot_encoder(x, num_char)
                inputs = torch.tensor(x).to(device)
                targets = torch.tensor(y).long().to(device)
                val_hidden = tuple([state.data for state in val_hidden])
                lstm_out, val_hidden = model.forward(inputs, val_hidden)
                loss = criterion(lstm_out, targets.view(batch_size * seq_len))
                val_losses.append(loss.item())

            print(f"epoch: {epoch + 1}, i: {i}, loss: {loss.item()}")
            model.train()

torch.save(model.state_dict(), "weights.pth")
