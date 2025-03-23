import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from model import LSTM
from onehot import one_hot_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


with open("charlm/shakespeare.txt") as f:
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
train_ind = int(len(encoded_text) * (train_percent))
train_data = encoded_text[:train_ind]
eval_data = encoded_text[train_ind:]
num_epoch = 20
batch_size = 100
seq_len = 100
i = 0
validation_step_threshold = 50
num_char = max(encoded_text) + 1

model.train()

for epoch in range(num_epoch):
    hidden = model.init_hidden(batch_size)

    for x, y in generate_batches(train_data, batch_size, seq_len):
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

            for x, y in generate_batches(eval_data, batch_size, seq_len):
                x = one_hot_encoder(x, num_char)
                inputs = torch.tensor(x).to(device)
                targets = torch.LongTensor(y).to(device)
                val_hidden = tuple([state.data for state in val_hidden])
                lstm_out, val_hidden = model.forward(inputs, val_hidden)
                loss = criterion(lstm_out, targets.view(batch_size * seq_len).long())
                val_losses.append(loss.item())

            print(f"epoch: {epoch + 1}, i: {i}, loss: {loss.item()}")
            model.train()

torch.save(model.state_dict(), "charlm/weights.pth")
