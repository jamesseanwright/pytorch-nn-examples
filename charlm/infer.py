import operator
from onehot import one_hot_encoder
from model import LSTM
import numpy as np
import torch
from torch.nn import functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("charlm/shakespeare.txt") as f:
    text = f.read()

all_char = set(text)

model = LSTM(
    all_char,
    num_hidden=1024,
    num_layers=4,
    drop_prob=0.4,
).to(device)

model.load_state_dict(
    torch.load(
        "charlm/weights.pth",
        map_location=torch.device("cpu"),
    ),
)


def predict_next_char(char: str, hidden=None, k=2):
    encoded_text = one_hot_encoder(
        np.array([[model.encoder[char]]]),
        len(all_char),
    )

    inputs = torch.tensor(encoded_text).to(device)
    hidden = tuple([state.data for state in hidden])

    lstm_out, hidden = model.forward(inputs, hidden)

    probs = F.softmax(lstm_out, dim=1).data.cpu()
    probs, idx = probs.topk(k)
    idx = idx.numpy().squeeze()
    probs = probs.numpy().flatten()
    probs /= probs.sum()

    char = np.random.choice(idx, p=probs)

    return model.decoder[char], hidden


def generate_text(size: int, seed="The ", k=2):
    model.eval()

    output_char = [c for c in seed]
    hidden = model.init_hidden(1)

    for char in seed:
        char, hidden = predict_next_char(char, hidden, k)

    output_char.append(char)

    for i in range(size):
        char, hidden = predict_next_char(output_char[-1], hidden, k)
        output_char.append(char)

    return "".join(output_char)


print(generate_text(100, seed="When forty winters shall besiege ", k=8))
