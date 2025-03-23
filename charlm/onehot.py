import numpy as np
from numpy.typing import NDArray


def one_hot_encoder(encoded_text: NDArray[any], n_unique_char: int):
    one_hot = np.zeros((encoded_text.size, n_unique_char)).astype(np.float32)
    one_hot[np.arange(one_hot.shape[0]), encoded_text.flatten()] = 1.0
    one_hot = one_hot.reshape((*encoded_text.shape, n_unique_char))

    return one_hot
