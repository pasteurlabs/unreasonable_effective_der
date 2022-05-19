import numpy as np


def generate_data(x_min, x_max, n, *, rng, train=True, dtype=None):
    if dtype is None:
        dtype = np.float32

    x = np.linspace(x_min, x_max, n)
    x = np.expand_dims(x, -1).astype(dtype)

    sigma = 3.0 * np.ones_like(x) if train else np.zeros_like(x)
    y = x**3 + rng.normal(0.0, sigma).astype(dtype)

    return np.concatenate((x, y), axis=1)
