import numpy as np


def pdf(x, width=0.05, normalize=True):
    height = 1.0 / width if normalize else 1.0
    return np.where(np.abs(x - 0.5) < 0.5 * width, height, 0.0)


def noise(x):
    return np.where(x < 0.5, 0.01, 0.1)


def generate_data(n, *, rng, train=True, dtype=None):
    x = np.linspace(0.0, 1.0, n)
    y = pdf(x, normalize=False)

    if train:
        y += rng.normal(0.0, noise(x), size=x.shape)

    return np.stack((x, y), axis=1).astype(dtype)
