import math

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_output, n_hidden=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )

    def forward(self, x):
        return self.model(x)


class DERLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        gamma = x[:, 0]
        nu = nn.functional.softplus(x[:, 1])
        alpha = nn.functional.softplus(x[:, 2]) + 1.0
        beta = nn.functional.softplus(x[:, 3])
        return torch.stack((gamma, nu, alpha, beta), dim=1)


class SDERLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        gamma = x[:, 0]
        nu = nn.functional.softplus(x[:, 1])
        alpha = nu + 1.0
        beta = nn.functional.softplus(x[:, 3])
        return torch.stack((gamma, nu, alpha, beta), dim=1)


def loss_der(y, y_pred, coeff):
    gamma, nu, alpha, beta = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    error = gamma - y_pred
    omega = 2.0 * beta * (1.0 + nu)

    return torch.mean(
        0.5 * torch.log(math.pi / nu)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log(error**2 * nu + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
        + coeff * torch.abs(error) * (2.0 * nu + alpha)
    )


def loss_sder(y, y_pred, coeff):
    gamma, nu, _, beta = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    error = gamma - y_pred
    var = beta / nu

    return torch.mean(torch.log(var) + (1. + coeff * nu) * error**2 / var)


def loss_gaussian(y, y_pred, coeff):
    gamma, _, _, var = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    error = gamma - y_pred

    return torch.mean(torch.log(var) + error**2 / var)
