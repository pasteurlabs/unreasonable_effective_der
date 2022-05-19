import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.optim as optim

from tqdm import tqdm


def get_best_device(fallback="cpu"):
    return torch.device("cuda:0" if torch.cuda.is_available() else fallback)


def get_ccycle(n, cmap="jet"):
    return [plt.get_cmap(cmap)(i) for i in np.linspace(0.0, 1.0, n)]


def scan(model, x_min, x_max, n=300, device=None):
    x = torch.linspace(x_min, x_max, n, device=device)

    if device:
        model = model.to(device)

    model.eval()
    with torch.no_grad():
        y = model(x.unsqueeze(1)).cpu().numpy()

    x = x.cpu().numpy()

    return x, y


def weighted_avg(x):
    nom = 0
    denom = 0
    for n, v in x:
        nom += n * v
        denom += n

    return nom / denom


def nop(*args, **kwargs):
    pass


def train_epoch(
    model, optimizer, *, train_dl, test_dl, loss_fct, error_fct=None, device=None
):
    if error_fct is None:
        error_fct = nop

    loss_train = []
    loss_test = []
    error_train = []
    error_test = []

    model = model.to(device)

    model.train()
    for xy in train_dl:
        optimizer.zero_grad()

        xy = xy.to(device)
        y_pred = model(xy[:, 0:1])
        loss = loss_fct(y_pred, xy[:, 1])

        n = y_pred.shape[0]

        loss_value = loss.detach().cpu().item()
        loss_train.append((n, loss_value))

        if error_fct is not None:
            error = error_fct(y_pred.detach(), xy[:, 1])
            error_value = error.cpu().item()
            error_train.append((n, error_value))

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for xy in test_dl:
            xy = xy.to(device)
            y_pred = model(xy[:, 0:1])
            loss = loss_fct(y_pred, xy[:, 1])

            error = error_fct(y_pred.detach(), xy[:, 1])

            n = y_pred.shape[0]

            loss_value = loss.detach().cpu().item()
            loss_test.append((n, loss_value))

            if error_fct is not None:
                error = error_fct(y_pred.detach(), xy[:, 1])
                error_value = error.cpu().item()
                error_test.append((n, error_value))

    loss_train = weighted_avg(loss_train)
    loss_test = weighted_avg(loss_test)

    if error_fct is not None:
        error_train = weighted_avg(error_train)
        error_test = weighted_avg(error_test)

    stats = {
        "loss_train": loss_train,
        "loss_test": loss_test,
        "error_train": None if error_fct is None else error_train,
        "error_test": None if error_fct is None else error_test,
    }

    return model, stats


def train(
    *,
    n_epochs,
    model,
    lr,
    loss_fct,
    error_fct,
    train_dl,
    test_dl,
    scan_lim,
    device=None
):
    model = torch.jit.script(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_train = []
    loss_test = []
    error_train = []
    error_test = []

    xy = {"x": None, "y": None}
    for _ in range(n_epochs):
        model, loss = train_epoch(
            model,
            optimizer,
            train_dl=train_dl,
            test_dl=test_dl,
            loss_fct=loss_fct,
            error_fct=error_fct,
            device=device,
        )

        loss_train.append(loss["loss_train"])
        loss_test.append(loss["loss_test"])
        error_train.append(loss["error_train"])
        error_test.append(loss["error_test"])

        x_min, x_max = scan_lim
        x, y = scan(model, x_min=x_min, x_max=x_max, device=device)
        xy["x"] = x

        if xy["y"] is None:
            xy["y"] = y[np.newaxis]
        else:
            xy["y"] = np.concatenate([xy["y"], y[np.newaxis]], axis=0)

    return (
        model,
        {
            "loss_train": loss_train,
            "loss_test": loss_test,
            "error_train": error_train,
            "error_test": error_test,
        },
        xy,
    )


def train_loop(f, *, n_samples, n_epochs, quiet=True):
    loss_train = []
    loss_test = []
    error_train = []
    error_test = []
    x_scan = None

    model = None

    for _ in tqdm(range(n_samples), disable=quiet):
        model, loss, xy = f(n_epochs=n_epochs)
        loss_train.append(loss["loss_train"])
        loss_test.append(loss["loss_test"])
        error_train.append(loss["error_train"])
        error_test.append(loss["error_test"])

        y = xy["y"][np.newaxis]
        if x_scan is None:
            x_scan = {"x": xy["x"], "y": y}
        else:
            x_scan["y"] = np.concatenate([x_scan["y"], y], axis=0)

    return (
        model,
        {
            "loss_train": loss_train,
            "loss_test": loss_test,
            "error_train": error_train,
            "error_test": error_test,
        },
        x_scan,
    )
