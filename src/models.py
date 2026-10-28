"""Arsitektur deep learning untuk prakiraan hotspot karhutla."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ProvinceEmbedding(nn.Module):
    def __init__(self, n_provinces: int, dim: int = 8):
        super().__init__()
        self.emb = nn.Embedding(n_provinces, dim)

    def forward(self, pid):
        return self.emb(pid)


class LSTMRegressor(nn.Module):
    """LSTM 2 lapis (many-to-one) + embedding provinsi + head multi-horizon."""

    def __init__(self, n_features: int, n_provinces: int, hidden: int = 64,
                 n_horizons: int = 3, dropout: float = 0.2):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True,
                           dropout=dropout)
        self.prov = ProvinceEmbedding(n_provinces)
        self.head = nn.Sequential(
            nn.Linear(hidden + 8, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_horizons),
        )

    def forward(self, x, pid):
        out, _ = self.rnn(x)
        h = torch.cat([out[:, -1], self.prov(pid)], dim=-1)
        return self.head(h)


class GRURegressor(nn.Module):
    """GRU 2 lapis, unit lebih sedikit (lebih ringan) sebagai pembanding."""

    def __init__(self, n_features: int, n_provinces: int, hidden: int = 64,
                 n_horizons: int = 3, dropout: float = 0.2):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden, num_layers=2, batch_first=True,
                          dropout=dropout)
        self.prov = ProvinceEmbedding(n_provinces)
        self.head = nn.Sequential(
            nn.Linear(hidden + 8, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_horizons),
        )

    def forward(self, x, pid):
        out, _ = self.rnn(x)
        h = torch.cat([out[:, -1], self.prov(pid)], dim=-1)
        return self.head(h)


class CNNLSTMRegressor(nn.Module):
    """Conv1D (ekstraksi pola lokal) -> LSTM (dinamika temporal) -> head."""

    def __init__(self, n_features: int, n_provinces: int, hidden: int = 64,
                 n_horizons: int = 3, dropout: float = 0.2, filters: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, filters, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(filters, filters, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.rnn = nn.LSTM(filters, hidden, num_layers=1, batch_first=True)
        self.prov = ProvinceEmbedding(n_provinces)
        self.head = nn.Sequential(
            nn.Linear(hidden + 8, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_horizons),
        )

    def forward(self, x, pid):
        z = self.conv(x.transpose(1, 2))     # (B, filters, L)
        z = self.pool(z)
        out, _ = self.rnn(z.transpose(1, 2))  # (B, L/2, hidden)
        h = torch.cat([out[:, -1], self.prov(pid)], dim=-1)
        return self.head(h)


def build_model(name: str, n_features: int, n_provinces: int, cfg) -> nn.Module:
    kwargs = dict(n_features=n_features, n_provinces=n_provinces,
                  hidden=cfg.hidden, n_horizons=len(cfg.horizons),
                  dropout=cfg.dropout)
    if name == "LSTM":
        return LSTMRegressor(**kwargs)
    if name == "GRU":
        return GRURegressor(**kwargs)
    if name == "CNN-LSTM":
        return CNNLSTMRegressor(**kwargs)
    raise ValueError(f"model tidak dikenal: {name}")


def train_model(model, train_data, val_data, cfg, verbose: bool = True):
    """Latih dengan early stopping pada MAE (h=1) di set validasi."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loss_fn = nn.SmoothL1Loss(beta=0.5)   # Huber: tahan terhadap lonjakan ekstrem

    x_tr, p_tr, y_tr = train_data
    x_va, p_va, y_va = val_data
    n = len(x_tr)
    history = {"train_loss": [], "val_mae": []}
    best = {"val_mae": float("inf"), "state": None, "epoch": 0}
    patience_left = cfg.patience

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb, pb, yb = x_tr[idx], p_tr[idx], y_tr[idx]
            opt.zero_grad()
            pred = model(xb, pb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(idx)
        ep_loss /= n

        model.eval()
        with torch.no_grad():
            pred_val = torch.cat([model(x_va[i:i + 1024], p_va[i:i + 1024])
                                  for i in range(0, len(x_va), 1024)])
            val_mae = float(torch.mean(torch.abs(pred_val[:, 0] - y_va[:, 0])))

        history["train_loss"].append(round(ep_loss, 5))
        history["val_mae"].append(round(val_mae, 5))
        sched.step(val_mae)

        if val_mae < best["val_mae"] - 1e-5:
            best = {"val_mae": val_mae, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
            patience_left = cfg.patience
        else:
            patience_left -= 1

        if verbose:
            print(f"    epoch {epoch:02d} | train {ep_loss:.4f} | val MAE {val_mae:.4f}")
        if patience_left <= 0:
            if verbose:
                print(f"    early stop di epoch {epoch}")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    history["best_epoch"] = best["epoch"]
    history["best_val_mae"] = round(best["val_mae"], 5)
    history["epochs_run"] = len(history["train_loss"])
    return model, history


@torch.no_grad()
def predict(model, x, pid, batch: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        out.append(model(x[i:i + batch], pid[i:i + batch]).numpy())
    return np.concatenate(out, axis=0)
