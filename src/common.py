"""Komponen bersama: konfigurasi, rekayasa fitur, windowing, metrik, uji DM."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "karhutla_daily.csv"
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
MODELS = ROOT / "models"
ARCH_OUT = ROOT / "ppt"
for _p in (RESULTS, FIG, MODELS, ARCH_OUT):
    _p.mkdir(parents=True, exist_ok=True)

# Split kronologis (bukan acak) agar tidak ada kebocoran informasi masa depan.
TRAIN_END = "2021-12-31"
VAL_END = "2022-12-31"
TEST_END = "2025-12-31"

WINDOW = 14          # panjang lookback (hari)
HORIZONS = (1, 3, 7)  # horizon prediksi (hari ke depan)
EVENT_THRESHOLD = 10  # ambang "kejadian signifikan" (titik panas/hari/provinsi)
SEED = 42

FEATURES = [
    # histori hotspot (autokorelasi & memory kebakaran gambut)
    "y_lag1", "y_lag2", "y_lag3", "y_lag7", "y_lag14",
    "y_rmean7", "y_rmean14", "y_rmax7",
    # cuaca & kekeringan
    "rainfall_mm", "rain_lag1", "rain_rsum7", "kbdi", "kbdi_diff7",
    "kbdi_rmean7", "temp_c", "rh_pct", "wind_ms",
    # iklim skala besar
    "oni", "iod_dmi",
    # musiman
    "doy_sin", "doy_cos",
]


@dataclass
class Config:
    window: int = WINDOW
    horizons: tuple = HORIZONS
    batch_size: int = 256
    epochs: int = 60
    patience: int = 8
    lr: float = 2e-3
    hidden: int = 64
    dropout: float = 0.2
    seed: int = SEED

    def to_dict(self) -> dict:
        d = asdict(self)
        d["horizons"] = list(self.horizons)
        return d


# ---------------------------------------------------------------------------
# Data & fitur
# ---------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, parse_dates=["date"])
    return df.sort_values(["province", "date"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fitur yang SEMUA-nya dapat diketahui pada akhir hari t (anti-leakage)."""
    out = []
    for province, g in df.groupby("province", sort=False):
        g = g.sort_values("date").copy()
        y = g["hotspot_count"].astype(float)
        rain = g["rainfall_mm"].astype(float)

        g["y_lag1"] = y.shift(1)
        g["y_lag2"] = y.shift(2)
        g["y_lag3"] = y.shift(3)
        g["y_lag7"] = y.shift(7)
        g["y_lag14"] = y.shift(14)
        g["y_rmean7"] = y.shift(1).rolling(7).mean()
        g["y_rmean14"] = y.shift(1).rolling(14).mean()
        g["y_rmax7"] = y.shift(1).rolling(7).max()
        g["rain_lag1"] = rain.shift(1)
        g["rain_rsum7"] = rain.rolling(7).sum()
        g["kbdi_diff7"] = g["kbdi"].diff(7)
        g["kbdi_rmean7"] = g["kbdi"].rolling(7).mean()
        g["doy_sin"] = np.sin(2 * np.pi * g["doy"] / 365.25)
        g["doy_cos"] = np.cos(2 * np.pi * g["doy"] / 365.25)
        g["province_id"] = province
        out.append(g)

    feat = pd.concat(out).sort_values(["province", "date"]).reset_index(drop=True)
    feat = feat.dropna(subset=FEATURES).reset_index(drop=True)
    return feat


def split_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    d = df["date"]
    return {
        "train": d <= TRAIN_END,
        "val": (d > TRAIN_END) & (d <= VAL_END),
        "test": (d > VAL_END) & (d <= TEST_END),
    }


@dataclass
class ScaledData:
    features: np.ndarray      # (n_prov, T, F) terstandardisasi per provinsi
    targets: np.ndarray       # (n_prov, T, H) log1p terstandardisasi per provinsi
    province_ids: np.ndarray  # (n_prov,)
    dates: np.ndarray         # (T,) tanggal
    provinces: list[str]
    y_raw: np.ndarray         # (n_prov, T) hotspot asli
    feat_scaler: dict
    target_scaler: dict
    feature_names: list[str]


def scale_per_province(feat: pd.DataFrame, provinces: list[str]) -> ScaledData:
    """Standardisasi per provinsi memakai statistik TRAIN saja."""
    train_mask = (feat["date"] <= TRAIN_END).to_numpy()
    n_prov, T = len(provinces), feat[feat["province"] == provinces[0]].shape[0]

    X = np.zeros((n_prov, T, len(FEATURES)), dtype=np.float32)
    yraw = np.zeros((n_prov, T), dtype=np.float32)
    dates = None
    feat_scaler, target_scaler = {}, {}

    for i, p in enumerate(provinces):
        g = feat[feat["province"] == p]
        vals = g[FEATURES].to_numpy(dtype=np.float32)
        tr = (g["date"] <= TRAIN_END).to_numpy()

        mu = vals[tr].mean(axis=0)
        sd = vals[tr].std(axis=0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        X[i] = (vals - mu) / sd
        feat_scaler[p] = {"mean": mu.tolist(), "std": sd.tolist()}

        target = np.log1p(g["hotspot_count"].to_numpy(dtype=np.float32))
        tmu, tsd = target[tr].mean(), target[tr].std()
        tsd = tsd if tsd > 1e-6 else 1.0
        target_scaler[p] = {"mean": float(tmu), "std": float(tsd)}

        yraw[i] = g["hotspot_count"].to_numpy(dtype=np.float32)
        if dates is None:
            dates = g["date"].to_numpy()

    # target multi-horizon, diskalakan
    Y = np.full((n_prov, T, len(HORIZONS)), np.nan, dtype=np.float32)
    for i, p in enumerate(provinces):
        logy = np.log1p(yraw[i])
        sc = target_scaler[p]
        for j, h in enumerate(HORIZONS):
            shifted = np.full(T, np.nan, dtype=np.float32)
            shifted[:-h] = (logy[h:] - sc["mean"]) / sc["std"]
            Y[i, :, j] = shifted

    return ScaledData(X, Y, np.arange(n_prov), dates, provinces, yraw,
                      feat_scaler, target_scaler, FEATURES)


def build_windows(data: ScaledData, window: int):
    """Bentuk tensor window (N, window, F), id provinsi (N,), target (N, H), tanggal (N,)."""
    n_prov, T, _ = data.features.shape
    xs, pids, ys, ds, prov_idx = [], [], [], [], []
    for i in range(n_prov):
        for t in range(window - 1, T - 1):
            y = data.targets[i, t]
            if np.isnan(y).any():
                continue
            xs.append(data.features[i, t - window + 1: t + 1])
            pids.append(i)
            ys.append(y)
            ds.append(data.dates[t])
            prov_idx.append(i)
    return (np.stack(xs), np.array(pids, dtype=np.int64),
            np.stack(ys).astype(np.float32), np.array(ds), np.array(prov_idx))


def inverse_target(z: np.ndarray, province: str, scaler: dict) -> np.ndarray:
    sc = scaler[province]
    return np.expm1(z * sc["std"] + sc["mean"])


# ---------------------------------------------------------------------------
# Metrik & uji statistik
# ---------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(np.mean(np.where(denom < 1e-9, 0.0, 2 * np.abs(err) / np.maximum(denom, 1e-9))) * 100)

    t_event = y_true >= EVENT_THRESHOLD
    p_event = y_pred >= EVENT_THRESHOLD
    tp = int(np.sum(t_event & p_event))
    fp = int(np.sum(~t_event & p_event))
    fn = int(np.sum(t_event & ~p_event))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 4),
        "sMAPE_pct": round(smape, 2),
        "F1_event": round(f1, 4),
        "precision_event": round(precision, 4),
        "recall_event": round(recall, 4),
        "n_obs": int(len(y_true)),
    }


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 7) -> dict:
    """Uji DM (selisih kuadrat error) dengan koreksi Newey-West.

    H0: akurasi model A = model B. statistik negatif -> A lebih akurat.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 10:
        return {"dm_stat": float("nan"), "p_value": float("nan"), "n": n}
    dbar = d.mean()
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for lag in range(1, h):
        cov = np.mean((d[lag:] - dbar) * (d[:-lag] - dbar))
        var += 2 * (1 - lag / h) * cov
    var = max(var, 1e-12)
    stat = dbar / np.sqrt(var / n)
    # p-value dua sisi dari normal baku
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(stat) / sqrt(2))))
    return {"dm_stat": round(float(stat), 3), "p_value": round(float(p), 4), "n": int(n)}


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
