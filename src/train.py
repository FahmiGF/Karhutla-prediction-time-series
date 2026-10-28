"""Latih baseline statistik/ML dan model deep learning, simpan metrik & prediksi."""

from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
import torch

from common import (FEATURES, HORIZONS, MODELS, RESULTS, SEED, TRAIN_END,
                    VAL_END, Config, add_features, build_windows,
                    diebold_mariano, load_raw, regression_metrics, save_json,
                    scale_per_province)
from models import build_model, train_model

warnings.filterwarnings("ignore")

# Batas input terakhir tiap fold (jeda 7 hari = horizon maksimum, mencegah bocor)
TRAIN_LAST_INPUT = pd.Timestamp(TRAIN_END) - pd.Timedelta(days=7)
VAL_LAST_INPUT = pd.Timestamp(VAL_END) - pd.Timedelta(days=7)
SARIMAX_EXOG = ["kbdi", "rain_lag1", "oni", "iod_dmi"]


def _series_lookup(feat: pd.DataFrame, provinces: list[str]):
    """Peta provinsi -> (array nilai, indeks tanggal) untuk baseline naive."""
    lookup = {}
    for prov in provinces:
        g = feat[feat["province"] == prov].reset_index(drop=True)
        lookup[prov] = {
            "values": g["hotspot_count"].to_numpy(dtype=float),
            "pos": {d: i for i, d in enumerate(g["date"].to_numpy())},
        }
    return lookup


def main() -> None:
    t0 = time.time()
    cfg = Config()

    # ---------------------------------------------------------------- data
    print("[1/6] Memuat data & rekayasa fitur ...")
    raw = load_raw()
    feat = add_features(raw)
    provinces = sorted(feat["province"].unique().tolist())
    data = scale_per_province(feat, provinces)
    x, pid, y, dates, _ = build_windows(data, cfg.window)

    d = pd.to_datetime(dates)
    m_train = d <= TRAIN_LAST_INPUT
    m_val = (d > TRAIN_LAST_INPUT) & (d <= VAL_LAST_INPUT)
    m_test = d > VAL_LAST_INPUT
    print(f"    window {len(x):,} | train {m_train.sum():,} | "
          f"val {m_val.sum():,} | test {m_test.sum():,}")

    xt = torch.tensor(x, dtype=torch.float32)
    pt = torch.tensor(pid, dtype=torch.long)
    yt = torch.tensor(y, dtype=torch.float32)
    idx_train, idx_val, idx_test = (np.where(m)[0] for m in (m_train, m_val, m_test))
    train_data = (xt[idx_train], pt[idx_train], yt[idx_train])
    val_data = (xt[idx_val], pt[idx_val], yt[idx_val])

    test_dates = dates[m_test]
    test_pid = pid[m_test]
    test_prov = np.array([provinces[i] for i in test_pid])
    y_true_z = y[m_test]

    # target mentah (skala asli) per baris test
    y_true_raw = np.zeros_like(y_true_z, dtype=float)
    for j, p in enumerate(test_pid):
        sc = data.target_scaler[provinces[p]]
        y_true_raw[j] = np.expm1(y_true_z[j].astype(float) * sc["std"] + sc["mean"])

    def to_raw(z: np.ndarray) -> np.ndarray:
        out = np.zeros_like(z, dtype=float)
        for j, p in enumerate(test_pid):
            sc = data.target_scaler[provinces[p]]
            out[j] = np.expm1(z[j].astype(float) * sc["std"] + sc["mean"])
        return np.clip(out, 0, None)

    lookup = _series_lookup(feat, provinces)
    preds: dict[str, np.ndarray] = {}
    histories: dict[str, dict] = {}

    # ------------------------------------------- baseline naive (semua horizon)
    print("[2/6] Baseline persistence & seasonal naive ...")
    persistence = np.zeros_like(y_true_raw)
    seasonal = np.zeros_like(y_true_raw)
    for j, prov in enumerate(test_prov):
        info = lookup[prov]
        pos = info["pos"][test_dates[j]]
        for hi, h in enumerate(HORIZONS):
            persistence[j, hi] = info["values"][pos]
            seasonal[j, hi] = (info["values"][pos + h - 7]
                               if pos + h - 7 >= 0 else info["values"][pos])
    preds["Persistence"] = persistence
    preds["Seasonal Naive (lag-7)"] = seasonal

    # --------------------------------------------- baseline SARIMAX (h=1 saja)
    print("[3/6] Baseline SARIMAX per provinsi ...")
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    sarimax_series: dict[str, pd.Series] = {}
    for prov in provinces:
        g = feat[feat["province"] == prov].reset_index(drop=True)
        ylog = np.log1p(g["hotspot_count"].to_numpy(dtype=float))
        exog = g[SARIMAX_EXOG].to_numpy(dtype=float)
        tr = (g["date"] <= TRAIN_LAST_INPUT).to_numpy()
        try:
            res = SARIMAX(ylog[tr], exog=exog[tr], order=(1, 0, 1),
                          seasonal_order=(1, 0, 0, 7), trend="c",
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False, maxiter=100)
            full = res.apply(endog=ylog, exog=exog)
            pred_log = np.asarray(
                full.get_prediction(start=0, end=len(ylog) - 1).predicted_mean)
            sarimax_series[prov] = pd.Series(np.expm1(pred_log), index=g["date"])
        except Exception as exc:  # fallback bila optimisasi gagal
            print(f"    ! SARIMAX {prov} gagal ({exc}); fallback persistence")
            sarimax_series[prov] = pd.Series(g["hotspot_count"].to_numpy(dtype=float),
                                             index=g["date"])

    sarimax_pred = np.full_like(y_true_raw, np.nan)
    for j, prov in enumerate(test_prov):
        sarimax_pred[j, 0] = sarimax_series[prov].loc[test_dates[j]]
    preds["SARIMAX(1,0,1)(1,0,0,7)"] = np.clip(sarimax_pred, 0, None)

    # --------------------------------------- baseline HistGB tabular (h=1 saja)
    print("[4/6] Baseline HistGradientBoosting ...")
    from sklearn.ensemble import HistGradientBoostingRegressor

    Xtab = np.zeros((len(x), len(FEATURES) + len(provinces)), dtype=np.float32)
    Xtab[:, :len(FEATURES)] = x[:, -1, :]
    Xtab[np.arange(len(x)), len(FEATURES) + pid] = 1.0
    hgb = HistGradientBoostingRegressor(
        max_iter=800, learning_rate=0.05, max_depth=6, min_samples_leaf=20,
        l2_regularization=1e-3, early_stopping=True, validation_fraction=0.1,
        random_state=SEED)
    hgb.fit(Xtab[idx_train], y[idx_train, 0])
    hgb_z = np.full_like(y_true_raw, np.nan)
    hgb_z[:, 0] = hgb.predict(Xtab[idx_test])
    preds["HistGB (tabular)"] = to_raw(hgb_z)
    preds["HistGB (tabular)"][:, 1:] = np.nan

    # ------------------------------------------------- model deep learning
    print("[5/6] Melatih model deep learning ...")
    for name in ["LSTM", "GRU", "CNN-LSTM"]:
        print(f"  -> {name}")
        model = build_model(name, n_features=x.shape[2],
                            n_provinces=len(provinces), cfg=cfg)
        model, hist = train_model(model, train_data, val_data, cfg)
        histories[name] = hist
        torch.save(model.state_dict(), MODELS / f"{name.replace('-', '_').lower()}.pt")

        pred_z = np.zeros((len(idx_test), len(HORIZONS)), dtype=np.float32)
        model.eval()
        with torch.no_grad():
            for s in range(0, len(idx_test), 2048):
                sl = idx_test[s:s + 2048]
                pred_z[s:s + 2048] = model(xt[sl], pt[sl]).numpy()
        preds[name] = to_raw(pred_z)

    # --------------------------------------------------------- metrik & uji
    print("[6/6] Metrik & uji Diebold-Mariano ...")
    metrics_h1, per_prov_rows, horizon_rows = {}, [], []
    for name, pred in preds.items():
        metrics_h1[name] = regression_metrics(y_true_raw[:, 0], pred[:, 0])
        for prov in provinces:
            m = test_prov == prov
            per_prov_rows.append({"model": name, "province": prov,
                                  **regression_metrics(y_true_raw[m, 0], pred[m, 0])})
        for hi, h in enumerate(HORIZONS):
            if np.isnan(pred[:, hi]).any():
                continue
            horizon_rows.append({"model": name, "horizon": h,
                                 **regression_metrics(y_true_raw[:, hi], pred[:, hi])})

    base_loss = (preds["Persistence"][:, 0] - y_true_raw[:, 0]) ** 2
    dm_rows = []
    for name, pred in preds.items():
        if name == "Persistence":
            continue
        loss = (pred[:, 0] - y_true_raw[:, 0]) ** 2
        res = diebold_mariano(loss, base_loss, h=7)
        dm_rows.append({"model": name, "reference": "Persistence", **res,
                        "lebih_baik": bool(res["dm_stat"] < 0 and res["p_value"] < 0.05)})

    dl_models = ["LSTM", "GRU", "CNN-LSTM"]
    best_non_dl = min([k for k in preds if k not in dl_models],
                      key=lambda k: metrics_h1[k]["MAE"])
    ref_loss = (preds[best_non_dl][:, 0] - y_true_raw[:, 0]) ** 2
    for name in dl_models:
        loss = (preds[name][:, 0] - y_true_raw[:, 0]) ** 2
        res = diebold_mariano(loss, ref_loss, h=7)
        dm_rows.append({"model": name, "reference": best_non_dl, **res,
                        "lebih_baik": bool(res["dm_stat"] < 0 and res["p_value"] < 0.05)})

    np.savez_compressed(
        RESULTS / "predictions_test.npz",
        y_true=y_true_raw.astype(np.float32), provinces=test_prov,
        dates=test_dates.astype("datetime64[D]").astype(str),
        horizons=np.array(HORIZONS),
        **{k.replace("-", "_").replace("(", "").replace(")", "")
           .replace(",", "_").replace(" ", "_").lower(): v for k, v in preds.items()})
    save_json({"horizons": list(HORIZONS), "metrics": metrics_h1,
               "best_non_dl": best_non_dl,
               "model_order": {k.replace("-", "_"): k for k in preds}},
              RESULTS / "metrics_h1.json")
    pd.DataFrame(per_prov_rows).to_csv(RESULTS / "metrics_per_province.csv", index=False)
    pd.DataFrame(horizon_rows).to_csv(RESULTS / "metrics_by_horizon.csv", index=False)
    pd.DataFrame(dm_rows).to_csv(RESULTS / "diebold_mariano.csv", index=False)
    save_json(histories, RESULTS / "training_history.json")
    save_json(cfg.to_dict(), RESULTS / "config.json")

    print("\n=== MAE h=1 (titik panas/hari/provinsi) ===")
    for name, m in sorted(metrics_h1.items(), key=lambda kv: kv[1]["MAE"]):
        print(f"  {name:26s} MAE {m['MAE']:8.3f} | RMSE {m['RMSE']:8.3f} | "
              f"R2 {m['R2']:6.3f} | F1 {m['F1_event']:.3f}")
    print(f"\nSelesai dalam {time.time() - t0:.1f} detik")


if __name__ == "__main__":
    main()
