"""Evaluasi: tabel metrik, uji Diebold-Mariano, dan seluruh gambar untuk PPT."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from common import FIG, RESULTS, add_features, load_raw, save_json

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
})
C = {
    "actual": "#111827", "gru": "#dc2626", "lstm": "#2563eb",
    "cnnlstm": "#059669", "persistence": "#9ca3af", "histgb": "#f59e0b",
    "sarimax": "#7c3aed", "seasonal": "#a3a3a3", "accent": "#f97316",
}
MODEL_LABEL = {
    "persistence": "Persistence", "seasonal_naive_lag_7": "Seasonal Naive",
    "sarimax1_0_11_0_0_7": "SARIMAX", "histgb_tabular": "HistGB",
    "lstm": "LSTM", "gru": "GRU", "cnn_lstm": "CNN-LSTM",
}
MODEL_COLOR = {
    "persistence": C["persistence"], "seasonal_naive_lag_7": C["seasonal"],
    "sarimax1_0_11_0_0_7": C["sarimax"], "histgb_tabular": C["histgb"],
    "lstm": C["lstm"], "gru": C["gru"], "cnn_lstm": C["cnnlstm"],
}


def fig_series(raw: pd.DataFrame) -> None:
    nat = raw.groupby("date")["hotspot_count"].sum()
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.plot(nat.index, nat.values, color="#cbd5e1", lw=0.7, label="Harian")
    ax.plot(nat.index, nat.rolling(7).mean(), color=C["accent"], lw=1.8,
            label="Rata-rata 7 hari")

    for start, end, label in [("2015-09-01", "2016-04-30", "El Niño 2015/16"),
                              ("2019-08-01", "2019-12-31", "El Niño lemah + IOD+ 2019"),
                              ("2023-07-01", "2024-03-31", "El Niño 2023/24")]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=C["gru"],
                   alpha=0.08)
        ax.text(pd.Timestamp(start), nat.max() * 0.94, label, fontsize=8,
                color="#7f1d1d", rotation=0)

    for split, txt in [("2021-12-24", "train | validasi"), ("2022-12-24", "validasi | uji")]:
        ax.axvline(pd.Timestamp(split), color="#111827", ls="--", lw=1.2)
        ax.text(pd.Timestamp(split), nat.max() * 0.55, txt, fontsize=8,
                rotation=90, va="top", ha="right")
    ax.set_title("Total titik panas harian nasional (10 provinsi), 2015–2025")
    ax.set_ylabel("Titik panas / hari")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig01_series.png", bbox_inches="tight")
    plt.close(fig)


def fig_climatology(raw: pd.DataFrame) -> None:
    prov_order = (raw.groupby("province")["hotspot_count"].mean()
                  .sort_values(ascending=False).index.tolist())
    piv = (raw.pivot_table(index="province", columns="month",
                           values="hotspot_count", aggfunc="mean")
           .reindex(prov_order))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4),
                            gridspec_kw={"width_ratios": [1.6, 1]})
    im = axes[0].imshow(piv.values, aspect="auto", cmap="YlOrRd")
    axes[0].set_xticks(range(12))
    axes[0].set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    axes[0].set_yticks(range(len(piv)))
    axes[0].set_yticklabels(piv.index, fontsize=9)
    axes[0].set_xlabel("Bulan")
    axes[0].set_title("Rata-rata titik panas per provinsi-bulan")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], fraction=0.03, label="Titik panas/hari")

    monthly = raw.groupby("month")["hotspot_count"].mean()
    axes[1].bar(monthly.index, monthly.values, color=C["accent"], alpha=0.85)
    axes[1].set_xticks(range(1, 13))
    axes[1].set_title("Rata-rata nasional per bulan")
    axes[1].set_xlabel("Bulan")
    axes[1].set_ylabel("Titik panas/hari/provinsi")
    fig.tight_layout()
    fig.savefig(FIG / "fig02_climatology.png", bbox_inches="tight")
    plt.close(fig)


def fig_drivers(feat: pd.DataFrame) -> None:
    cols = ["hotspot_count", "kbdi", "rainfall_mm", "rh_pct", "temp_c",
            "wind_ms", "oni", "iod_dmi", "y_lag1", "y_rmean7"]
    labels = ["Hotspot", "KBDI", "Hujan", "RH", "Suhu", "Angin", "ONI",
              "IOD", "Hotspot t-1", "Rata2 7h"]
    corr = feat[cols].corr()
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.55 else "#111827")
    ax.set_title("Korelasi fitur pendorong dengan hotspot")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(FIG / "fig03_drivers.png", bbox_inches="tight")
    plt.close(fig)


def fig_curves(hist: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, h in hist.items():
        key = {"LSTM": "lstm", "GRU": "gru", "CNN-LSTM": "cnn_lstm"}[name]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        axes[0].plot(ep, h["train_loss"], label=name, color=MODEL_COLOR[key], lw=1.8)
        axes[1].plot(ep, h["val_mae"], label=f"{name} (best ep {h['best_epoch']})",
                     color=MODEL_COLOR[key], lw=1.8)
        axes[1].scatter([h["best_epoch"]], [h["best_val_mae"]], s=45,
                        color=MODEL_COLOR[key], zorder=5, edgecolor="white")
    axes[0].set_title("Kurva latih (Huber loss)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[1].set_title("Kurva validasi (MAE, skala ternormalisasi)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE validasi")
    for ax in axes:
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig04_curves.png", bbox_inches="tight")
    plt.close(fig)


def fig_prediction(pred: dict) -> None:
    df = pd.DataFrame({
        "date": pd.to_datetime(pred["dates"]),
        "province": pred["provinces"],
        "y": pred["y_true"][:, 0],
        "gru": pred["gru"][:, 0],
        "lstm": pred["lstm"][:, 0],
        "cnn": pred["cnn_lstm"][:, 0],
    })
    d = df[(df.province == "Kalimantan Tengah") &
           (df.date >= "2023-07-01") & (df.date <= "2023-11-30")].sort_values("date")
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.plot(d.date, d.y, color=C["actual"], lw=2.4, label="Aktual")
    ax.plot(d.date, d.gru, color=C["gru"], lw=1.8, label="Prakiraan GRU (H+1)")
    ax.plot(d.date, d.cnn, color=C["cnnlstm"], lw=1.3, alpha=0.85,
            label="Prakiraan CNN-LSTM")
    ax.fill_between(d.date, 0, d.y, color="#e5e7eb", alpha=0.6, zorder=0)
    ax.set_title("Kalimantan Tengah — musim kemarau 2023 (periode uji)")
    ax.set_ylabel("Titik panas/hari")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig05_prediction.png", bbox_inches="tight")
    plt.close(fig)


def fig_metrics(m: dict) -> None:
    key_of = {v: k for k, v in MODEL_LABEL.items()}
    order = sorted(m, key=lambda k: m[k]["MAE"])
    labels = [MODEL_LABEL.get(key_of.get(k, k), k) for k in order]
    mae = [m[k]["MAE"] for k in order]
    rmse = [m[k]["RMSE"] for k in order]
    f1 = [m[k]["F1_event"] for k in order]
    cols = [MODEL_COLOR.get(key_of.get(k, k), "#6b7280") for k in order]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    ypos = np.arange(len(order))[::-1]
    axes[0].barh(ypos, mae, color=cols)
    axes[0].set_yticks(ypos); axes[0].set_yticklabels(labels)
    axes[0].set_title("MAE (lebih kecil = lebih baik)")
    for y, v in zip(ypos, mae):
        axes[0].text(v + 0.15, y, f"{v:.2f}", va="center", fontsize=8)
    axes[1].barh(ypos, rmse, color=cols)
    axes[1].set_yticks(ypos); axes[1].set_yticklabels([""] * len(labels))
    axes[1].set_title("RMSE")
    for y, v in zip(ypos, rmse):
        axes[1].text(v + 0.4, y, f"{v:.1f}", va="center", fontsize=8)
    axes[2].barh(ypos, f1, color=cols)
    axes[2].set_yticks(ypos); axes[2].set_yticklabels([""] * len(labels))
    axes[2].set_title("F1 deteksi kejadian (≥10 titik)")
    axes[2].set_xlim(0, 1)
    for y, v in zip(ypos, f1):
        axes[2].text(v + 0.015, y, f"{v:.2f}", va="center", fontsize=8)
    fig.suptitle("Perbandingan performa model, horizon H+1 (periode uji 2023–2025)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig06_metrics.png", bbox_inches="tight")
    plt.close(fig)


def fig_horizon(h: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    series = [("GRU", C["gru"]), ("CNN-LSTM", C["cnnlstm"]), ("LSTM", C["lstm"]),
              ("Persistence", C["persistence"]), ("Seasonal Naive (lag-7)", C["seasonal"])]
    offsets = {"GRU": -13, "CNN-LSTM": 8, "LSTM": 8,
               "Persistence": 8, "Seasonal Naive (lag-7)": -13}
    for label, color in series:
        sub = h[h.model == label].sort_values("horizon")
        if sub.empty:
            continue
        ax.plot(sub.horizon, sub.MAE, marker="o", lw=2, color=color, label=label)
        for _, r in sub.iterrows():
            ax.annotate(f"{r.MAE:.1f}", (r.horizon, r.MAE), textcoords="offset points",
                        xytext=(0, offsets.get(label, 7)), fontsize=8, ha="center",
                        color=color)
    ax.set_xticks([1, 3, 7])
    ax.set_xlim(0.5, 7.6)
    ax.set_xlabel("Horizon prakiraan (hari)")
    ax.set_ylabel("MAE")
    ax.set_title("Degradasi akurasi terhadap horizon prakiraan")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig07_horizon.png", bbox_inches="tight")
    plt.close(fig)


def fig_scatter(pred: dict) -> None:
    y = pred["y_true"][:, 0]
    p = np.clip(pred["gru"][:, 0], 0, None)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    hb = ax.hexbin(np.log1p(y), np.log1p(p), gridsize=45, cmap="magma_r", mincnt=1)
    lim = max(np.log1p(y).max(), np.log1p(p).max()) * 1.02
    ax.plot([0, lim], [0, lim], color="#2563eb", ls="--", lw=1.2, label="Garis ideal")
    ax.set_xlabel("log(1 + aktual)"); ax.set_ylabel("log(1 + prakiraan)")
    ax.set_title("GRU: sebaran prakiraan vs aktual (H+1)")
    ax.legend(frameon=False, fontsize=9)
    fig.colorbar(hb, ax=ax, fraction=0.045, label="jumlah observasi")

    ax = axes[1]
    err = p - y
    ax.hist(err, bins=70, color=C["gru"], alpha=0.8)
    ax.axvline(0, color="#111827", lw=1)
    ax.set_xlim(-60, 160)
    ax.set_xlabel("Galat (prakiraan − aktual), titik panas")
    ax.set_ylabel("Frekuensi")
    ax.set_title(f"Distribusi galat GRU (bias {err.mean():.2f})")
    fig.tight_layout()
    fig.savefig(FIG / "fig08_scatter.png", bbox_inches="tight")
    plt.close(fig)


def fig_province(per_prov: pd.DataFrame) -> None:
    piv = per_prov[per_prov.model.isin(["GRU", "Persistence"])].pivot(
        index="province", columns="model", values="MAE")
    piv["gain"] = (piv["Persistence"] - piv["GRU"]) / piv["Persistence"] * 100
    piv = piv.sort_values("gain")
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    colors = [C["cnnlstm"] if v > 20 else C["histgb"] for v in piv["gain"]]
    ax.barh(piv.index, piv["gain"], color=colors)
    for i, v in enumerate(piv["gain"]):
        ax.text(v + 0.4, i, f"{v:.0f}%", va="center", fontsize=9)
    ax.set_xlabel("Penurunan MAE dibanding persistence (%)")
    ax.set_title("Perbaikan akurasi GRU per provinsi")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig09_province.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    raw = load_raw()
    feat = add_features(raw)
    pred = np.load(RESULTS / "predictions_test.npz", allow_pickle=True)
    pred = {k: pred[k] for k in pred.files}
    metrics = json.loads((RESULTS / "metrics_h1.json").read_text())["metrics"]
    hist = json.loads((RESULTS / "training_history.json").read_text())
    per_prov = pd.read_csv(RESULTS / "metrics_per_province.csv")
    by_h = pd.read_csv(RESULTS / "metrics_by_horizon.csv")
    dm = pd.read_csv(RESULTS / "diebold_mariano.csv")

    print("Membuat gambar ...")
    fig_series(raw)
    fig_climatology(raw)
    fig_drivers(feat)
    fig_curves(hist)
    fig_prediction(pred)
    fig_metrics(metrics)
    fig_horizon(by_h)
    fig_scatter(pred)
    fig_province(per_prov)

    best = min(metrics, key=lambda k: metrics[k]["MAE"])
    pers = metrics["Persistence"]
    summary = {
        "test_period": [str(pd.to_datetime(pred["dates"]).min().date()),
                        str(pd.to_datetime(pred["dates"]).max().date())],
        "n_test_obs": int(len(pred["y_true"])),
        "best_model": best,
        "best_MAE": metrics[best]["MAE"],
        "persistence_MAE": pers["MAE"],
        "mae_reduction_pct": round((pers["MAE"] - metrics[best]["MAE"]) / pers["MAE"] * 100, 2),
        "metrics": metrics,
        "dm": dm.to_dict(orient="records"),
        "event_threshold": 10,
        "figures": sorted(p.name for p in FIG.glob("*.png")),
    }
    save_json(summary, RESULTS / "eval_summary.json")
    print(json.dumps({k: summary[k] for k in
                      ["best_model", "best_MAE", "mae_reduction_pct", "n_test_obs"]},
                     indent=2))
    print(f"Gambar tersimpan di {FIG}")


if __name__ == "__main__":
    main()
