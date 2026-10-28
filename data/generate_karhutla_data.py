"""
Generator dataset harian karhutla (kebakaran hutan dan lahan) Indonesia.

PENTING — SIFAT DATA
--------------------
Dataset ini adalah SIMULASI yang dibangkitkan secara fisik-meteorologis, bukan
data observasi. Tujuannya: menyediakan data deret waktu yang panjang, konsisten,
dan punya pola realistis (musim kemarau, episodik El Nino, overdispersion
hotspot, kelembapan gambut) sehingga seluruh pipeline deep learning dan
evaluasinya dapat dijalankan dan diaudit tanpa bergantung pada API berbayar.

Skema pembangkitan mengikuti rantai kausal yang dipakai literatur karhutla:
    ENSO (ONI) -> curah hujan -> suhu/RH/angin -> KBDI (kekeringan bahan bakar)
    -> intensitas hotspot (NegBin, autokorelasi spasial-temporal)

Untuk data asli, unduh hotspot VIIRS/MODIS dari NASA FIRMS
(https://firms.modaps.eosdis.nasa.gov/api/) lalu ganti `data/karhutla_daily.csv`
dengan agregasi harian per provinsi memakai nama kolom yang sama. Pipeline
pelatihan & evaluasi tidak perlu diubah.

Keluaran: data/karhutla_daily.csv (+ data/data_dictionary.md)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START = "2015-01-01"
END = "2025-12-31"
OUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# ONI (Oceanic Nino Index, anomaly SST Nino 3.4) bulanan, pendekatan NOAA.
# Nilai >= +0.5 = El Nino, <= -0.5 = La Nina.
# ---------------------------------------------------------------------------
ONI = {
    2015: [-0.5, -0.4, -0.1, 0.2, 0.5, 0.8, 1.1, 1.5, 1.9, 2.2, 2.4, 2.5],
    2016: [2.5, 2.2, 1.7, 1.1, 0.6, 0.2, -0.2, -0.4, -0.6, -0.7, -0.8, -0.7],
    2017: [-0.4, -0.2, 0.1, 0.3, 0.4, 0.4, 0.3, 0.0, -0.3, -0.6, -0.8, -0.9],
    2018: [-0.9, -0.8, -0.6, -0.4, -0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.8],
    2019: [0.7, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1, 0.0, 0.2, 0.5, 0.6],
    2020: [0.6, 0.5, 0.4, 0.1, -0.2, -0.4, -0.4, -0.5, -0.7, -0.9, -1.1, -1.2],
    2021: [-1.1, -1.0, -0.8, -0.6, -0.4, -0.3, -0.3, -0.4, -0.6, -0.8, -0.9, -1.0],
    2022: [-1.0, -0.9, -0.9, -0.8, -0.7, -0.6, -0.5, -0.5, -0.6, -0.8, -0.9, -0.8],
    2023: [-0.7, -0.4, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.7, 1.9, 2.0, 2.0],
    2024: [1.8, 1.5, 1.1, 0.7, 0.4, 0.2, 0.0, -0.1, -0.3, -0.4, -0.5, -0.6],
    2025: [-0.6, -0.5, -0.3, -0.2, -0.1, -0.1, -0.1, -0.2, -0.3, -0.4, -0.4, -0.4],
}

# Dipole Mode Index (IOD) bulanan, pendekatan NOAA/BoM.
# IOD positif kuat (2019, 2023) menekan hujan Indonesia bagian barat -> memperparah karhutla.
IOD = {
    2015: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.6, 0.7, 0.8, 0.7],
    2016: [0.4, 0.2, 0.0, -0.2, -0.4, -0.5, -0.6, -0.6, -0.5, -0.4, -0.3, -0.2],
    2017: [-0.1, 0.0, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.2],
    2018: [-0.3, -0.4, -0.3, -0.1, 0.1, 0.3, 0.5, 0.6, 0.5, 0.4, 0.2, 0.1],
    2019: [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.2, 1.1, 0.8, 0.5],
    2020: [0.3, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.4, -0.3, -0.2, -0.1, 0.0],
    2021: [0.0, 0.1, 0.2, 0.2, 0.1, 0.0, -0.1, -0.2, -0.2, -0.2, -0.1, -0.1],
    2022: [-0.2, -0.3, -0.4, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.1, 0.0, -0.1],
    2023: [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 0.9, 0.7, 0.5, 0.3],
    2024: [0.1, -0.1, -0.2, -0.3, -0.4, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.1],
    2025: [0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.0, -0.1, -0.1, -0.1],
}

# Parameter iklim & kerentanan kebakaran per provinsi.
# dry_peak      : hari ke- (1-365) puncak musim kemarau
# monsoon_amp   : amplitudo musiman curah hujan (0-1), kecil = hampir sepanjang tahun hujan
# wet_rain_mm   : rata-rata hujan harian musim basah
# dry_rain_mm   : rata-rata hujan harian musim kering
# base_temp_c   : suhu rata-rata tahunan
# peat          : dominasi lahan gambut -> kebakaran bawah permukaan, lebih persisten
# fire_base     : skala kerentanan hotspot relatif
PROVINCES = {
    "Riau": dict(island="Sumatera", dry_peak=238, monsoon_amp=0.55, wet_rain_mm=10.5,
                 dry_rain_mm=3.2, base_temp_c=27.2, peat=True, fire_base=0.30),
    "Jambi": dict(island="Sumatera", dry_peak=243, monsoon_amp=0.50, wet_rain_mm=10.0,
                  dry_rain_mm=3.6, base_temp_c=27.0, peat=True, fire_base=0.18),
    "Sumatera Selatan": dict(island="Sumatera", dry_peak=246, monsoon_amp=0.58, wet_rain_mm=10.8,
                             dry_rain_mm=3.0, base_temp_c=27.4, peat=True, fire_base=0.38),
    "Lampung": dict(island="Sumatera", dry_peak=236, monsoon_amp=0.62, wet_rain_mm=9.5,
                    dry_rain_mm=2.4, base_temp_c=27.6, peat=False, fire_base=0.10),
    "Kalimantan Barat": dict(island="Kalimantan", dry_peak=235, monsoon_amp=0.45, wet_rain_mm=11.5,
                             dry_rain_mm=4.6, base_temp_c=27.0, peat=True, fire_base=0.32),
    "Kalimantan Tengah": dict(island="Kalimantan", dry_peak=238, monsoon_amp=0.50, wet_rain_mm=11.0,
                              dry_rain_mm=4.0, base_temp_c=27.3, peat=True, fire_base=0.45),
    "Kalimantan Selatan": dict(island="Kalimantan", dry_peak=243, monsoon_amp=0.55, wet_rain_mm=10.2,
                               dry_rain_mm=3.4, base_temp_c=27.4, peat=True, fire_base=0.20),
    "Kalimantan Timur": dict(island="Kalimantan", dry_peak=236, monsoon_amp=0.42, wet_rain_mm=11.8,
                             dry_rain_mm=5.2, base_temp_c=27.1, peat=False, fire_base=0.14),
    "Papua": dict(island="Papua", dry_peak=225, monsoon_amp=0.32, wet_rain_mm=10.5,
                  dry_rain_mm=6.5, base_temp_c=26.8, peat=True, fire_base=0.16),
    "Nusa Tenggara Timur": dict(island="Nusa Tenggara", dry_peak=238, monsoon_amp=0.80,
                                wet_rain_mm=8.0, dry_rain_mm=0.8, base_temp_c=27.8,
                                peat=False, fire_base=0.11),
}


def monthly_to_daily(table: dict, dates: pd.DatetimeIndex) -> np.ndarray:
    """Indeks iklim bulanan yang diinterpolasi harian (hanya nilai bulan berjalan)."""
    monthly = []
    for year in range(2015, 2026):
        monthly.extend(table[year])
    monthly_idx = pd.date_range("2015-01-01", periods=len(monthly), freq="MS")
    s = pd.Series(monthly, index=monthly_idx).reindex(
        pd.date_range(monthly_idx.min(), monthly_idx.max() + pd.offsets.MonthEnd(0), freq="D")
    )
    s = s.ffill()
    return s.reindex(dates).to_numpy(dtype=float)


def kbdi_step(q: float, temp_c: float, rain_mm: float) -> float:
    """Satu langkah Keetch-Byram Drought Index (0-203.2).

    dQ = (203.2 - Q) * (0.968 * exp(0.0875*T + 1.5552) - 8.30) * 1e-3 * 24
    Hujan efektif (> 5.1 mm) menurunkan Q sebesar 100*(R-5.1)/25.4.
    """
    # faktor skala 1e-3 dipakai per hari (setara kekeringan +4 s.d. +6 poin/hari)
    drought_factor = (0.968 * np.exp(0.0875 * temp_c + 1.5552) - 8.30) * 1e-3
    q = q + (203.2 - q) * drought_factor
    if rain_mm > 5.1:
        q -= 100.0 * (rain_mm - 5.1) / 25.4
    return float(np.clip(q, 0.0, 203.2))


def simulate_province(name: str, params: dict, dates: pd.DatetimeIndex,
                      oni: np.ndarray, iod: np.ndarray,
                      rng: np.random.Generator) -> pd.DataFrame:
    n = len(dates)
    doy = dates.dayofyear.to_numpy()
    # dryness_season: 1 = puncak kemarau, 0 = puncak musim basah
    dryness_season = 0.5 + 0.5 * np.cos(2 * np.pi * (doy - params["dry_peak"]) / 365.25)
    dryness_season = dryness_season ** 1.2
    wet_weight = 1.0 - params["monsoon_amp"] * dryness_season

    # --- curah hujan: zero-inflated gamma, dipengaruhi ENSO ---
    rain_mean = (params["dry_rain_mm"]
                 + (params["wet_rain_mm"] - params["dry_rain_mm"])
                 * (1.0 - dryness_season) ** 1.6) * np.exp(-0.35 * oni - 0.25 * iod)
    p_zero = np.clip(0.10 + 0.55 * dryness_season, 0, 0.95)
    u = rng.random(n)
    gamma_draw = rng.gamma(shape=1.4, scale=np.maximum(rain_mean, 0.2) / 1.4)
    rain = np.where(u < p_zero, 0.0, gamma_draw)

    # --- suhu, kelembapan, angin ---
    temp = (params["base_temp_c"] + 1.6 * dryness_season + 0.9 * oni
            + rng.normal(0, 0.55, n) - 0.03 * rain)
    rh = np.clip(88.0 - 10.0 * dryness_season - 3.5 * oni - 0.35 * rain
                 + rng.normal(0, 3.0, n), 35.0, 99.0)
    wind = np.clip(1.6 + 1.0 * dryness_season + 0.35 * oni
                   + rng.normal(0, 0.45, n), 0.2, 9.0)

    # --- KBDI (kekeringan bahan bakar) ---
    kbdi = np.empty(n)
    q = 60.0
    for i in range(n):
        q = kbdi_step(q, temp[i], rain[i])
        kbdi[i] = q

    # --- hotspot: NegBin dengan mean bergantung KBDI + ENSO + komponen laten AR(1) ---
    phi = 0.88 if params["peat"] else 0.65   # gambut -> kebakaran bertahan lama
    latent = np.zeros(n)
    eps = rng.normal(0, 0.22, n)
    for i in range(1, n):
        latent[i] = phi * latent[i - 1] + eps[i]
    latent = np.clip(latent, -1.6, 1.9)

    # Ambang kekeringan: di bawah KBDI ~110 bahan bakar belum mudah menyala,
    # di atasnya respons hotspot tumbuh sangat tajam (non-linear).
    drought_excess = np.maximum(0.0, kbdi - 110.0) / 93.2
    log_mu = (np.log(params["fire_base"])
              + 5.0 * drought_excess ** 1.3
              + 0.45 * oni
              + 0.30 * iod
              + 1.15 * latent
              + 0.20 * (dates.dayofweek.to_numpy() >= 5))  # pembukaan lahan akhir pekan
    mu = np.exp(log_mu)
    mu = np.clip(mu, 0, 2600)
    # overdispersion: NegBin(n=1.15, p) dengan mean = mu
    nb_n = 1.15
    p = nb_n / (nb_n + mu)
    hotspot = rng.negative_binomial(nb_n, p).astype(float)
    # hari basah hampir pasti tanpa titik panas
    hotspot = np.where(rain > 25, np.round(hotspot * 0.05), hotspot)
    hotspot = np.where(rain > 8, np.round(hotspot * 0.35), hotspot)

    return pd.DataFrame({
        "date": dates,
        "province": name,
        "island": params["island"],
        "peat_dominant": int(params["peat"]),
        "hotspot_count": hotspot.astype(int),
        "rainfall_mm": np.round(rain, 2),
        "temp_c": np.round(temp, 2),
        "rh_pct": np.round(rh, 2),
        "wind_ms": np.round(wind, 2),
        "kbdi": np.round(kbdi, 2),
        "oni": np.round(oni, 3),
        "iod_dmi": np.round(iod, 3),
    })


def main() -> None:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START, END, freq="D")
    oni = monthly_to_daily(ONI, dates)
    iod = monthly_to_daily(IOD, dates)

    frames = []
    for name, params in PROVINCES.items():
        prov_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        frames.append(simulate_province(name, params, dates, oni, iod, prov_rng))

    df = pd.concat(frames, ignore_index=True)
    df["month"] = df["date"].dt.month
    df["doy"] = df["date"].dt.dayofyear
    df = df[["date", "province", "island", "peat_dominant", "hotspot_count",
             "rainfall_mm", "temp_c", "rh_pct", "wind_ms", "kbdi", "oni", "iod_dmi",
             "month", "doy"]].sort_values(["province", "date"]).reset_index(drop=True)

    df.to_csv(OUT_DIR / "karhutla_daily.csv", index=False)

    yearly = (df.groupby(df["date"].dt.year)["hotspot_count"].sum()
              .rename("total_hotspot").reset_index().rename(columns={"date": "year"}))
    summary = {
        "n_rows": int(len(df)),
        "n_provinces": int(df["province"].nunique()),
        "period": [START, END],
        "seed": SEED,
        "yearly_hotspot_totals": dict(zip(yearly["year"], yearly["total_hotspot"])),
        "share_zero_days_pct": round(float((df["hotspot_count"] == 0).mean() * 100), 2),
        "mean_hotspot_per_day": round(float(df["hotspot_count"].mean()), 3),
        "note": "data simulasi fisik-meteorologis; bukan observasi",
    }
    (OUT_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nTersimpan: {OUT_DIR / 'karhutla_daily.csv'} ({len(df):,} baris)")


if __name__ == "__main__":
    main()
