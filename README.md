# Prediksi Karhutla dengan Deep Learning

Prakiraan jumlah titik panas (hotspot) harian per provinsi untuk horizon **H+1, H+3, H+7**
menggunakan **LSTM, GRU, dan CNN-LSTM**, dibandingkan dengan empat baseline
(persistence, seasonal naive, SARIMAX musiman, HistGradientBoosting).

Keluaran utama:

| Berkas | Isi |
|---|---|
| `ppt/Prediksi_Karhutla_Deep_Learning.pptx` | 23 slide: latar belakang, data, metode, hasil, evaluasi, diskusi |
| `results/metrics_h1.json` | Metrik semua model pada horizon H+1 |
| `results/diebold_mariano.csv` | Uji signifikansi selisih galat antar model |
| `results/figures/*.png` | 9 gambar evaluasi (deret waktu, kurva latih, prakiraan, galat, dll.) |
| `models/*.pt` | Bobot model terlatih (PyTorch) |

## Hasil ringkas (periode uji Des 2022 – Des 2025, 10.960 observasi)

| Model | MAE | RMSE | R² | F1 kejadian |
|---|---|---|---|---|
| **GRU** | **10,02** | 42,1 | 0,339 | 0,77 |
| HistGB (tabular) | 10,13 | 43,0 | 0,311 | 0,76 |
| SARIMAX (1,0,1)(1,0,0)₇ | 10,14 | 43,7 | 0,290 | 0,76 |
| LSTM | 10,47 | **41,3** | **0,364** | 0,78 |
| CNN-LSTM | 10,54 | 41,9 | 0,345 | **0,78** |
| Persistence | 13,54 | 53,9 | −0,080 | 0,71 |
| Seasonal Naive (lag-7) | 14,52 | 55,6 | −0,151 | 0,68 |

* MAE satuannya titik panas/hari/provinsi.
* GRU menurunkan MAE **26,0%** dibanding persistence (uji Diebold-Mariano p < 0,001).
* GRU (p = 0,0067) dan LSTM (p = 0,0066) signifikan di atas HistGB; CNN-LSTM tidak (p = 0,088).
* Akurasi relatif stabil sampai H+7 (MAE GRU 10,02 → 10,39).
* Semua provinsi membaik 15–35% dibanding persistence.
* Kelemahan utama: puncak ekstrem diremehkan (bias −4,4 titik panas/hari) — lihat `results/figures/fig08_scatter.png`.
* Angka di atas berasal dari run terakhir; latihan CPU tidak bit-exact, sehingga rerun dapat bergeser ±0,5%.

## Struktur

```
karhutla-prediction/
├── data/
│   ├── generate_karhutla_data.py   # generator dataset simulasi (seed 42)
│   ├── karhutla_daily.csv          # 40.180 baris provinsi-hari
│   ├── dataset_summary.json        # ringkasan & kalibrasi data
│   └── data_dictionary.md          # arti kolom + asumsi simulasi
├── src/
│   ├── common.py                   # konfigurasi, fitur, windowing, metrik, uji DM
│   ├── models.py                   # LSTM / GRU / CNN-LSTM + loop pelatihan
│   ├── train.py                    # latih baseline + DL, simpan metrik & prediksi
│   ├── evaluate.py                 # gambar & ringkasan evaluasi
│   └── make_ppt.py                 # bangun deck PPTX dari artefak hasil
├── results/                        # metrik (json/csv), prediksi (.npz), figures/
├── models/                         # bobot model terlatih
└── ppt/                            # deck presentasi
```

## Menjalankan ulang

```bash
python -m pip install -r requirements.txt
# PyTorch CPU (Windows/Linux): pip install torch --index-url https://download.pytorch.org/whl/cpu
# Python 3.13, CPU, tanpa GPU; total waktu ~5 menit

python data/generate_karhutla_data.py   # dataset simulasi
python src/train.py                     # baseline + 3 model DL  (~4 menit di CPU)
python src/evaluate.py                  # 9 gambar + ringkasan
python src/make_ppt.py                  # deck PPTX
```

Data dan pemisahan fold deterministik (seed 42); pelatihan PyTorch di CPU dapat bergeser tipis
(±0,5% MAE) karena paralelisme, sehingga angka pada deck selalu diambil dari run terakhir.

## Metodologi singkat

1. **Data.** 10 provinsi rawan, harian 2015–2025: hotspot, curah hujan, suhu, RH, angin,
   KBDI (Keetch-Byram Drought Index), ONI (El Niño), IOD.
2. **Fitur (21).** Lag hotspot (1, 2, 3, 7, 14 hari), rolling mean/max, hujan hari ini/lag/akumulasi
   7 hari, KBDI dan turunannya, suhu, RH, angin, ONI, IOD, sinus–kosinus hari ke-.
3. **Windowing.** Jendela 14 hari → satu model multi-output untuk H+1, H+3, H+7.
4. **Anti-leakage.** Split kronologis train 2015–2021 / validasi 2022 / uji 2023–2025,
   jeda 7 hari antar fold, scaler hanya di-fit pada train (per provinsi), target `log(1+y)`.
5. **Model.** LSTM & GRU 2 lapis × 64 unit, CNN-LSTM (Conv1D 32 filter → maxpool → LSTM),
   embedding provinsi 8 dimensi, kepala keluaran Linear(72→64)→Linear(64→3).
6. **Pelatihan.** Huber loss (β=0,5), AdamW, early stopping pada MAE validasi, batch 256.
7. **Evaluasi.** MAE, RMSE, R², sMAPE, F1 deteksi kejadian (≥10 titik), MAE per provinsi
   dan per horizon, serta uji Diebold-Mariano (Newey-West, 7 lag).

## Penting: sifat data

Dataset di repo ini adalah **simulasi fisik-meteorologis**, bukan observasi. Rantai
kausal yang dipakai: ENSO/IOD → curah hujan → suhu & RH → KBDI → hotspot
(NegBin, AR(1), efek gambut). Kalibrasinya: ±60% hari tanpa hotspot, puncak
Agustus–Oktober, lonjakan pada tahun El Niño (2015, 2023).

Untuk memakai data asli:

1. Unduh hotspot VIIRS/MODIS dari [NASA FIRMS API](https://firms.modaps.eosdis.nasa.gov/api/)
   (butuh MAP_KEY gratis), agregasi ke harian per provinsi → kolom `date, province, hotspot_count`.
2. Lengkapi fitur cuaca dari ERA5-Land/BMKG (hujan, suhu, RH, angin), ONI dari NOAA,
   IOD dari BoM, lalu hitung KBDI harian.
3. Simpan dengan nama kolom yang sama seperti `data/data_dictionary.md` — pipeline
   `src/train.py` tidak perlu diubah.
4. Latih ulang dan bandingkan; jangan memakai angka pada README ini untuk data asli.

## Keterbatasan

* Data simulasi; belum divalidasi terhadap observasi FIRMS.
* Tanpa komponen spasial (grid/piksel) dan tanpa faktor manusia (pembukaan lahan).
* Horizon pendek (≤7 hari) dan belum ada interval ketidakpastian (kuantil/probabilistik).
* Model cenderung meremehkan puncak ekstrem (regresi ke rata-rata bersyarat).
* Satu titik prakiraan per provinsi per hari — bukan peringatan tingkat kecamatan.
