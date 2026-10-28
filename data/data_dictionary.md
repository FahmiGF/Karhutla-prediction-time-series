# Kamus data — `karhutla_daily.csv`

Satu baris = satu provinsi pada satu tanggal. 10 provinsi × 4.018 hari
(1 Jan 2015 – 31 Des 2025) = **40.180 baris**. Semua nilai dibangkitkan oleh
`generate_karhutla_data.py` (seed 42): **simulasi**, bukan observasi.

| Kolom | Tipe | Satuan | Keterangan |
|---|---|---|---|
| `date` | tanggal | — | Tanggal pengamatan (harian) |
| `province` | teks | — | Nama provinsi (10 provinsi) |
| `island` | teks | — | Pulau: Sumatera, Kalimantan, Papua, Nusa Tenggara |
| `peat_dominant` | 0/1 | — | 1 = lahan gambut dominan (kebakaran bawah permukaan, lebih persisten) |
| `hotspot_count` | integer | titik/hari | **Target**: jumlah titik panas |
| `rainfall_mm` | desimal | mm/hari | Curah hujan harian (zero-inflated gamma) |
| `temp_c` | desimal | °C | Suhu udara rata-rata harian |
| `rh_pct` | desimal | % | Kelembapan relatif (35–99) |
| `wind_ms` | desimal | m/s | Kecepatan angin |
| `kbdi` | desimal | indeks 0–203,2 | Keetch-Byram Drought Index (kekeringan bahan bakar kumulatif) |
| `oni` | desimal | °C anomali | Oceanic Niño Index bulanan (El Niño/La Niña), interpolasi harian |
| `iod_dmi` | desimal | °C anomali | Dipole Mode Index Samudra Hindia, interpolasi harian |
| `month` | integer | 1–12 | Bulan |
| `doy` | integer | 1–366 | Hari ke- dalam tahun |

## Skema pembangkitan (ringkas)

1. **Musim.** `dryness_season = (0,5 + 0,5·cos(2π(doy − dry_peak)/365,25))^1,2`,
   dengan `dry_peak` 225–248 (Agustus–September) tergantung provinsi.
2. **Hujan.** Rata-rata musiman = `dry_rain + (wet_rain − dry_rain)·(1 − dryness)^1,6`,
   dikalikan `exp(−0,35·ONI − 0,25·IOD)`; nilainya dijadikan zero-inflated gamma
   (peluang hari kering naik saat musim kemarau).
3. **Suhu/RH/angin.** Fungsi musim + ONI + noise; RH turun saat kering, angin naik.
4. **KBDI.** `dQ = (203,2 − Q)·(0,968·e^(0,0875T + 1,5552) − 8,30)·10⁻³` per hari;
   hujan > 5,1 mm menurunkan `Q` sebesar `100·(R − 5,1)/25,4`.
5. **Hotspot.** NegBin (n = 1,15) dengan
   `log μ = log fire_base + 5,0·[(KBDI − 110)/93,2]₊^1,3 + 0,45·ONI + 0,30·IOD
   + 1,15·latent_AR(1) + 0,20·(akhir pekan)`,
   lalu dikurangi drastis pada hari dengan hujan > 8 mm.
6. **Kalibrasi.** ±60% hari bernilai nol; puncak musim Agustus–Oktober;
   tahun El Niño (2015, 2023) jauh di atas tahun normal; `peat_dominant=1`
   memakai koefisien AR lebih tinggi (0,88 vs 0,65).

## Memakai data asli

Pertahankan nama kolom di atas. Sumber yang disarankan:

* **Hotspot** — NASA FIRMS (`VIIRS S-NPP`, 375 m), agregasi harian per provinsi.
* **Cuaca** — ERA5-Land (ECMWF) atau data pengamatan BMKG.
* **ONI** — NOAA CPC; **IOD** — Bureau of Meteorology Australia.
* **KBDI** — hitung sendiri dari hujan & suhu harian dengan rumus pada poin 4.

Setelah CSV diganti, jalankan `python src/train.py` lalu `python src/evaluate.py`
dan `python src/make_ppt.py`; seluruh angka pada deck akan diperbarui otomatis.
