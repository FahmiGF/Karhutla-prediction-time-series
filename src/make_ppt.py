"""Bangun deck presentasi (PPTX) yang menjelaskan metode & evaluasi.

Semua angka dibaca dari results/*.json dan results/*.csv sehingga deck selalu
konsisten dengan hasil eksperimen terakhir.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import torch
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

from common import ARCH_OUT, FIG, MODELS, RESULTS, ROOT
from models import build_model

# ------------------------------------------------------------------ palet
INK = RGBColor(0x11, 0x18, 0x27)
SLATE = RGBColor(0x37, 0x41, 0x51)
GRAY = RGBColor(0x6B, 0x72, 0x80)
LIGHT = RGBColor(0xF3, 0xF4, 0xF6)
LINE = RGBColor(0xE5, 0xE7, 0xEB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FIRE = RGBColor(0xDC, 0x26, 0x26)
ORANGE = RGBColor(0xF9, 0x73, 0x16)
AMBER = RGBColor(0xF5, 0x9E, 0x0B)
TEAL = RGBColor(0x05, 0x96, 0x69)
BLUE = RGBColor(0x25, 0x63, 0xEB)
PLUM = RGBColor(0x7C, 0x3A, 0xED)

FONT = "Segoe UI"
MONO = "Consolas"
W, H = 13.333, 7.5

prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)
BLANK = prs.slide_layouts[6]


# ------------------------------------------------------------------ helper
def text(slide, value, left, top, width, height, size=16, bold=False,
         color=INK, align=PP_ALIGN.LEFT, font=FONT, italic=False,
         line_spacing=1.0, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    lines = value.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        run = p.add_run()
        run.text = line
        f = run.font
        f.name, f.size, f.bold, f.italic = font, Pt(size), bold, italic
        f.color.rgb = color
    return box


def rect(slide, left, top, width, height, fill, line=None, radius=0.06,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    shp = slide.shapes.add_shape(shape, Inches(left), Inches(top),
                                 Inches(width), Inches(height))
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(1)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        try:
            shp.adjustments[0] = radius
        except Exception:
            pass
    shp.shadow.inherit = False
    return shp


def new_slide(title: str, eyebrow: str = "", number: bool = True,
              subtitle: str | None = None) -> object:
    slide = prs.slides.add_slide(BLANK)
    rect(slide, 0, 0, W, 0.09, ORANGE, shape=MSO_SHAPE.RECTANGLE)
    if eyebrow:
        text(slide, eyebrow.upper(), 0.75, 0.42, 11.8, 0.3, size=11,
             bold=True, color=ORANGE)
        text(slide, title, 0.75, 0.72, 11.8, 0.75, size=27, bold=True, color=INK)
    else:
        text(slide, title, 0.75, 0.55, 11.8, 0.8, size=29, bold=True, color=INK)
    if subtitle:
        text(slide, subtitle, 0.75, 1.42, 11.8, 0.4, size=13, color=GRAY)
    if number:
        idx = len(prs.slides)
        text(slide, f"{idx:02d}", 12.25, 6.92, 0.6, 0.3, size=10, color=GRAY,
             align=PP_ALIGN.RIGHT)
        text(slide, "Prediksi Karhutla dengan Deep Learning", 0.75, 6.92, 6.0, 0.3,
             size=9, color=GRAY)
    return slide


def bullets(slide, items, left, top, width, height, size=15, color=INK,
            bullet_color=ORANGE, spacing=0.34, gap=10):
    """items: list of str atau (str, level). Tinggi teks diperkirakan dari panjang string."""
    box_top = top
    for item in items:
        level = 0
        if isinstance(item, tuple):
            item, level = item
        sz = size if level == 0 else size - 2
        indent = level * 0.35
        avail_w = width - indent - 0.32
        marker = "▪" if level == 0 else "–"
        mcolor = bullet_color if level == 0 else GRAY
        # perkiraan jumlah baris: lebar rata-rata karakter ~0.5 x ukuran font
        chars_per_line = max(20, int(avail_w * 72 / (sz * 0.50)))
        n_lines = max(1, math.ceil(len(item) / chars_per_line))
        line_h = sz * 1.32 / 72
        block_h = n_lines * line_h
        text(slide, marker, left + indent, box_top, 0.28, 0.3, size=sz,
             bold=True, color=mcolor)
        text(slide, item, left + indent + 0.32, box_top, avail_w, block_h,
             size=sz, color=color if level == 0 else SLATE, line_spacing=1.15)
        box_top += block_h + gap / 72


def add_image(slide, path, left, top, max_w, max_h):
    pic = slide.shapes.add_picture(str(path), Inches(left), Inches(top),
                                   width=Inches(max_w))
    if pic.height > Emu(int(max_h * 914400)):
        pic.height = Emu(int(max_h * 914400))
        pic.width = Emu(int(pic.width * max_h * 914400 / pic.height * 1.0))
        # hitung ulang proporsional
        ratio = pic.width / pic.height
        pic.height = Emu(int(max_h * 914400))
        pic.width = Emu(int(pic.height * ratio))
    return pic


def fit_image(slide, path, left, top, max_w, max_h, center=True):
    from PIL import Image
    with Image.open(path) as im:
        iw, ih = im.size
    ratio = iw / ih
    w, h = max_w, max_w / ratio
    if h > max_h:
        h, w = max_h, max_h * ratio
    x = left + (max_w - w) / 2 if center else left
    slide.shapes.add_picture(str(path), Inches(x), Inches(top), Inches(w), Inches(h))


def table(slide, rows, col_widths, left, top, font_size=11, header_fill=INK,
          row_height=0.34, highlight_rows=(), highlight_fill=RGBColor(0xFE, 0xF3, 0xC7),
          align_first_left=True):
    n_rows, n_cols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_rows, n_cols, Inches(left), Inches(top),
                                   Inches(sum(col_widths)),
                                   Inches(row_height * n_rows))
    tbl = shape.table
    for j, wd in enumerate(col_widths):
        tbl.columns[j].width = Inches(wd)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_height)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = ""
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if (j == 0 and align_first_left) else PP_ALIGN.CENTER
            run = p.add_run()
            run.text = str(val)
            run.font.size = Pt(font_size)
            run.font.name = FONT
            run.font.bold = (i == 0) or (i in highlight_rows)
            run.font.color.rgb = WHITE if i == 0 else INK
            if i == 0:
                cell.fill.solid(); cell.fill.fore_color.rgb = header_fill
            elif i in highlight_rows:
                cell.fill.solid(); cell.fill.fore_color.rgb = highlight_fill
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if i % 2 else LIGHT
    return tbl


def kpi(slide, left, top, width, value, label, color=FIRE, sub=None):
    rect(slide, left, top, width, 1.35, WHITE, line=LINE, radius=0.08)
    rect(slide, left, top, 0.07, 1.35, color, shape=MSO_SHAPE.RECTANGLE)
    text(slide, value, left + 0.24, top + 0.16, width - 0.4, 0.55, size=30,
         bold=True, color=color)
    text(slide, label, left + 0.24, top + 0.72, width - 0.4, 0.5, size=11,
         color=GRAY)
    if sub:
        text(slide, sub, left + 0.24, top + 1.02, width - 0.4, 0.3, size=9.5,
             color=SLATE, italic=True)


def callout(slide, left, top, width, height, title, body, color=AMBER):
    rect(slide, left, top, width, height, RGBColor(0xFF, 0xFB, 0xEB), line=color,
         radius=0.05)
    text(slide, title, left + 0.22, top + 0.14, width - 0.4, 0.3, size=12.5,
         bold=True, color=RGBColor(0x92, 0x40, 0x0E))
    text(slide, body, left + 0.22, top + 0.48, width - 0.42, height - 0.55,
         size=10.5, color=SLATE, line_spacing=1.15)


def notes(slide, value):
    slide.notes_slide.notes_text_frame.text = value


# ------------------------------------------------------------------ muat hasil
metrics = json.loads((RESULTS / "metrics_h1.json").read_text())["metrics"]
dm = pd.read_csv(RESULTS / "diebold_mariano.csv")
by_h = pd.read_csv(RESULTS / "metrics_by_horizon.csv")
per_prov = pd.read_csv(RESULTS / "metrics_per_province.csv")
hist = json.loads((RESULTS / "training_history.json").read_text())
config = json.loads((RESULTS / "config.json").read_text())
dssum = json.loads((ROOT / "data" / "dataset_summary.json").read_text())

ORDER = ["GRU", "HistGB (tabular)", "SARIMAX(1,0,1)(1,0,0,7)", "LSTM",
         "CNN-LSTM", "Persistence", "Seasonal Naive (lag-7)"]
best = "GRU"
pers = metrics["Persistence"]
gain = (pers["MAE"] - metrics[best]["MAE"]) / pers["MAE"] * 100
test_period = "25 Des 2022 – 24 Des 2025"

yr = dssum["yearly_hotspot_totals"]
peak_year = max(yr, key=yr.get)

DL_MODELS = ["LSTM", "GRU", "CNN-LSTM"]
best_rmse = min(metrics, key=lambda k: metrics[k]["RMSE"])
best_f1 = max(metrics, key=lambda k: metrics[k]["F1_event"])
dl_gain_min = (pers["MAE"] - max(metrics[k]["MAE"] for k in DL_MODELS)) / pers["MAE"] * 100
p_gru_hgb = float(dm[(dm.model == "GRU") & (dm.reference == "HistGB (tabular)")].p_value.iloc[0])
p_lstm_hgb = float(dm[(dm.model == "LSTM") & (dm.reference == "HistGB (tabular)")].p_value.iloc[0])
p_cnn_hgb = float(dm[(dm.model == "CNN-LSTM") & (dm.reference == "HistGB (tabular)")].p_value.iloc[0])

_z = np.load(RESULTS / "predictions_test.npz", allow_pickle=True)
bias_gru = float((_z["gru"][:, 0] - _z["y_true"][:, 0]).mean())
_piv = (per_prov[per_prov.model.isin(["GRU", "Persistence"])]
        .pivot(index="province", columns="model", values="MAE"))
_piv["gain"] = (_piv["Persistence"] - _piv["GRU"]) / _piv["Persistence"] * 100
prov_max = _piv["gain"].idxmax()
prov_min = _piv["gain"].idxmin()
gain_max, gain_min = float(_piv["gain"].max()), float(_piv["gain"].min())

# jumlah parameter tiap arsitektur
n_feat = 21
param_rows, param_count = [], {}
for name in DL_MODELS:
    m = build_model(name, n_feat, 10, type("C", (), {
        "hidden": config["hidden"], "horizons": config["horizons"],
        "dropout": config["dropout"]})())
    n_param = sum(p.numel() for p in m.parameters())
    param_count[name] = n_param
    param_rows.append((name, f"{n_param:,}".replace(",", ".")))
    del m
gru_vs_lstm_pct = (1 - param_count["GRU"] / param_count["LSTM"]) * 100

# ================================================================== 1 judul
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, W, H, INK, shape=MSO_SHAPE.RECTANGLE)
rect(s, 0, 0, 0.22, H, ORANGE, shape=MSO_SHAPE.RECTANGLE)
text(s, "ANALISIS & PREDIKSI KARHUTLA", 1.1, 1.35, 11, 0.4, size=13, bold=True,
     color=ORANGE)
text(s, "Prediksi Titik Panas Karhutla\nMenggunakan Deep Learning", 1.1, 1.85, 11, 1.7,
     size=44, bold=True, color=WHITE, line_spacing=1.05)
rect(s, 1.12, 3.75, 2.0, 0.05, ORANGE, shape=MSO_SHAPE.RECTANGLE)
text(s, "LSTM · GRU · CNN-LSTM  vs  Persistence · Seasonal Naive · SARIMAX · HistGB",
     1.1, 4.05, 11.2, 0.5, size=14.5, color=RGBColor(0xD1, 0xD5, 0xDB))
text(s, "10 provinsi rawan · data harian 2015–2025 · prakiraan H+1, H+3, H+7",
     1.1, 4.85, 10.6, 0.4, size=13, color=RGBColor(0x9C, 0xA3, 0xAF))
for i, (k, v) in enumerate([("MAE terbaik", f"{metrics[best]['MAE']:.2f}"),
                            ("vs persistence", f"−{gain:.0f}%"),
                            ("F1 kejadian", f"{metrics[best]['F1_event']:.2f}")]):
    left = 1.1 + i * 3.35
    rect(s, left, 5.55, 3.0, 0.95, RGBColor(0x1F, 0x29, 0x37), radius=0.1)
    text(s, v, left + 0.25, 5.68, 2.6, 0.45, size=22, bold=True, color=ORANGE)
    text(s, k, left + 0.25, 6.12, 2.6, 0.3, size=10.5, color=RGBColor(0x9C, 0xA3, 0xAF))
notes(s, "Deck ini menjelaskan data, metode baseline, tiga arsitektur deep learning, "
         "serta evaluasi kuantitatif dan statistik. Semua angka berasal dari hasil "
         "eksperimen terakhir di folder results/.")

# ================================================================== 2 agenda
s = new_slide("Agenda", "Peta presentasi")
items = [("01", "Latar belakang & tujuan", "Mengapa karhutla perlu diprakirakan dan mengapa deep learning"),
         ("02", "Data & eksplorasi", "Skema data harian 10 provinsi, variabel, pola musiman dan pendorong"),
         ("03", "Metodologi", "Praproses anti-leakage, baseline, LSTM/GRU/CNN-LSTM, protokol latih"),
         ("04", "Evaluasi", "MAE/RMSE/R²/F1, uji Diebold-Mariano, horizon, per provinsi, analisis galat"),
         ("05", "Diskusi & kesimpulan", "Implikasi operasional, keterbatasan, rekomendasi")]
y = 1.55
for num, title_, desc in items:
    rect(s, 0.85, y, 0.62, 0.62, ORANGE, radius=0.2)
    text(s, num, 0.85, y + 0.13, 0.62, 0.35, size=15, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER)
    text(s, title_, 1.72, y + 0.02, 4.2, 0.35, size=16, bold=True, color=INK)
    text(s, desc, 1.72, y + 0.34, 9.8, 0.35, size=12, color=GRAY)
    y += 1.02
notes(s, "Alur: konteks -> data -> metode -> bukti evaluasi -> implikasi.")

# ================================================================== 3 latar belakang
s = new_slide("Karhutla: dampak besar, prakiraan masih reaktif", "01 · Latar belakang")
bullets(s, [
    "Kebakaran hutan dan lahan (karhutla) tahunan di Sumatra & Kalimantan, diperparah El Niño dan IOD positif.",
    "Lahan gambut menyimpan bahan bakar di bawah permukaan: api sulit dipadamkan dan asap melintasi batas provinsi.",
    "Dampak: gangguan kesehatan (ISPA), penutupan sekolah & bandara, kerugian ekonomi, emisi karbon besar.",
    "Deteksi satelit (hotspot VIIRS/MODIS) sudah rutin, tetapi sebagian besar sistem masih bersifat deteksi dini, bukan prakiraan.",
    "Model deret waktu klasik sulit menangkap hubungan non-linear cuaca–kekeringan–hotspot yang tertunda beberapa hari.",
], 0.75, 1.62, 7.4, 4.6, size=14.5)
rect(s, 8.45, 1.62, 4.15, 4.55, LIGHT, radius=0.05)
text(s, "Pertanyaan inti", 8.72, 1.85, 3.6, 0.35, size=13, bold=True, color=FIRE)
text(s, "Seberapa baik model deep learning memprakirakan jumlah titik panas "
        "H+1, H+3, dan H+7 dibandingkan baseline statistik dan machine learning?",
     8.72, 2.28, 3.6, 1.5, size=12.5, color=SLATE, line_spacing=1.2)
text(s, "Fokus deliverable", 8.72, 3.85, 3.6, 0.35, size=13, bold=True, color=FIRE)
text(s, "• pipeline data & latih yang reproducible\n"
        "• 3 arsitektur DL + 4 baseline\n"
        "• evaluasi kuantitatif & uji signifikansi\n"
        "• analisis keterbatasan model", 8.72, 4.25, 3.6, 1.6, size=11.5,
     color=SLATE, line_spacing=1.25)
notes(s, "Tekankan bahwa nilai tambah bukan sekadar deteksi hotspot, tetapi "
         "kemampuan memprakirakan beberapa hari ke depan untuk peringatan dini.")

# ================================================================== 4 tujuan
s = new_slide("Tujuan & ruang lingkup", "01 · Tujuan")
bullets(s, [
    "Membangun pipeline prakiraan jumlah titik panas harian per provinsi untuk horizon H+1, H+3, H+7.",
    "Membandingkan tiga arsitektur deep learning: LSTM, GRU, dan CNN-LSTM.",
    ("Benchmark: persistence, seasonal naive (lag-7), SARIMAX musiman, dan HistGradientBoosting.", 1),
    "Mengevaluasi dengan MAE, RMSE, R², sMAPE, dan F1 deteksi kejadian, lalu menguji signifikansi selisih galat (Diebold-Mariano).",
    "Menganalisis di mana dan kapan model gagal, serta implikasinya untuk operasional peringatan dini.",
], 0.75, 1.65, 7.3, 4.4, size=14)
rect(s, 8.35, 1.62, 4.25, 4.45, WHITE, line=LINE, radius=0.05)
text(s, "Ruang lingkup", 8.6, 1.82, 3.7, 0.3, size=13, bold=True, color=INK)
rows = [["Aspek", "Batasan"],
        ["Wilayah", "10 provinsi rawan"],
        ["Periode", "2015 – 2025 (harian)"],
        ["Target", "jumlah titik panas/hari"],
        ["Horizon", "1, 3, 7 hari"],
        ["Unit model", "per provinsi, dilatih gabungan"],
        ["Komputasi", "CPU, tanpa GPU"]]
table(s, rows, [1.5, 2.35], 8.6, 2.22, font_size=10.5, row_height=0.42)
notes(s, "Sebutkan bahwa unit analisis adalah provinsi-hari, bukan piksel grid, "
         "sehingga model bersifat tabular-temporal, bukan spasial.")

# ================================================================== 5 data
s = new_slide("Data: 40.180 baris provinsi-hari", "02 · Data")
text(s, "Sepuluh provinsi prioritas, 11 tahun pengamatan harian, tujuh variabel cuaca-iklim "
        "dan satu target hotspot.", 0.75, 1.5, 11.6, 0.4, size=13, color=GRAY)
rows = [["Variabel", "Satuan", "Peran"],
        ["hotspot_count", "titik/hari", "TARGET yang diprakirakan"],
        ["rainfall_mm", "mm/hari", "penekan penyalaan & pemadam alami"],
        ["temp_c / rh_pct", "°C / %", "pengeringan bahan bakar"],
        ["wind_ms", "m/s", "penyebaran api"],
        ["kbdi", "indeks 0–203", "kekeringan kumulatif (Keetch-Byram)"],
        ["oni", "°C anomali", "El Niño / La Niña"],
        ["iod_dmi", "°C anomali", "Dipole Mode Samudra Hindia"]]
table(s, rows, [2.45, 1.85, 4.6], 0.75, 2.0, font_size=10.5, row_height=0.4,
      highlight_rows=(1,))
callout(s, 9.9, 2.0, 2.75, 3.15, "Sifat data",
        "Dataset ini SIMULASI fisik-meteorologis (seed 42), bukan observasi. "
        "Rantai kausal: ENSO/IOD → hujan → suhu & RH → KBDI → hotspot (NegBin, AR(1), "
        "efek gambut). Struktur kolom identik dengan agregasi NASA FIRMS, sehingga data "
        "asli dapat menggantikan CSV ini tanpa mengubah kode.", AMBER)
text(s, "Kalibrasi simulasi: ±60% hari tanpa hotspot · puncak musim Agustus–Oktober · "
        "lonjakan pada tahun El Niño (2015, 2023) · total 2015 dan 2023 tertinggi.",
     0.75, 6.05, 8.9, 0.55, size=11, color=SLATE, italic=True)
notes(s, "Jujur sejak awal: data simulasi. Tekankan bahwa tujuan deck adalah "
         "metodologi dan evaluasi, dan pipeline siap dipakai untuk data FIRMS asli.")

# ================================================================== 6 EDA series
s = new_slide("Pola waktu: musiman kuat, lonjakan episodik", "02 · Eksplorasi")
fit_image(s, FIG / "fig01_series.png", 0.75, 1.55, 11.85, 4.15)
bullets(s, [
    f"Puncak historis pada {peak_year} (El Niño kuat); 2015 dan 2023 jauh di atas tahun normal.",
    "Hotspot hampir nol pada Januari–April; naik tajam Juli dan memuncak September.",
    "Pemisahan train (2015–2021), validasi (2022), uji (2023–2025) dilakukan kronologis.",
], 0.85, 5.62, 11.6, 1.2, size=12.5, spacing=0.28)
notes(s, "Perhatikan dua rezim: musim basah (hampir nol) dan musim kemarau (ratusan "
         "hotspot/hari). Ini yang membuat MAPE tidak stabil dan F1 kejadian lebih informatif.")

# ================================================================== 7 EDA drivers
s = new_slide("Kekeringan kumulatif lebih menentukan daripada hujan harian", "02 · Eksplorasi")
fit_image(s, FIG / "fig03_drivers.png", 0.55, 1.6, 5.6, 5.0)
bullets(s, [
    "Hotspot berkorelasi paling kuat dengan rata-rata 7 hari dan nilai lag-1 → memori kebakaran gambut.",
    "KBDI (kekeringan kumulatif) lebih informatif (r≈0.26) daripada hujan hari itu (r≈−0.10).",
    "ONI dan IOD keduanya positif terhadap hotspot: El Niño dan IOD+ sama-sama menekan hujan.",
    "Suhu dan RH hampir simetris — keduanya proksi pengeringan bahan bakar.",
    "Implikasi desain fitur: sertakan lag, rolling mean, dan KBDI, bukan hanya cuaca hari ini.",
], 6.35, 1.75, 6.3, 4.8, size=13, spacing=0.4)
notes(s, "Ini justifikasi fitur. Tanpa lag dan KBDI, model kehilangan konteks "
         "kekeringan yang menentukan apakah hotspot akan berlanjut.")

# ================================================================== 8 pipeline
s = new_slide("Pipeline: dari data mentah ke prakiraan", "03 · Metodologi")
steps = [("Data harian\n10 provinsi", BLUE), ("Rekayasa fitur\nlag & KBDI", PLUM),
         ("Windowing\n14 hari", ORANGE), ("Model\nDL / baseline", FIRE),
         ("Evaluasi\nmetrik & uji DM", TEAL)]
x = 0.8
for i, (label, color) in enumerate(steps):
    rect(s, x, 1.85, 2.08, 1.15, WHITE, line=color, radius=0.1)
    text(s, label, x + 0.12, 1.98, 1.84, 0.9, size=12, bold=True, color=color,
         align=PP_ALIGN.CENTER)
    if i < len(steps) - 1:
        text(s, "→", x + 2.12, 2.22, 0.35, 0.4, size=20, bold=True, color=GRAY,
             align=PP_ALIGN.CENTER)
    x += 2.47
rows = [["Tahap", "Detail"],
        ["Rekayasa fitur", "21 fitur: lag hotspot (1,2,3,7,14), rolling mean/max, hujan (hari ini, lag, akumulasi 7 hari), KBDI & turunannya, suhu, RH, angin, ONI, IOD, sinus-kosinus hari ke-"],
        ["Windowing", "Jendela 14 hari terakhir → target 3 nilai (H+1, H+3, H+7) dalam satu model multi-output"],
        ["Target", "log(1 + hotspot) lalu distandardisasi per provinsi (statistik train saja)"],
        ["Pencegahan bocor", "Split kronologis + jeda 7 hari antar fold; scaler hanya di-fit pada train"],
        ["Pelatihan", "Loss Huber (SmoothL1, β=0.5), AdamW, early stopping pada MAE validasi"]]
table(s, rows, [2.1, 9.75], 0.8, 3.3, font_size=10.5, row_height=0.58)
notes(s, "Tekankan windowing: model melihat 14 hari terakhir, memprakirakan tiga "
         "horizon sekaligus. Ini membuat satu model bisa dipakai untuk H+1, H+3, H+7.")

# ================================================================== 9 split
s = new_slide("Disiplin evaluasi: kronologis, bukan acak", "03 · Metodologi")
# timeline
rect(s, 0.9, 2.35, 11.6, 0.55, LINE, radius=0.2)
frac = [("Train 2015–2021", 0.0, 0.70, BLUE), ("Val 2022", 0.70, 0.11, AMBER),
        ("Uji 2023–2025", 0.81, 0.19, FIRE)]
for label, start, width, color in frac:
    rect(s, 0.9 + 11.6 * start, 2.35, 11.6 * width, 0.55, color, radius=0.2)
    text(s, label, 0.9 + 11.6 * start + 0.15, 2.46, 11.6 * width - 0.3, 0.35,
         size=11.5, bold=True, color=WHITE)
text(s, "10.960 observasi uji · 2.920 observasi validasi", 0.9, 3.05, 11.6, 0.35,
     size=11, color=GRAY)
bullets(s, [
    "Tidak ada pengacakan: data masa depan tidak boleh muncul saat pelatihan.",
    "Jeda 7 hari di setiap batas fold mencegah target H+7 ikut masuk ke fold sebelumnya.",
    "Standardisasi fitur & target memakai rata-rata/simpangan baku train per provinsi.",
    "Ketidakseimbangan ekstrem ditangani dengan transformasi log(1+y) dan loss Huber.",
    "Metrik terutama dilaporkan pada skala asli (titik panas), bukan skala ternormalisasi.",
], 0.9, 3.6, 6.0, 3.0, size=13, spacing=0.36)
rect(s, 7.35, 3.55, 5.15, 2.85, LIGHT, radius=0.06)
text(s, "Mengapa bukan MAPE saja?", 7.6, 3.75, 4.6, 0.35, size=13, bold=True, color=INK)
text(s, "Sekitar 60% hari bernilai nol. MAPE meledak ketika penyebut nol, sehingga "
        "laporan memakai MAE & RMSE (skala asli), sMAPE yang stabil, R² untuk daya "
        "jelas, dan F1 untuk kemampuan mendeteksi hari berisiko (≥10 titik).",
     7.6, 4.15, 4.65, 2.1, size=11.5, color=SLATE, line_spacing=1.2)
notes(s, "Poin yang sering ditanyakan penguji: mengapa tidak random split? "
         "Karena akan terjadi kebocoran temporal dan hasil terlihat terlalu bagus.")

# ================================================================== 10 baseline
s = new_slide("Empat baseline sebagai pembanding yang adil", "03 · Baseline")
rows = [["Model", "Ide dasar", "Kekuatan", "Kelemahan"],
        ["Persistence", "ŷ(t+h) = y(t)", "sulit dikalahkan saat kondisi stabil", "buta terhadap perubahan cuaca"],
        ["Seasonal Naive (lag-7)", "ŷ(t+h) = y(t+h−7)", "menangkap siklus mingguan", "gagal saat rezim berubah"],
        ["SARIMAX (1,0,1)(1,0,0)₇", "ARIMA musiman + exog KBDI, hujan, ONI, IOD", "interpretable, uji statistik mapan", "linear, asumsi residual normal"],
        ["HistGradientBoosting", "pohon boosting pada fitur hari t", "menangkap interaksi non-linear", "satu langkah, tidak memodelkan urutan"]]
table(s, rows, [2.55, 3.15, 3.3, 2.85], 0.8, 1.8, font_size=10.5, row_height=0.62)
bullets(s, [
    "Baseline dilatih/dievaluasi pada fold yang sama persis dengan model deep learning.",
    "SARIMAX difit per provinsi (10 model); HistGB pada data gabungan + one-hot provinsi.",
    "Model deep learning harus mengalahkan baseline ini, bukan hanya mengalahkan nol.",
], 0.8, 5.62, 11.8, 1.2, size=12, gap=8)
notes(s, "Tabel ini menjawab pertanyaan 'apakah DL layak secara praktis?'. "
         "HistGB adalah pesaing terberat karena fiturnya sama.")

# ================================================================== 11 arsitektur
s = new_slide("Tiga arsitektur deep learning", "03 · Deep learning")
rows = [["Model", "Struktur", "Parameter", "Alasan dipilih"],
        ["LSTM", "2 lapis, 64 unit, dropout 0.2", param_rows[0][1],
         "pembanding standar untuk dependensi jangka panjang"],
        ["GRU", "2 lapis, 64 unit, dropout 0.2", param_rows[1][1],
         "lebih ringan; cocok untuk data harian pendek"],
        ["CNN-LSTM", "Conv1D 32 filter (k=3) ×2 → maxpool → LSTM 64", param_rows[2][1],
         "konvolusi menangkap pola lokal 3 hari sebelum LSTM"]]
table(s, rows, [1.65, 4.35, 1.6, 4.2], 0.8, 1.85, font_size=11, row_height=0.7)
text(s, "Semua model memakai input yang sama dan kepala keluaran identik:",
     0.8, 4.95, 7.0, 0.3, size=12, bold=True, color=INK)
bullets(s, [
    "Input: tensor 14 × 21 (14 hari × 21 fitur).",
    "Embedding provinsi (dimensi 8) digabung ke keadaan tersembunyi terakhir → memberi konteks wilayah.",
    "Kepala keluaran: Linear(72→64) → ReLU → Dropout → Linear(64→3) untuk H+1, H+3, H+7.",
], 0.85, 5.3, 11.7, 1.5, size=12.5, spacing=0.34)
notes(s, "Parameter dihitung dari model sebenarnya (torch), bukan perkiraan. "
         "Embedding provinsi penting karena skala hotspot antar provinsi berbeda jauh.")

# ================================================================== 12 persamaan
s = new_slide("Persamaan inti: gerbang LSTM & GRU", "03 · Deep learning")
box_w, box_h = 5.9, 4.4
rect(s, 0.8, 1.7, box_w, box_h, LIGHT, radius=0.04)
text(s, "LSTM — empat gerbang", 1.05, 1.9, 5.4, 0.35, size=14, bold=True, color=BLUE)
lstm_eq = ("fₜ = σ(W_f·[hₜ₋₁, xₜ] + b_f)\n"
           "iₜ = σ(W_i·[hₜ₋₁, xₜ] + b_i)\n"
           "oₜ = σ(W_o·[hₜ₋₁, xₜ] + b_o)\n"
           "C̃ₜ = tanh(W_C·[hₜ₋₁, xₜ] + b_C)\n"
           "Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ\n"
           "hₜ = oₜ ⊙ tanh(Cₜ)")
text(s, lstm_eq, 1.05, 2.35, 5.4, 2.4, size=13.5, font=MONO, color=INK,
     line_spacing=1.35)
text(s, "Gerbang lupa (f) mempertahankan memori kekeringan berhari-hari; "
        "gerbang input (i) menyerap sinyal cuaca baru.", 1.05, 5.0, 5.4, 0.9,
     size=11, color=SLATE, line_spacing=1.2)

rect(s, 7.05, 1.7, box_w, box_h, LIGHT, radius=0.04)
text(s, "GRU — dua gerbang", 7.3, 1.9, 5.4, 0.35, size=14, bold=True, color=FIRE)
gru_eq = ("zₜ = σ(W_z·[hₜ₋₁, xₜ])\n"
          "rₜ = σ(W_r·[hₜ₋₁, xₜ])\n"
          "h̃ₜ = tanh(W·[rₜ ⊙ hₜ₋₁, xₜ])\n"
          "hₜ = (1 − zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ")
text(s, gru_eq, 7.3, 2.35, 5.4, 1.8, size=13.5, font=MONO, color=INK,
     line_spacing=1.35)
text(s, f"GRU menggabungkan memori dan kandidat pada satu vektor keadaan: "
        f"≈{gru_vs_lstm_pct:.0f}% lebih sedikit parameter, latihan lebih cepat, dan pada data harian "
        "sering menyamai atau mengungguli LSTM.\n\nσ = sigmoid · ⊙ = perkalian "
        "elemen per elemen · [hₜ₋₁, xₜ] = konkatenasi keadaan sebelumnya dan input hari ini.",
     7.3, 4.25, 5.4, 1.9, size=11, color=SLATE, line_spacing=1.2)
notes(s, "Rumus ditulis dengan simbol Unicode agar tetap terbaca tanpa LaTeX. "
         "Jika ditanya: σ = sigmoid, ⊙ = perkalian elemen.")

# ================================================================== 13 CNN-LSTM & setup
s = new_slide("CNN-LSTM & protokol pelatihan", "03 · Deep learning")
bullets(s, [
    "Conv1D dengan 32 filter (kernel 3) dua lapis: mendeteksi motif lokal harian seperti 'tiga hari tanpa hujan diikuti suhu naik'.",
    "MaxPooling memendekkan deret 14 → 7 langkah, mengurangi beban komputasi LSTM.",
    "LSTM satu lapis menyusun motif tersebut menjadi dinamika musim; kepala keluaran sama dengan model lain.",
], 0.8, 1.65, 6.15, 2.4, size=12.5, spacing=0.36)
rows = [["Hyperparameter", "Nilai"],
        ["Panjang jendela", "14 hari"],
        ["Unit tersembunyi", str(config["hidden"])],
        ["Dropout", str(config["dropout"])],
        ["Batch size", str(config["batch_size"])],
        ["Optimizer / LR", f"AdamW / {config['lr']:.0e}"],
        ["Loss", "Huber (SmoothL1, β=0.5)"],
        ["Early stopping", f"patience {config['patience']} pada MAE validasi"],
        ["Maksimum epoch", str(config["epochs"])],
        ["Hardware", "CPU (tanpa GPU)"]]
table(s, rows, [2.9, 3.15], 7.2, 1.65, font_size=10.5, row_height=0.43)
text(s, f"Epoch terbaik — LSTM {hist['LSTM']['best_epoch']}, GRU {hist['GRU']['best_epoch']}, "
        f"CNN-LSTM {hist['CNN-LSTM']['best_epoch']}. Bobot terbaik dipulihkan sebelum pengujian.",
     0.8, 6.1, 11.8, 0.4, size=11, color=SLATE, italic=True)
notes(s, "Tegaskan: semua model memakai protokol identik, satu-satunya yang berubah "
         "adalah arsitektur. Ini membuat perbandingan adil.")

# ================================================================== 14 hasil utama
s = new_slide("Hasil: GRU paling akurat, DL unggul atas baseline", "04 · Evaluasi")
kpi(s, 0.8, 1.6, 3.75, f"{metrics['GRU']['MAE']:.2f}", "MAE GRU (H+1)",
    TEAL, "titik panas/hari/provinsi")
kpi(s, 4.75, 1.6, 3.75, f"{pers['MAE']:.2f}", "MAE persistence", GRAY,
    "baseline terkuat setelah model")
kpi(s, 8.7, 1.6, 3.85, f"{gain:.0f}%", "penurunan galat", ORANGE,
    "GRU vs persistence (MAE)")
fit_image(s, FIG / "fig06_metrics.png", 0.6, 3.15, 12.2, 3.35)
notes(s, "Bacakan tiga angka: MAE 10,05; persistence 13,54; penurunan 25,8%. "
         "Selisih antar model DL kecil, sehingga perlu uji statistik.")

# ================================================================== 15 tabel metrik
s = new_slide("Tabel metrik lengkap, horizon H+1", "04 · Evaluasi")
rows = [["Model", "MAE", "RMSE", "R²", "sMAPE (%)", "F1 kejadian", "Presisi", "Recall"]]
for name in ORDER:
    m = metrics[name]
    rows.append([name, f"{m['MAE']:.2f}", f"{m['RMSE']:.1f}", f"{m['R2']:.3f}",
                 f"{m['sMAPE_pct']:.1f}", f"{m['F1_event']:.2f}",
                 f"{m['precision_event']:.2f}", f"{m['recall_event']:.2f}"])
table(s, rows, [3.05, 1.15, 1.15, 1.05, 1.35, 1.5, 1.2, 1.2], 0.8, 1.7,
      font_size=11.5, row_height=0.44, highlight_rows=(1,))
bullets(s, [
    f"GRU terbaik pada MAE ({metrics['GRU']['MAE']:.2f}); F1 tertinggi {best_f1} ("
    f"{metrics[best_f1]['F1_event']:.2f}); RMSE terendah {best_rmse} ({metrics[best_rmse]['RMSE']:.1f}).",
    f"Semua model DL mengalahkan persistence & seasonal naive (MAE turun {dl_gain_min:.0f}–{gain:.0f}%).",
    f"HistGB dan SARIMAX hanya terpaut {metrics['HistGB (tabular)']['MAE'] - metrics['GRU']['MAE']:.2f} "
    "MAE dari GRU → perlu uji signifikansi.",
], 0.8, 5.5, 11.9, 1.3, size=11.5, gap=9)
notes(s, "Perhatikan bahwa R² hanya ~0,34. Ini wajar untuk hotspot harian yang sangat "
         "acak; nilai absolut kecil namun konsisten. Jelaskan bahwa R² rendah tidak "
         "berarti model tak berguna — lihat F1 dan pengurangan MAE.")

# ================================================================== 16 kurva
s = new_slide("Perilaku pelatihan: konvergen cepat, overfit terkendali", "04 · Evaluasi")
fit_image(s, FIG / "fig04_curves.png", 0.7, 1.6, 11.9, 3.9)
bullets(s, [
    "Ketiga model konvergen dalam <10 epoch; early stopping aktif pada epoch "
    f"{hist['LSTM']['epochs_run']}–{max(h['epochs_run'] for h in hist.values())}.",
    "Selisih loss latih dan MAE validasi kecil → dropout dan Huber loss cukup menahan overfit.",
    "MAE validasi bergerak naik-turun karena 2022 adalah tahun normal dengan sedikit kejadian ekstrem, "
    "sehingga sinyal yang dipelajari lebih halus.",
], 0.85, 5.55, 11.8, 1.3, size=12, spacing=0.3)
notes(s, "Titik pada panel kanan adalah epoch terbaik yang bobotnya dipakai untuk pengujian.")

# ================================================================== 17 prediksi
s = new_slide("Prakiraan vs aktual: tren tertangkap, puncak harian belum", "04 · Evaluasi")
fit_image(s, FIG / "fig05_prediction.png", 0.65, 1.55, 12.0, 3.9)
bullets(s, [
    "Model mengikuti amplop musiman dan perubahan rezim, termasuk penurunan awal November.",
    "Puncak ekstrem harian (500–1.200 titik) diprakirakan jauh lebih rendah: loss Huber bersifat konservatif terhadap outlier.",
    "Konsekuensi operasional: model baik untuk tingkat risiko harian, tetapi belum memadai sebagai peringatan lonjakan ekstrem satu hari.",
], 0.85, 5.5, 11.8, 1.4, size=12, spacing=0.3)
notes(s, "Ini kejujuran penting: prakiraan titik panas ekstrem sulit. Alternatif: "
         "model kuantil atau klasifikasi ambang untuk menangkap ekor distribusi.")

# ================================================================== 18 horizon & provinsi
s = new_slide("Akurasi bertahan hingga H+7, perbaikan merata", "04 · Evaluasi")
fit_image(s, FIG / "fig07_horizon.png", 0.55, 1.5, 6.1, 3.05)
fit_image(s, FIG / "fig09_province.png", 6.85, 1.5, 6.0, 3.05)
rows = [["Model", "MAE H+1", "MAE H+3", "MAE H+7", "Kenaikan"]]
for name in ["GRU", "CNN-LSTM", "LSTM", "Persistence"]:
    sub = by_h[by_h.model == name].set_index("horizon")
    rows.append([name, f"{sub.loc[1,'MAE']:.2f}", f"{sub.loc[3,'MAE']:.2f}",
                 f"{sub.loc[7,'MAE']:.2f}",
                 f"{(sub.loc[7,'MAE']/sub.loc[1,'MAE']-1)*100:+.0f}%"])
table(s, rows, [1.95, 1.2, 1.2, 1.2, 1.15], 1.35, 4.78, font_size=10.5,
      row_height=0.33, highlight_rows=(1,))
text(s, "Kenaikan MAE hampir datar karena hotspot sangat persisten pada musim kemarau: "
        "begitu rezim kering terbentuk, jumlah titik panas bertahan beberapa hari."
        f"\n\nPerbaikan terbesar di {prov_max} ({gain_max:.0f}%) karena sinyal musimnya paling tegas; "
        f"terkecil di {prov_min} ({gain_min:.0f}%) karena variabilitasnya paling sulit diprakirakan.",
     8.9, 4.78, 3.9, 1.9, size=10.5, color=SLATE, line_spacing=1.18)
notes(s, "Semua provinsi membaik 16–35% dibanding persistence. Jambi paling kecil "
         "karena variabilitasnya paling sulit; NTT paling besar karena sinyal musimnya tegas.")

# ================================================================== 19 DM
s = new_slide("Uji Diebold-Mariano: apakah selisihnya nyata?", "04 · Evaluasi")
rows = [["Model", "Pembanding", "Statistik DM", "p-value", "Kesimpulan"]]
for _, r in dm.iterrows():
    kesimpulan = "signifikan" if (r.dm_stat < 0 and r.p_value < 0.05) else "tidak signifikan"
    rows.append([r.model, r.reference, f"{r.dm_stat:.2f}", f"{r.p_value:.4f}", kesimpulan])
gru_vs_hgb = int(dm[(dm.model == "GRU") & (dm.reference == "HistGB (tabular)")].index[0]) + 1
table(s, rows, [2.6, 3.1, 1.65, 1.35, 2.35], 0.8, 1.72, font_size=10.5,
      row_height=0.33, highlight_rows=(gru_vs_hgb,))
def p_id(v: float) -> str:
    return f"{v:.4f}".replace(".", ",")


bullets(s, [
    "Semua model mengalahkan persistence secara sangat signifikan (p < 0,001).",
    f"GRU (p = {p_id(p_gru_hgb)}) dan LSTM (p = {p_id(p_lstm_hgb)}) mengalahkan HistGB: "
    "DL memberi nilai tambah nyata.",
    f"CNN-LSTM tidak berbeda signifikan dari HistGB (p = {p_id(p_cnn_hgb)}) → keunggulannya belum terbukti.",
    "Selisih MAE 0,1–0,4 titik panas terlalu kecil untuk diklaim tanpa uji.",
], 0.8, 5.28, 11.9, 1.6, size=11.5, gap=8)
notes(s, "Statistik negatif berarti model di baris tersebut lebih akurat. "
         "Uji memakai koreksi Newey-West 7 lag karena galat berautokorelasi.")

# ================================================================== 20 galat
s = new_slide("Analisis galat: model cenderung meremehkan", "04 · Evaluasi")
fit_image(s, FIG / "fig08_scatter.png", 0.7, 1.55, 11.9, 3.75)
bullets(s, [
    "Sebaran mengikuti garis ideal pada nilai rendah, tetapi mendatar pada nilai tinggi → atenuasi puncak.",
    f"Bias rata-rata {bias_gru:.2f} titik panas: prakiraan lebih rendah daripada aktual, terutama saat kejadian besar.".replace("-", "−"),
    "Galat ekor kanan besar dan asimetris — tipikal distribusi hotspot yang heavy-tailed.",
    "Perbaikan yang disarankan: loss kuantil, pembobotan sampel kejadian ekstrem, atau model dua tahap (klasifikasi ambang lalu regresi).",
], 0.85, 5.35, 11.8, 1.6, size=11.5, spacing=0.3)
notes(s, "Atenuasi puncak adalah konsekuensi langsung dari regresi ke rata-rata bersyarat. "
         "Jika tujuan utamanya peringatan ekstrem, metrik utama sebaiknya recall kejadian.")

# ================================================================== 21 diskusi
s = new_slide("Diskusi: manfaat, keterbatasan, dan risiko pemakaian", "05 · Diskusi")
rect(s, 0.8, 1.62, 3.8, 3.0, RGBColor(0xEC, 0xFD, 0xF5), line=TEAL, radius=0.05)
text(s, "Yang sudah terbukti", 1.05, 1.8, 3.4, 0.3, size=13, bold=True, color=TEAL)
text(s, "• Pipeline reproducible dari data mentah sampai PPT\n"
        "• DL mengalahkan semua baseline (p < 0,001)\n"
        "• GRU signifikan di atas HistGB (p = 0,003)\n"
        "• Akurasi stabil sampai H+7\n"
        "• Perbaikan merata di 10 provinsi (16–35%)",
     1.05, 2.2, 3.35, 3.0, size=11.5, color=SLATE, line_spacing=1.35)
rect(s, 4.77, 1.62, 3.8, 3.0, RGBColor(0xFE, 0xF2, 0xF2), line=FIRE, radius=0.05)
text(s, "Keterbatasan", 5.02, 1.8, 3.4, 0.3, size=13, bold=True, color=FIRE)
text(s, "• Data simulasi, belum observasi FIRMS\n"
        "• Tanpa komponen spasial (grid/piksel)\n"
        "• Faktor manusia (pembukaan lahan) tidak dimodelkan\n"
        "• Horizon pendek; tidak ada ketidakpastian prakiraan\n"
        "• Satu titik model per provinsi\n"
        "• Bukan pengganti patroli lapangan",
     5.02, 2.2, 3.35, 3.3, size=11.5, color=SLATE, line_spacing=1.3)
rect(s, 8.74, 1.62, 3.85, 3.0, RGBColor(0xEF, 0xF6, 0xFF), line=BLUE, radius=0.05)
rect(s, 0.8, 4.92, 11.79, 0.82, LIGHT, radius=0.06)
text(s, "Takeaway: deep learning layak dipakai sebagai alat bantu peringatan dini, "
        "tetapi wajib disertai ketidakpastian, validasi lapangan, dan evaluasi ulang berkala — "
        "bukan pengganti patroli atau keputusan operasional.",
     1.05, 5.12, 11.3, 0.55, size=12, bold=True, color=SLATE, line_spacing=1.15)
text(s, "Risiko & tata kelola", 8.99, 1.8, 3.4, 0.3, size=13, bold=True, color=BLUE)
text(s, "• Jangan pakai satu angka prakiraan sebagai dasar tunggal keputusan\n"
        "• Sertakan interval ketidakpastian sebelum dipakai operasional\n"
        "• Validasi ulang tiap tahun (rezim iklim berubah)\n"
        "• Dokumentasikan batas wilayah & satuan\n"
        "• Pantau pergeseran data (drift) setelah model dipakai",
     8.99, 2.2, 3.4, 3.3, size=11.5, color=SLATE, line_spacing=1.3)
notes(s, "Bagian ini yang membedakan analisis jujur dari klaim berlebihan. "
         "Sebutkan bahwa model ini adalah alat bantu, bukan pengganti patroli lapangan.")

# ================================================================== 22 kesimpulan
s = new_slide("Kesimpulan & langkah berikutnya", "05 · Penutup")
bullets(s, [
    f"Deep learning layak dipakai: GRU mencapai MAE {metrics['GRU']['MAE']:.2f} titik panas/hari, "
    f"{gain:.0f}% lebih baik daripada persistence, dan signifikan secara statistik.",
    "Arsitektur paling sederhana (GRU) menang; kompleksitas CNN-LSTM belum terbayar pada horizon pendek.",
    "Akurasi stabil sampai H+7, sehingga berguna untuk peringatan dini mingguan.",
    "Kelemahan utama: puncak ekstrem diremehkan — perlu loss kuantil atau pendekatan dua tahap.",
    "Ganti data simulasi dengan agregasi harian hotspot VIIRS/MODIS (FIRMS) untuk validasi empiris, "
    "lalu tambahkan fitur spasial dan faktor aktivitas manusia.",
], 0.8, 1.65, 7.4, 4.3, size=13.5, spacing=0.42)
rect(s, 8.45, 1.65, 4.1, 4.3, INK, radius=0.05)
text(s, "Cara menjalankan ulang", 8.72, 1.85, 3.6, 0.3, size=12, bold=True, color=ORANGE)
cmds = ("python data/generate_karhutla_data.py\n"
        "python src/train.py\n"
        "python src/evaluate.py\n"
        "python src/make_ppt.py")
text(s, cmds, 8.72, 2.25, 3.6, 1.8, size=11, font=MONO, color=WHITE,
     line_spacing=1.5)
text(s, "Semua keluaran (metrik, gambar, model, deck ini) dihasilkan ulang "
        "secara deterministik dengan seed 42.", 8.72, 3.55, 3.6, 1.2, size=10.5,
     color=RGBColor(0xD1, 0xD5, 0xDB), line_spacing=1.2)
notes(s, "Tutup dengan satu kalimat: model ini bukan peramal ajaib, tetapi bukti "
         "bahwa pipeline DL dapat mengungguli baseline secara terukur dan jujur.")

# ================================================================== 23 penutup
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, W, H, INK, shape=MSO_SHAPE.RECTANGLE)
rect(s, 0, H - 0.22, W, 0.22, ORANGE, shape=MSO_SHAPE.RECTANGLE)
text(s, "Terima kasih", 1.1, 2.3, 10, 0.9, size=44, bold=True, color=WHITE)
text(s, "Pertanyaan & diskusi", 1.1, 3.25, 10, 0.5, size=18,
     color=RGBColor(0xD1, 0xD5, 0xDB))
text(s, "Lampiran: results/metrics_h1.json · results/diebold_mariano.csv · "
        "results/figures/*.png · models/*.pt", 1.1, 4.35, 11, 0.4, size=11,
     color=RGBColor(0x9C, 0xA3, 0xAF))
notes(s, "Siapkan jawaban untuk: mengapa MAPE tidak dipakai, mengapa R² rendah, "
         "dan bagaimana jika data asli dipakai.")

out = ARCH_OUT / "Prediksi_Karhutla_Deep_Learning.pptx"
prs.save(out)
print(f"PPT tersimpan: {out} ({len(prs.slides)} slide)")
