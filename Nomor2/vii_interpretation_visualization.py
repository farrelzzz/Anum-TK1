"""Bagian vii: interpretasi koefisien dan grafik overlay train/test."""
from __future__ import annotations

from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from setar_analysis_common import RESULTS, fit_models, load_data, select_best_method, write_csv


def direction(value: float) -> str:
    if value > 0.0:
        return "positif: pengaruh searah dengan return lag (kecenderungan momentum)"
    if value < 0.0:
        return "negatif: pengaruh berlawanan arah dengan return lag (kecenderungan mean-reverting)"
    return "nol: tidak ada pengaruh linear dari lag ini"


def main() -> None:
    data = load_data()
    models = fit_models(data)
    method = select_best_method(models)
    coefficients = models.coefficients[method]
    train_hat = models.train_predictions[method]
    test_hat = models.test_predictions[method]
    out = RESULTS / "vii"
    out.mkdir(parents=True, exist_ok=True)

    train_dates = [datetime.strptime(date, "%Y-%m-%d") for date in data.train_target_dates]
    all_date_strings = data.train_target_dates + data.test_dates
    all_dates = [datetime.strptime(date, "%Y-%m-%d") for date in all_date_strings]
    actual = np.concatenate((data.b, data.test_returns))
    estimated = np.concatenate((train_hat, test_hat))
    split = len(data.b)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train_dates, data.b, label="Return aktual train", linewidth=1)
    ax.plot(train_dates, train_hat, label=f"Estimasi train ({method})", linewidth=1)
    ax.set_title("SETAR: return aktual dan estimasi pada data train")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Return harian")
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "overlay_train_vii.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(all_dates, actual, label="Return aktual", linewidth=1)
    ax.plot(all_dates, estimated, label=f"Estimasi SETAR ({method})", linewidth=1)
    ax.axvline(all_dates[split], color="black", linestyle="--", label="Batas train/test")
    ax.set_title("SETAR: overlay kontinu train dan test")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Return harian")
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "overlay_train_test_vii.png", dpi=160)
    plt.close(fig)

    write_csv(
        out / "koefisien_vii.csv",
        ["Parameter", "Nilai", "Interpretasi"],
        [
            ["alpha1", coefficients[0], "intersep rezim bullish"],
            ["phi11", coefficients[1], direction(coefficients[1])],
            ["phi12", coefficients[2], direction(coefficients[2])],
            ["alpha2", coefficients[3], "intersep rezim bearish"],
            ["phi21", coefficients[4], direction(coefficients[4])],
            ["phi22", coefficients[5], direction(coefficients[5])],
        ],
    )
    report = f"""# Nomor 2 bagian vii — Interpretasi finansial dan visualisasi

Metode yang digunakan adalah **{method}**, dipilih memakai aturan evaluasi bagian vi. RMSE train = {models.train_rmse[method]:.8g}; RMSE test = {models.test_rmse[method]:.8g}.

Model SETAR hasil estimasi:

- **Bullish** (\\(R_{{t-1}}\\ge 0\\)): \\(\\widehat{{R}}_t={coefficients[0]:.10g}+({coefficients[1]:.10g})R_{{t-1}}+({coefficients[2]:.10g})R_{{t-2}}\\).
- **Bearish** (\\(R_{{t-1}}<0\\)): \\(\\widehat{{R}}_t={coefficients[3]:.10g}+({coefficients[4]:.10g})R_{{t-1}}+({coefficients[5]:.10g})R_{{t-2}}\\).

Interpretasi koefisien lag:

| Rezim | Koefisien | Nilai | Interpretasi tanda |
|---|---|---:|---|
| Bullish | \\(\\phi_{{1,1}}\\) | {coefficients[1]:.10g} | {direction(coefficients[1])} |
| Bullish | \\(\\phi_{{1,2}}\\) | {coefficients[2]:.10g} | {direction(coefficients[2])} |
| Bearish | \\(\\phi_{{2,1}}\\) | {coefficients[4]:.10g} | {direction(coefficients[4])} |
| Bearish | \\(\\phi_{{2,2}}\\) | {coefficients[5]:.10g} | {direction(coefficients[5])} |

Tanda koefisien menyatakan arah pengaruh linear parsial dari return lag, bukan jaminan bahwa tren pasar akan berlanjut atau berbalik. Grafik `overlay_train_vii.png` membandingkan aktual dan estimasi train. `overlay_train_test_vii.png` menggabungkan kedua segmen waktu dengan garis batas yang jelas.
"""
    (out / "laporan_vii.md").write_text(report, encoding="utf-8")
    print(f"Model terpilih: {method}")
    print(f"Grafik dan laporan disimpan di {out}")


if __name__ == "__main__":
    main()
