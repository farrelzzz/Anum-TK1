"""Bagian vi: evaluasi RMSE train dan test dengan parameter kedua metode."""
from __future__ import annotations

from setar_analysis_common import (
    RESULTS, fit_models, load_data, select_best_method, write_csv,
)


def main() -> None:
    data = load_data()
    models = fit_models(data)
    best = select_best_method(models)
    out = RESULTS / "vi"
    rows = [
        [name, f"{models.train_rmse[name]:.12g}", f"{models.test_rmse[name]:.12g}",
         f"{models.residuals[name]:.12g}"]
        for name in models.coefficients
    ]
    write_csv(out / "evaluasi_vi.csv", ["Metode", "RMSE_train", "RMSE_test", "Residual_train_L2"], rows)
    write_csv(out / "model_terpilih.csv", ["Metode_terpilih"], [[best]])
    prediction_rows = []
    for date, actual, qr, normal in zip(
        data.train_target_dates, data.b, models.train_predictions["QR Householder"],
        models.train_predictions["Persamaan Normal"],
    ):
        prediction_rows.append([date, "train", actual, qr, normal])
    for date, actual, qr, normal in zip(
        data.test_dates, data.test_returns, models.test_predictions["QR Householder"],
        models.test_predictions["Persamaan Normal"],
    ):
        prediction_rows.append([date, "test", actual, qr, normal])
    write_csv(
        out / "prediksi_vi.csv",
        ["Tanggal", "Segmen", "Return_aktual", "Prediksi_QR", "Prediksi_Persamaan_Normal"],
        prediction_rows,
    )
    report = f"""# Nomor 2 bagian vi — Evaluasi out-of-sample

Model dilatih hanya dengan data train (\\(A\\in\\mathbb{{R}}^{{{data.A.shape[0]}\\times 6}}\\)). Setiap prediksi test adalah one-step ahead: lag awal menggunakan dua return train terakhir, target pertama memasukkan perubahan dari harga train terakhir ke harga test pertama, dan target berikutnya menggunakan return aktual sebelumnya. Dengan demikian {len(data.test_prices)} harga test menghasilkan {len(data.test_returns)} return target test.

| Metode | RMSE train | RMSE test | Residual train \\(L_2\\) |
|---|---:|---:|---:|
| QR Householder | {models.train_rmse['QR Householder']:.12g} | {models.test_rmse['QR Householder']:.12g} | {models.residuals['QR Householder']:.12g} |
| Persamaan Normal | {models.train_rmse['Persamaan Normal']:.12g} | {models.test_rmse['Persamaan Normal']:.12g} | {models.residuals['Persamaan Normal']:.12g} |

Metode dengan RMSE test terkecil adalah **{best}**. Jika RMSE kedua metode berbeda tidak lebih dari toleransi numerik \\(10^{{-12}}\\), QR dipilih karena lebih stabil secara numerik daripada persamaan normal. Pilihan ini dan semua prediksi per tanggal tersimpan dalam berkas CSV bagian vi.
"""
    out.mkdir(parents=True, exist_ok=True)
    (out / "laporan_vi.md").write_text(report, encoding="utf-8")
    print(f"RMSE train/test: {rows}")
    print(f"Metode terpilih: {best}")


if __name__ == "__main__":
    main()
