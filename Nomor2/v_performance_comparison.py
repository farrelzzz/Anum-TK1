"""Bagian v: perbandingan teori dan galat QR Householder vs Persamaan Normal."""
from __future__ import annotations

import numpy as np

from setar_analysis_common import RESULTS, fit_models, load_data, write_csv


def main() -> None:
    data = load_data()
    models = fit_models(data)
    m, n = data.A.shape
    item_size = data.A.dtype.itemsize

    # FLOPs konvensional: Householder QR + penerapan refleksi ke satu RHS;
    # persamaan normal membentuk Gram matrix penuh lalu menyelesaikan SPL n x n.
    flops_qr = 2 * m * n**2 - (2 / 3) * n**3 + 2 * m * n
    flops_normal = 2 * m * n**2 + (2 / 3) * n**3
    common_bytes = (m * n + m) * item_size
    # R, transformed RHS, one active reflector, and solution vector.
    qr_workspace = (m * n + 2 * m + n) * item_size
    qr_explicit_q = m * m * item_size
    # A^T A and A^T b plus the copies made by solve_spl in section iii.
    normal_workspace = (2 * n * n + 3 * n) * item_size

    rows = [
        ["QR Householder", f"{flops_qr:.0f}", str(qr_workspace),
         str(qr_workspace + qr_explicit_q), f"{models.residuals['QR Householder']:.12g}"],
        ["Persamaan Normal", f"{flops_normal:.0f}", str(normal_workspace),
         str(normal_workspace), f"{models.residuals['Persamaan Normal']:.12g}"],
    ]
    out = RESULTS / "v"
    write_csv(
        out / "perbandingan_v.csv",
        ["Metode", "FLOPs_perkiraan", "Workspace_tanpa_Q_eksplisit_byte",
         "Workspace_dengan_Q_eksplisit_byte", "Residual_train_L2"],
        rows,
    )
    gap = abs(models.residuals["QR Householder"] - models.residuals["Persamaan Normal"])
    report = f"""# Nomor 2 bagian v — Perbandingan performa dan pengaruh outlier

Matriks train berukuran **{m} x {n}**. Penyimpanan masukan bersama \\(A,b\\) adalah {common_bytes:,} byte pada float64 dan tidak dihitung sebagai workspace tambahan.

| Metode | FLOPs perkiraan | Workspace tanpa \\(Q\\) eksplisit (byte) | Workspace dengan \\(Q\\) eksplisit (byte) | Residual train \\(L_2\\) |
|---|---:|---:|---:|---:|
| QR Householder | {flops_qr:.0f} | {qr_workspace:,} | {qr_workspace + qr_explicit_q:,} | {models.residuals['QR Householder']:.12g} |
| Persamaan Normal | {flops_normal:.0f} | {normal_workspace:,} | {normal_workspace:,} | {models.residuals['Persamaan Normal']:.12g} |

Perkiraan QR menggunakan \\(2mn^2-\\frac{{2}}{{3}}n^3\\) untuk faktorisasi dan sekitar \\(2mn\\) untuk mentransformasikan ruas kanan. Persamaan Normal menggunakan sekitar \\(2mn^2\\) untuk membentuk \\(A^TA\\), lalu \\(\\frac{{2}}{{3}}n^3\\) untuk eliminasi pada sistem kecil. Estimasi workspace memperhitungkan array kerja solver (tidak termasuk masukan bersama A dan b dan temporary internal NumPy); penyimpanan matriks ortogonal penuh menambah \\(m^2\\) elemen ({qr_explicit_q:,} byte). Implementasi solver QR di bagian iv menerapkan refleksi secara implisit dan hanya membentuk \\(H_1\\) eksplisit untuk verifikasi.

Condition number yang dihitung oleh prosedur bagian iii: \\(\\kappa_2(A)={models.condition_A:.8g}\\) dan \\(\\kappa_2(A^TA)={models.condition_AtA:.8g}\\). Selisih residual kedua metode adalah {gap:.6e}. QR secara umum lebih stabil karena tidak membentuk \\(A^TA\\), yang menguadratkan condition number.

Outlier ekstrem pada return berpengaruh kuadrat terhadap fungsi objektif least squares. Satu observasi ekstrem dapat menaikkan residual dan menggeser koefisien agar lebih mengikuti observasi tersebut. Karena itu residual besar dapat mencerminkan shock pada data, bukan hanya kekurangan solver; pemeriksaan return ekstrem dan evaluasi out-of-sample tetap diperlukan.
"""
    out.mkdir(parents=True, exist_ok=True)
    (out / "laporan_v.md").write_text(report, encoding="utf-8")
    print(f"Bagian v selesai: {out}")
    print(f"kappa(A)={models.condition_A:.8g}; kappa(A^T A)={models.condition_AtA:.8g}")
    for row in rows:
        print(", ".join(row))


if __name__ == "__main__":
    main()
