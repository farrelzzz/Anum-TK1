# =====================================================================
# ii) Investigasi Isu Matriks
# =====================================================================
"""
Nomor 2 Bagian ii - Investigasi Isu Matriks
Memeriksa isu yang mungkin terjadi dalam menjalankan perhitungan matriks pada data stock_train.csv
    - Mencari risiko pivot nol (Isu pembagian oleh nol)
    - Mencari isu keterbatasan presisi floating point yang mendekati nol
"""

import numpy as np
from pathlib import Path
import csv
import i_formulate_overdetermined_matrix as SETAR

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "hasil" / "ii"

def trigger_precision_loss(A):
    A_32 = A.astype(np.float32)
    A_64 = A.astype(np.float64)

    AtA_32 = np.dot(A_32.T, A_32)
    AtA_64 = np.dot(A_64.T, A_64)

    # Jika matriks stabil, error_diff seharusnya bernilai 0.
    error_diff = np.max(np.abs(AtA_64 - AtA_32))
    
    return error_diff

if __name__ == "__main__":
    file_path = HERE / "stock_train.csv"
    prices = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=1)
    returns = SETAR.return_price(prices)
    A, b = SETAR.create_matrix_SETAR(returns)
    max_error = trigger_precision_loss(A)
    A64 = A.astype(np.float64)
    gram64 = A64.T @ A64
    gram32 = A.astype(np.float32).T @ A.astype(np.float32)
    relative_error = float(max_error / max(1.0, np.max(np.abs(gram64))))
    rank = int(np.linalg.matrix_rank(A64))
    condition = float(np.linalg.cond(A64, 2))
    column_norms = np.sqrt(np.sum(A64 * A64, axis=0))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "diagnostik_ii.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Metrik", "Nilai"])
        writer.writerows([
            ["jumlah_baris", A.shape[0]], ["jumlah_parameter", A.shape[1]],
            ["rank_A", rank], ["condition_number_2_A", condition],
            ["galat_absolut_maks_AtA_float32_vs_float64", max_error],
            ["galat_relatif_maks_AtA", relative_error],
            *[[f"norma_kolom_{i + 1}", value] for i, value in enumerate(column_norms)],
        ])
    report = f"""# Nomor 2 bagian ii — Investigasi isu numerik

Matriks desain berukuran \\({A.shape[0]}\\times{A.shape[1]}\\), berpangkat {rank} dari 6 parameter. Condition number 2-norm \\(\\kappa_2(A)\\) adalah {condition:.8g}. Norma kolom beragam: minimum {column_norms.min():.8g}, maksimum {column_norms.max():.8g}; perbedaan skala ini dapat membuat pengujian pivot berdasarkan ambang absolut saja kurang andal.

Membandingkan pembentukan \\(A^TA\\) dengan float32 dan float64 menghasilkan galat absolut maksimum {max_error:.8g} dan galat relatif maksimum {relative_error:.8g}. Galat float32/float64 tidak seharusnya diharapkan nol; perbedaan tersebut adalah akibat pembulatan floating point. Risiko yang lebih besar muncul ketika persamaan normal membentuk \\(A^TA\\), sebab \\(\\kappa_2(A^TA)\\approx\\kappa_2(A)^2\\), sehingga galat pembulatan lebih kuat memengaruhi solusi. Rank penuh pada data ini menunjukkan tidak ada kolom yang redundan secara numerik pada toleransi rank NumPy.

Mitigasi teknis yang disarankan:

1. Simpan harga, return, dan matriks dalam float64; jangan menurunkan presisi sebelum membentuk matriks.
2. Gunakan QR Householder sebagai solver utama karena transformasi ortogonal menghindari pembentukan \\(A^TA\\).
3. Bila persamaan normal dipakai sebagai pembanding, gunakan partial pivoting dan tolak pivot yang nol atau terlalu kecil terhadap skala matriks.
4. Periksa rank, condition number, norma kolom, serta residual solusi; penskalaan kolom dapat dipertimbangkan bila rentang skala menghambat perhitungan, dengan parameter dikembalikan ke skala semula.

Metrik rinci tersedia dalam `diagnostik_ii.csv`.
"""
    (OUTPUT / "laporan_ii.md").write_text(report, encoding="utf-8")
    print(f"Galat absolut float32/float64 A^T A: {max_error:.10g}")
    print(f"Rank(A)={rank}/{A.shape[1]}; kappa_2(A)={condition:.8g}")
    print(f"Laporan tersimpan di {OUTPUT / 'laporan_ii.md'}")
