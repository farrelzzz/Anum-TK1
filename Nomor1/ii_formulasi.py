"""
ii_formulasi.py  (bagian ii -- Formulasi dan Penanganan Singularitas)
===================================================================
Bentuk matriks B dan vektor b dari T, sesuai definisi di soal, serta
utilitas untuk membuktikan secara numerik bahwa B hasil penggantian baris
pertama nonsingular (mendukung argumen di laporan bagian ii).

    A = I - T^T                (A pi = 0, A singular)
    B[0, :] = [1, 0, ..., 0]   (baris pertama diganti)
    B[i, :] = A[i, :]  for i = 1..N-1
    b = [1, 0, ..., 0]^T

Solusi z dari B z = b lalu dinormalkan pi = z / (1^T z), karena B z = b
hanya menjamin komponen pertama z sebanding dengan syarat normalisasi
yang kita "titipkan" (z_1-like constraint), sedangkan skala z secara
keseluruhan belum tentu memenuhi 1^T pi = 1. Normalisasi memaksa jumlah
komponen menjadi tepat 1, sesuai syarat pada soal, yaitu 2.
"""

from __future__ import annotations
import numpy as np

def build_B_b(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Bentuk B dan b sesuai definisi di soal
    N = T.shape[0]
    A = np.eye(N) - T.T
    B = A.copy()
    B[0, :] = 0.0
    B[0, 0] = 1.0
    b = np.zeros(N)
    b[0] = 1.0
    return B, b


def check_nonsingular(B: np.ndarray) -> dict:
    """
    Cek nonsingularity B secara numerik 
    Di sini hanya menghitung nilai, bukan menyelesaikan SPL/faktorisasi utama, jadi pakai numpy di sini (harusnya masih) sesuai aturan):
      - rank(B) via numpy.linalg.matrix_rank (SVD-based, hanya untuk cek, bukan untuk solve)
      - condition number via numpy.linalg.cond (default 2-norm)
    B dengan rank penuh (= N) dan condition number berhingga (tidak inf/nan)
    dianggap nonsingular secara numerik.
    """
    N = B.shape[0]
    rank = int(np.linalg.matrix_rank(B))
    cond = float(np.linalg.cond(B))
    return {
        "N": N,
        "rank": rank,
        "full_rank": rank == N,
        "cond_2norm": cond,
        "numerically_nonsingular": (rank == N) and np.isfinite(cond),
    }


if __name__ == "__main__":
    from i_validasi_data import load_T

    files = [
        "T_16.csv",
        "T_32.csv",
        "T_64.csv",
        "T_128.csv",
        "T_256.csv",
        "T_512.csv"
    ]

    print(f"{'file':<8}{'N':>5}{'rank=N':>11}{'cond(B)':>10}")
    for f in files:
        T = load_T(f)
        B, b = build_B_b(T)
        sing = check_nonsingular(B)
        print(f"{f:<8}{sing['N']:>6}{str(sing['full_rank']):>8}{sing['cond_2norm']:>14.3e}")