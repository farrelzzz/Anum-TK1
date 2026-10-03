"""
iii_struktur_matriks.py  (bagian iii -- Identifikasi Struktur Matriks)
====================================================================
Deteksi otomatis lower/upper bandwidth (p, q) dari B, verifikasi bahwa
penggantian baris pertama tidak merusak struktur banded, konversi ke
representasi pita padat, dan perbandingan kebutuhan memori dense vs band.
"""

from __future__ import annotations
import numpy as np


def detect_bandwidth(B: np.ndarray, tol: float = 1e-12) -> tuple[int, int]:
    """
    Deteksi otomatis lower bandwidth p dan upper bandwidth q dari matriks B:
        p = max(i - j) untuk semua (i, j) dengan |B[i,j]| > tol dan i > j
        q = max(j - i) untuk semua (i, j) dengan |B[i,j]| > tol dan j > i
    Dikembalikan (p, q). Jika matriks diagonal murni, p = q = 0.
    """
    nz_i, nz_j = np.nonzero(np.abs(B) > tol)
    if nz_i.size == 0:
        return 0, 0
    diff = nz_i - nz_j  # positif -> di bawah diagonal (lower), negatif -> upper
    p = int(diff.max()) if diff.max() > 0 else 0
    q = int((-diff).max()) if (-diff).max() > 0 else 0
    return p, q


def band_from_dense(B: np.ndarray, p: int, q: int) -> np.ndarray:
    """
    Konversi matriks dense B (N x N, banded dengan lower bandwidth p dan
    upper bandwidth q) menjadi representasi pita padat ala LAPACK:
    array berukuran (p + q + 1) x N, dengan
        ab[q + i - j, j] = B[i, j]   untuk max(0, j-q) <= i <= min(N-1, j+p)
    Baris ke-0 representasi ini adalah diagonal ke-q (paling atas),
    baris terakhir adalah diagonal ke-(-p) (paling bawah).
    Ini HANYA transformasi penyimpanan (reshape), bukan operasi solver.
    """
    N = B.shape[0]
    ab = np.zeros((p + q + 1, N))
    for j in range(N):
        i_lo = max(0, j - q)
        i_hi = min(N - 1, j + p)
        for i in range(i_lo, i_hi + 1):
            ab[q + i - j, j] = B[i, j]
    return ab


def dense_memory_bytes(N: int, itemsize: int = 8) -> int:
    # Estimasi memori matriks dense N x N (byte)
    return N * N * itemsize


def band_memory_bytes(N: int, p: int, q: int, itemsize: int = 8) -> int:
    #Estimasi memori representasi pita (p+q+1) x N (byte)
    return (p + q + 1) * N * itemsize


def band_preserved_after_row_replace(T: np.ndarray, tol: float = 1e-12) -> bool:
    """
    Verifikasi klaim di poin iii: penggantian baris pertama A (-> B) TIDAK
    merusak struktur banded, karena baris yang diganti ([1,0,...,0]) justru
    punya bandwidth lebih SEMPIT (hanya elemen diagonal) daripada baris
    aslinya. Fungsi ini membandingkan bandwidth A=I-T^T (sebelum ganti
    baris) dengan B (sesudah ganti baris) hasilnya harus sama atau B
    punya bandwidth <= A, membuktikan band tidak melebar.
    """
    from ii_formulasi import build_B_b

    N = T.shape[0]
    A = np.eye(N) - T.T
    B, _ = build_B_b(T)
    p_A, q_A = detect_bandwidth(A, tol)
    p_B, q_B = detect_bandwidth(B, tol)
    return (p_B <= p_A) and (q_B <= q_A)


if __name__ == "__main__":
    from i_validasi_data import load_T
    from ii_formulasi import build_B_b

    files = [
        "T_16.csv",
        "T_32.csv",
        "T_64.csv",
        "T_128.csv",
        "T_256.csv",
        "T_512.csv"
    ]
    
    print(f"{'file':8}{'N':>5}{'p':>5}{'q':>4}{'band_ok':>12}{'dense KB':>12}{'band KB':>8}")
    for f in files:
        T = load_T(f)
        N = T.shape[0]
        B, b = build_B_b(T)
        p, q = detect_bandwidth(B)
        band_ok = band_preserved_after_row_replace(T)
        dense_kb = dense_memory_bytes(N) / 1024
        band_kb = band_memory_bytes(N, p, q) / 1024
        print(f"{f:<8}{N:>5}{p:>5}{q:>4}{str(band_ok):>9}{dense_kb:>10.1f}{band_kb:>9.1f}")