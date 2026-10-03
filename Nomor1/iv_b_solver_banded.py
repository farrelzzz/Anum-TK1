"""
solver_banded.py
=================
Solver (iv-b): algoritma gaya Thomas yang DIGENERALISASI untuk matriks
banded dengan lower bandwidth p dan upper bandwidth q sembarang (bukan
hanya tridiagonal p=q=1). Ini perlu karena B pada dataset punya p=1, q=2
(lihat formulate.detect_bandwidth) -- bukan tridiagonal murni.

Hanya menyimpan (2p+q+1) x N nilai (representasi pita + ruang cadangan
untuk fill-in akibat pivoting), TIDAK PERNAH membentuk matriks dense N x N.
Tidak memakai numpy.linalg.solve / scipy.linalg.solve_banded / library
solver-SPL lain -- murni eliminasi manual di atas array numpy sebagai
container.

Kenapa ruang (2p+q+1) baris, bukan (p+q+1) saja?
  Saat partial pivoting menukar baris j dengan baris j+i (i <= p), baris
  yang naik ke posisi j bisa punya entri sejauh kolom j+p (bukan cuma
  j+q seperti baris asli di posisi j). Jadi upper bandwidth EFEKTIF bisa
  melebar dari q menjadi (p+q) selama faktorisasi. Supaya tidak perlu
  realokasi/fallback ke dense, kita sediakan p baris ekstra di atas sejak
  awal (= workspace fill-in), mengikuti skema band-storage ala LAPACK
  (dgbtrf/dgbsv).

Pseudocode:
    function banded_lu_partial_pivot(B, p, q):
        AB <- band_storage(B, p, q)   # (2p+q+1) x N, p baris atas = 0 (fill)
        perm <- [0, ..., N-1]
        for j = 0 .. N-1:
            i_hi <- min(N-1, j+p)
            # cari pivot di antara baris j..i_hi pada kolom j
            pivot <- argmax_{i=j..i_hi} |AB[row(i,j), j]|
            jika pivot != j: tukar baris j & pivot (hanya kolom j..j+p+q),
                             catat di perm
            for i = j+1 .. i_hi:
                factor <- AB[row(i,j), j] / AB[row(j,j), j]
                simpan factor sbg L[i,j]
                for col = j+1 .. min(j+p+q, N-1):
                    AB[row(i,col), col] -= factor * AB[row(j,col), col]
        return AB, perm

    function solve_banded(B, b, p, q):
        AB, perm <- banded_lu_partial_pivot(B, p, q)
        bp <- b[perm]
        y  <- forward_substitution_banded(AB, bp, p)   # pakai L di AB
        x  <- backward_substitution_banded(AB, y, p, q) # pakai U di AB (lebar p+q)
        return x
"""

from __future__ import annotations
import numpy as np


def build_band_storage(B: np.ndarray, p: int, q: int) -> np.ndarray:
    """
    Simpan B (N x N, lower bw p, upper bw q) ke representasi pita dengan
    p baris ekstra di atas (workspace fill-in untuk pivoting), total
    (2p+q+1) x N. Elemen B[i,j] disimpan di AB[p+q + i - j, j].
    """
    N = B.shape[0]
    AB = np.zeros((2 * p + q + 1, N))
    for j in range(N):
        i_lo = max(0, j - q)
        i_hi = min(N - 1, j + p)
        for i in range(i_lo, i_hi + 1):
            AB[p + q + i - j, j] = B[i, j]
    return AB


def _row(i: int, j: int, p: int, q: int) -> int:
    # Baris penyimpanan di AB untuk elemen matriks (i, j
    return p + q + i - j


def banded_lu_partial_pivot(B: np.ndarray, p: int, q: int):
    """
    Faktorisasi banded-LU dengan partial pivoting, hanya menyimpan pita.
    Return AB (berisi L di bawah diagonal & U di diagonal+atas) dan perm.
    """
    N = B.shape[0]
    AB = build_band_storage(B, p, q)
    perm = np.arange(N)

    for j in range(N):
        i_hi = min(N - 1, j + p)          # baris kandidat pivot (dalam band asli)
        col_hi = min(N - 1, j + p + q)    # batas kanan fill-in akibat pivoting

        # --- cari pivot: |AB[row(i,j), j]| terbesar untuk i = j..i_hi ---
        best_i = j
        best_val = abs(AB[_row(j, j, p, q), j])
        for i in range(j + 1, i_hi + 1):
            val = abs(AB[_row(i, j, p, q), j])
            if val > best_val:
                best_val = val
                best_i = i

        if best_i != j:
            for col in range(j, col_hi + 1):
                r1, r2 = _row(j, col, p, q), _row(best_i, col, p, q)
                AB[r1, col], AB[r2, col] = AB[r2, col], AB[r1, col]
            perm[[j, best_i]] = perm[[best_i, j]]

        diag = AB[_row(j, j, p, q), j]
        if diag == 0.0:
            raise ZeroDivisionError(
                f"Pivot nol di kolom {j} walau sudah partial pivoting -- "
                f"matriks B singular secara numerik."
            )

        # --- eliminasi, hanya di dalam band (+fill) ---
        for i in range(j + 1, i_hi + 1):
            factor = AB[_row(i, j, p, q), j] / diag
            AB[_row(i, j, p, q), j] = factor  # simpan multiplier (L)
            for col in range(j + 1, col_hi + 1):
                AB[_row(i, col, p, q), col] -= factor * AB[_row(j, col, p, q), col]

    return AB, perm


def forward_substitution_banded(AB: np.ndarray, bp: np.ndarray, p: int, q: int) -> np.ndarray:
    # L y = bp, L unit-lower-triangular dgn lower bandwidth p (multiplier di AB
    N = bp.shape[0]
    y = np.zeros(N)
    for i in range(N):
        k_lo = max(0, i - p)
        s = bp[i]
        for k in range(k_lo, i):
            s -= AB[_row(i, k, p, q), k] * y[k]
        y[i] = s  # diagonal L = 1
    return y


def backward_substitution_banded(AB: np.ndarray, y: np.ndarray, p: int, q: int) -> np.ndarray:
    # U x = y, U upper-triangular dgn upper bandwidth EFEKTIF (p+q) akibat fill-in
    N = y.shape[0]
    x = np.zeros(N)
    for i in range(N - 1, -1, -1):
        col_hi = min(N - 1, i + p + q)
        s = y[i]
        for col in range(i + 1, col_hi + 1):
            s -= AB[_row(i, col, p, q), col] * x[col]
        x[i] = s / AB[_row(i, i, p, q), i]
    return x


def solve_banded(B: np.ndarray, b: np.ndarray, p: int, q: int) -> np.ndarray:
    AB, perm = banded_lu_partial_pivot(B, p, q)
    bp = b[perm]
    y = forward_substitution_banded(AB, bp, p, q)
    x = backward_substitution_banded(AB, y, p, q)
    return x


def solve_pi_banded(B: np.ndarray, b: np.ndarray, p: int, q: int) -> np.ndarray:
    z = solve_banded(B, b, p, q)
    return z / z.sum()


if __name__ == "__main__":
    import os
    from i_validasi_data import load_T
    from ii_formulasi import build_B_b
    from iii_struktur_matriks import detect_bandwidth
    from iv_reference_check import reference_pi
    from iv_a_solver_dense import solve_pi_dense

    for f, N in [
        # file yang tidak diberi comment bisa di un-comment, 
        # comment ini hanya untuk membatasi pengecekan ke N kecil agar perbandingannya lebih terlihat 
        # kalau mau cek N besar, tinggal un-comment saja mereka
        ("T_16.csv", 16),
        ("T_32.csv", 32),
        ("T_64.csv", 64),
        ("T_128.csv", 128),
        ("T_256.csv", 256),
        ("T_512.csv", 512),
    ]:
        T = load_T(f)
        B, b = build_B_b(T)
        p, q = detect_bandwidth(B)

        pi_banded = solve_pi_banded(B, b, p, q)
        pi_dense = solve_pi_dense(B, b)
        pi_ref = reference_pi(T) # pi pembanding yang didapat dari numpy

        diff_vs_dense = np.max(np.abs(pi_banded - pi_dense))
        diff_vs_ref = np.max(np.abs(pi_banded - pi_ref))
        r = np.linalg.norm(T.T @ pi_banded - pi_banded)
        e_norm = abs(np.ones(N) @ pi_banded - 1.0)

        print(f"--- N={N}, p={p}, q={q} (solver banded manual) ---")
        print("sum(pi) =", pi_banded.sum())
        print("min(pi) =", pi_banded.min())
        print("max|pi_banded - pi_dense_manual| = ", diff_vs_dense)
        print("max|pi_banded - pi_numpy_ref|    = ", diff_vs_ref)
        print("residual r =", r) 
        print("e_norm =", e_norm)
        print()