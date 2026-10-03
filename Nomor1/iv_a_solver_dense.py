"""
solver_dense.py
================
Solver (iv-a): faktorisasi LU dense dengan partial pivoting, diimplementasi
manual (TANPA numpy.linalg.solve, numpy.linalg.lu, scipy.linalg.lu, atau
fungsi solver/faktorisasi utama lain). numpy di sini hanya dipakai sebagai
array container (indexing, alokasi) -- bukan untuk menyelesaikan SPL.

Notasi mengikuti soal: P B = L U, dengan P matriks permutasi (partial
pivoting berbasis baris).

Pseudocode:
    function lu_decompose_partial_pivot(B):
        U <- copy(B); L <- I; perm <- [0, 1, ..., N-1]
        for k = 0 .. N-2:
            # cari baris pivot: |U[i,k]| terbesar untuk i >= k
            p <- argmax_{i=k..N-1} |U[i,k]|
            if p != k:
                tukar baris k dan p di U
                tukar baris k dan p di L (hanya kolom < k)
                tukar perm[k] dan perm[p]
            for i = k+1 .. N-1:
                L[i,k] <- U[i,k] / U[k,k]
                U[i,k:] <- U[i,k:] - L[i,k] * U[k,k:]
        return L, U, perm

    function solve_dense(B, b):
        L, U, perm <- lu_decompose_partial_pivot(B)
        bp <- b[perm]                      # terapkan permutasi P ke b
        y  <- forward_substitution(L, bp)  # L y = P b
        x  <- backward_substitution(U, y)  # U x = y
        return x
"""

from __future__ import annotations
import numpy as np


def lu_decompose_partial_pivot(B: np.ndarray):
    """
    Faktorisasi P B = L U dengan partial pivoting, manual (Gaussian
    elimination). L unit-lower-triangular, U upper-triangular.

    Return:
        L    : (N,N) lower-triangular, diagonal 1
        U    : (N,N) upper-triangular
        perm : (N,) array permutasi baris, sehingga B[perm, :] == P @ B
    """
    N = B.shape[0]
    U = B.astype(float).copy()
    L = np.eye(N)
    perm = np.arange(N)

    for k in range(N - 1):
        # partial pivoting: cari baris dgn |U[i,k]| terbesar, i >= k 
        pivot_rel = np.argmax(np.abs(U[k:, k]))
        pivot_row = k + pivot_rel

        if pivot_row != k:
            U[[k, pivot_row], :] = U[[pivot_row, k], :]
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
            perm[[k, pivot_row]] = perm[[pivot_row, k]]

        if U[k, k] == 0.0:
            raise ZeroDivisionError(
                f"Pivot nol di kolom {k} walau sudah partial pivoting -- "
                f"matriks B singular secara numerik."
            )

        # --- eliminasi ---
        for i in range(k + 1, N):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]

    return L, U, perm


def forward_substitution(L: np.ndarray, bp: np.ndarray) -> np.ndarray:
    # Selesaikan L y = bp, L unit-lower-triangular, manual (tanpa library
    N = L.shape[0]
    y = np.zeros(N)
    for i in range(N):
        s = bp[i] - L[i, :i] @ y[:i]
        y[i] = s / L[i, i]
    return y


def backward_substitution(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Selesaikan U x = y, U upper-triangular, manual (tanpa library
    N = U.shape[0]
    x = np.zeros(N)
    for i in range(N - 1, -1, -1):
        s = y[i] - U[i, i + 1:] @ x[i + 1:]
        x[i] = s / U[i, i]
    return x


def solve_dense(B: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Interface utama solver dense (dipanggil dari eksperimen bagian v, dan
    dari laporan bagian iv untuk uji skala kecil).
    """
    L, U, perm = lu_decompose_partial_pivot(B)
    bp = b[perm]
    y = forward_substitution(L, bp)
    x = backward_substitution(U, y)
    return x


def solve_pi_dense(B: np.ndarray, b: np.ndarray) -> np.ndarray:
    # z = solve_dense(B,b), lalu normalisasi pi = z / (1^T z)
    z = solve_dense(B, b)
    pi = z / z.sum()
    return pi


if __name__ == "__main__":
    from i_validasi_data import load_T
    from ii_formulasi import build_B_b
    from iv_reference_check import reference_pi

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

        pi_manual = solve_pi_dense(B, b)
        pi_ref = reference_pi(T)  # pi pembanding yang didapat dari numpy

        max_abs_diff = np.max(np.abs(pi_manual - pi_ref))
        r = np.linalg.norm(T.T @ pi_manual - pi_manual)
        e_norm = abs(np.ones(N) @ pi_manual - 1.0)

        print(f"--- N={N} (solver dense PB=LU manual) ---")
        print("sum(pi) =", pi_manual.sum())
        print("min(pi) = ", pi_manual.min())
        print("max|pi_manual - pi_numpy_ref| =", max_abs_diff)
        print("residual r =", r )
        print("e_norm =", e_norm)
        print()