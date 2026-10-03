"""
reference_check.py
===================
Di sini menggunakan numpy.linalg.solve sebagai pembanding saja agar
solver manual di bagian iv (dense LU & banded) bisa diverifikasi 
"""

from __future__ import annotations
import numpy as np
from i_validasi_data import load_T, validate_T
from ii_formulasi import build_B_b


def reference_pi(T: np.ndarray) -> np.ndarray:
    B, b = build_B_b(T)
    z = np.linalg.solve(B, b)         
    pi = z / (np.ones(len(z)) @ z)
    return pi


if __name__ == "__main__":
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
        val = validate_T(T)
        pi = reference_pi(T)

        r = np.linalg.norm(T.T @ pi - pi)
        e_norm = abs(np.ones(N) @ pi - 1.0)

        print(f"--- N={N} ---")
        print("valid steady state indication: ", val["unique_steady_state_indication"])
        print("sum(pi) = ", pi.sum())
        print("min(pi) = ", pi.min())
        print("residual r = ", r)
        print("e_norm = ", e_norm)
        print("pi[:5] = ", pi[:5])
        print()