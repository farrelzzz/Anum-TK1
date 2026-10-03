"""
i_validasi_data.py  (bagian i -- Validasi Data)
==============================================
Baca matriks transisi T dari CSV dan validasi:
  - dimensi persegi (N x N)
  - semua elemen >= 0
  - setiap baris berjumlah 1
  - indikasi steady state tunggal (irreducible & aperiodic)

load_T() dipakai juga oleh bagian lain (formulasi, struktur matriks, solver, dst) 
sebagai titik masuk baca data
"""

from __future__ import annotations
import numpy as np


def load_T(path: str) -> np.ndarray:
    # Baca matriks transisi T dari file CSV
    T = np.loadtxt(path, delimiter=",")
    if T.ndim != 2:
        raise ValueError(f"{path}: hasil baca bukan matriks 2D, shape={T.shape}")
    return T


def validate_T(T: np.ndarray, tol: float = 1e-9) -> dict:
    """
    Validasi dasar matriks transisi T:
      - dimensi harus persegi (N x N)
      - semua elemen >= 0 (dengan toleransi numerik kecil)
      - setiap baris berjumlah 1
      - indikasi apakah steady state kemungkinan tunggal (irreducible &
        aperiodic, dicek lewat graf ketetanggaan T dan elemen diagonal > 0)
    Mengembalikan dict berisi status & detail supaya bisa dilaporkan.
    """
    n_rows, n_cols = T.shape
    is_square = (n_rows == n_cols)
    N = n_rows

    nonneg_mask = T >= -tol
    is_nonneg = bool(np.all(nonneg_mask))
    min_val = float(T.min())

    row_sums = T.sum(axis=1)
    row_sum_err = np.abs(row_sums - 1.0)
    max_row_sum_err = float(row_sum_err.max())
    rows_ok = bool(np.all(row_sum_err <= tol))

    # Indikasi irreducibility: graf ketetanggaan T (T_ij > tol berarti ada
    # edge i->j) harus berupa satu komponen terhubung kuat (strongly
    # connected). Dicek dengan BFS maju & mundur dari node 0; kalau semua
    # node terjangkau di kedua arah -> indikasi irreducible.
    adj = T > tol

    def reachable(adj_matrix, start):
        seen = np.zeros(N, dtype=bool)
        stack = [start]
        seen[start] = True
        while stack:
            u = stack.pop()
            neighbors = np.nonzero(adj_matrix[u])[0]
            for v in neighbors:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        return seen

    fwd = reachable(adj, 0)
    bwd = reachable(adj.T, 0)
    irreducible_indication = bool(np.all(fwd) and np.all(bwd))

    # Indikasi aperiodic: ada self-loop di setidaknya satu state (T_ii > 0)
    aperiodic_indication = bool(np.any(np.diag(T) > tol))

    return {
        "N": N,
        "is_square": is_square,
        "is_nonneg": is_nonneg,
        "min_value": min_val,
        "rows_sum_to_one": rows_ok,
        "max_row_sum_error": max_row_sum_err,
        "irreducible_indication": irreducible_indication,
        "aperiodic_indication": aperiodic_indication,
        "unique_steady_state_indication": irreducible_indication and aperiodic_indication,
    }


if __name__ == "__main__":
    import glob
    import os

    files = [
        "T_16.csv",
        "T_32.csv",
        "T_64.csv",
        "T_128.csv",
        "T_256.csv",
        "T_512.csv"
    ]
    print(f"{'file':<8}{'N':>5}{'square':>11}{'nonneg':>8}{'rows=1':>8}{'unique_ss':>14}")
    for f in files:
        T = load_T(f)
        val = validate_T(T)
        print(f"{f}{val['N']:>6}{str(val['is_square']):>8}"
              f"{str(val['is_nonneg']):>8}{str(val['rows_sum_to_one']):>8}"
              f"{str(val['unique_steady_state_indication']):>11}")