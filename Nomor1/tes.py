"""
formulate.py
============
Modul inti untuk bagian i-iv (validasi, formulasi, struktur matriks, solver)
yang dikerjakan sendiri. Modul ini juga jadi titik kontrak dengan bagian
v-ix (eksperimen & analisis, dikerjakan teman): fungsi solve_dense() dan
solve_banded() di solver_dense.py / solver_banded.py memakai B, b, p, q
yang dibentuk di sini, sehingga teman tinggal memanggil fungsi solver tanpa
perlu tahu detail formulasinya.

Konvensi yang dipakai (dicatat di ASSUMPTIONS.md, dikonfirmasi ke teman):
- Indexing 0-based secara internal (baris/kolom python index 0..N-1).
  Baris pertama yang disebut di soal (baris ke-1, 1-based) == index 0
  di sini.
- T dibaca sebagai matriks dense (numpy array) langsung dari CSV.
  Membaca CSV bukan termasuk "menyelesaikan SPL / faktorisasi", jadi
  boleh pakai numpy hanya untuk I/O dan operasi aljabar dasar (transpose,
  perkalian, dsb) -- BUKAN untuk numpy.linalg.solve atau scipy.linalg.lu.
- B disimpan sebagai matriks dense NxN (untuk solver dense) ATAU sebagai
  representasi pita (band) tergantung solver yang memakainya. Fungsi
  band_from_dense() ada untuk mengonversi dense -> representasi pita.

Fungsi utama:
    load_T(path)                 -> np.ndarray (N x N)
    validate_T(T, tol)           -> dict hasil validasi (bagian i)
    build_B_b(T)                 -> (B, b) sesuai definisi di soal
    detect_bandwidth(B, tol)     -> (p, q) lower/upper bandwidth
    band_from_dense(B, p, q)     -> representasi pita padat (ab) ala
                                     LAPACK banded-storage, untuk solver
                                     hemat memori
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------
# i) VALIDASI DATA
# ---------------------------------------------------------------------

def load_T(path: str) -> np.ndarray:
    """Baca matriks transisi T dari file CSV. I/O saja, bukan solver."""
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
        aperiodic, dicek secara praktis lewat graf ketetanggaan T dan
        elemen diagonal > 0)
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

    # Indikasi aperiodic (cukup untuk kasus praktis): ada self-loop di
    # setidaknya satu state (T_ii > 0), umum terjadi pada rantai "corridor"
    # seperti pada soal ini.
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


# ---------------------------------------------------------------------
# ii) FORMULASI & PENANGANAN SINGULARITAS
# ---------------------------------------------------------------------

def build_B_b(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Bentuk B dan b sesuai definisi di soal:
        A = I - T^T                (A pi = 0, A singular)
        B[0, :] = [1, 0, ..., 0]   (baris pertama diganti)
        B[i, :] = A[i, :]  for i = 1..N-1
        b = [1, 0, ..., 0]^T

    Solusi z dari B z = b lalu dinormalkan pi = z / (1^T z), karena B z = b
    hanya menjamin komponen pertama z sebanding dengan syarat normalisasi
    yang kita "titipkan" (z_1-like constraint), sedangkan skala z secara
    keseluruhan belum tentu memenuhi 1^T pi = 1. Normalisasi memaksa
    jumlah komponen menjadi tepat 1, sesuai syarat (2) pada soal.
    """
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
    Cek nonsingularity B secara NUMERIK (operasi minor, hanya menghitung
    nilai -- bukan menyelesaikan SPL/faktorisasi utama, jadi boleh pakai
    numpy di sini sesuai batasan tools):
      - rank(B) via numpy.linalg.matrix_rank (SVD-based, hanya untuk cek,
        bukan untuk solve)
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


# ---------------------------------------------------------------------
# iii) IDENTIFIKASI STRUKTUR MATRIKS (bandwidth)
# ---------------------------------------------------------------------

def detect_bandwidth(B: np.ndarray, tol: float = 1e-12) -> tuple[int, int]:
    """
    Deteksi otomatis lower bandwidth p dan upper bandwidth q dari matriks B:
        p = max(i - j) untuk semua (i, j) dengan |B[i,j]| > tol dan i > j
        q = max(j - i) untuk semua (i, j) dengan |B[i,j]| > tol dan j > i
    Dikembalikan (p, q). Jika matriks diagonal murni, p = q = 0.
    """
    N = B.shape[0]
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
    """Estimasi memori matriks dense N x N (byte)."""
    return N * N * itemsize


def band_memory_bytes(N: int, p: int, q: int, itemsize: int = 8) -> int:
    """Estimasi memori representasi pita (p+q+1) x N (byte)."""
    return (p + q + 1) * N * itemsize


def band_preserved_after_row_replace(T: np.ndarray, tol: float = 1e-12) -> bool:
    """
    Verifikasi klaim di poin iii: penggantian baris pertama A (-> B) TIDAK
    merusak struktur banded, karena baris yang diganti ([1,0,...,0]) justru
    punya bandwidth lebih SEMPIT (hanya elemen diagonal) daripada baris
    aslinya. Fungsi ini membandingkan bandwidth A=I-T^T (sebelum ganti
    baris) dengan B (sesudah ganti baris) -- hasilnya harus SAMA atau B
    punya bandwidth <= A, membuktikan band tidak melebar.
    """
    N = T.shape[0]
    A = np.eye(N) - T.T
    B, _ = build_B_b(T)
    p_A, q_A = detect_bandwidth(A, tol)
    p_B, q_B = detect_bandwidth(B, tol)
    return (p_B <= p_A) and (q_B <= q_A)


if __name__ == "__main__":
    # Sanity check cepat untuk semua N yang tersedia.
    import glob
    import os

    files = sorted(
        glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "*_T_*.csv")),
        key=lambda f: int(f.split("_T_")[-1].split(".")[0]),
    )
    header = (f"{'file':<28}{'N':>6}{'valid?':>8}{'p':>4}{'q':>4}"
              f"{'band_ok':>9}{'rank=N':>8}{'cond(B)':>12}{'dense KB':>10}{'band KB':>9}")
    print(header)
    for f in files:
        T = load_T(f)
        val = validate_T(T)
        B, b = build_B_b(T)
        p, q = detect_bandwidth(B)
        band_ok = band_preserved_after_row_replace(T)
        sing = check_nonsingular(B)
        dense_kb = dense_memory_bytes(val["N"]) / 1024
        band_kb = band_memory_bytes(val["N"], p, q) / 1024
        ok = val["is_square"] and val["is_nonneg"] and val["rows_sum_to_one"]
        print(f"{os.path.basename(f):<28}{val['N']:>6}{str(ok):>8}{p:>4}{q:>4}"
              f"{str(band_ok):>9}{str(sing['full_rank']):>8}{sing['cond_2norm']:>12.3e}"
              f"{dense_kb:>10.1f}{band_kb:>9.1f}")