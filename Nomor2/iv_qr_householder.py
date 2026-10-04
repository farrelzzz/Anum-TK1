"""Nomor 2 bagian iv: least squares SETAR dengan QR Householder buatan sendiri."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from i_formulate_overdetermined_matrix import create_matrix_SETAR, return_price


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "hasil" / "iv"
PARAMETERS = ("alpha1", "phi11", "phi12", "alpha2", "phi21", "phi22")


def back_substitute(U: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Selesaikan sistem segitiga atas dengan substitusi balik."""
    n = len(rhs)
    x = np.zeros(n, dtype=float)
    scale = max(1.0, float(np.max(np.abs(U))))
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) <= np.finfo(float).eps * scale:
            raise np.linalg.LinAlgError(
                f"Matriks desain tidak berpangkat penuh; pivot U[{i},{i}] terlalu kecil"
            )
        x[i] = (rhs[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]
    return x


def householder_lstsq(A: np.ndarray, b: np.ndarray):
    """Faktorkan A=QR secara implisit, transformasikan b, lalu selesaikan R x=Q^T b."""
    R = A.astype(float).copy()
    Qt_b = b.astype(float).copy()
    m, n = R.shape
    first_reflector = None

    for k in range(n):
        x = R[k:, k].copy()
        norm_x = float(np.sqrt(np.dot(x, x)))
        if norm_x == 0.0:
            raise np.linalg.LinAlgError(f"Kolom {k + 1} tidak dapat direfleksikan")
        alpha = -np.copysign(norm_x, x[0] if x[0] != 0.0 else 1.0)
        v = x
        v[0] -= alpha
        v_norm_sq = float(np.dot(v, v))
        if v_norm_sq == 0.0:
            raise np.linalg.LinAlgError(f"Reflektor Householder kolom {k + 1} degenerat")
        beta = 2.0 / v_norm_sq

        # R[k:, k:] <- H_k R[k:, k:] dan Qt_b[k:] <- H_k Qt_b[k:]
        R[k:, k:] -= beta * np.outer(v, v @ R[k:, k:])
        Qt_b[k:] -= beta * v * np.dot(v, Qt_b[k:])
        if k == 0:
            first_reflector = (v.copy(), beta, alpha)

    x_ls = back_substitute(R[:n, :n], Qt_b[:n])
    return x_ls, R, first_reflector


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    train_path = HERE / "stock_train.csv"
    prices = np.genfromtxt(train_path, delimiter=",", skip_header=1, usecols=1)
    if prices.ndim != 1 or len(prices) < 4 or not np.all(np.isfinite(prices)):
        raise ValueError("stock_train.csv harus memuat setidaknya empat harga Close yang valid")

    returns = return_price(prices)
    A, b = create_matrix_SETAR(returns)
    m, n = A.shape
    if m < n:
        raise ValueError(f"Sistem harus overdetermined: ditemukan A berukuran {A.shape}")

    x_ls, R, reflector = householder_lstsq(A, b)
    v, beta, alpha = reflector
    # Representasi H1 = I - beta*v_full*v_full^T; H1A dihitung implisit.
    v_full = np.zeros(m, dtype=float)
    v_full[:] = v
    H1 = np.eye(m) - beta * np.outer(v_full, v_full)
    H1A = A - beta * np.outer(v_full, v_full @ A)
    subdiagonal = H1A[1:, 0]
    subdiag_norm = float(np.linalg.norm(subdiagonal))
    residual_norm = float(np.linalg.norm(A @ x_ls - b))
    factorization_error = float(np.linalg.norm(np.tril(R, -1)))
    condition_number = float(np.linalg.cond(A, 2))
    column_norms = np.sqrt(np.sum(A * A, axis=0))
    tolerance = 1e-10 * max(1.0, float(np.linalg.norm(A[:, 0])))
    if subdiag_norm > tolerance:
        raise ArithmeticError(
            f"H1 tidak mengeliminasi subdiagonal: norma {subdiag_norm:.3e} > {tolerance:.3e}"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT / "x_ls_iv.csv",
        ["Parameter", "Nilai"],
        zip(PARAMETERS, x_ls),
    )
    write_csv(
        OUTPUT / "H1_iv.csv",
        [f"Kolom_{j + 1}" for j in range(m)],
        H1,
    )
    write_csv(
        OUTPUT / "H1A_iv.csv",
        [f"Kolom_{j + 1}" for j in range(n)],
        H1A,
    )

    # Tampilkan blok awal H1A yang cukup ringkas untuk ditinjau di laporan.
    preview_rows = "\n".join(
        "| " + " | ".join(f"{value:.8e}" for value in row) + " |"
        for row in H1A[:8]
    )
    parameter_rows = "\n".join(
        f"| {name} | {value:.12g} |" for name, value in zip(PARAMETERS, x_ls)
    )
    report = f"""# Nomor 2 bagian iv — Penyelesaian LSP dengan QR Householder

## Metode dan alasan pemilihan

Kelompok C9 bernomor ganjil, sehingga metode QR yang ditugaskan adalah Householder Reflections. Matriks desain SETAR berukuran \\(m\\times 6\\), dengan \\(m={m}\\) dan enam koefisien. Investigasi bagian ii menunjukkan rank penuh 6, condition number 2-norm \\(\\kappa_2(A)={condition_number:.8g}\\), dan norma kolom berkisar {column_norms.min():.8g} hingga {column_norms.max():.8g}. Matriksnya tall dan full rank, tetapi kolom-kolom prediktor memiliki skala berbeda. Refleksi Householder menghilangkan elemen subdiagonal secara ortogonal, mempertahankan norma 2, dan menghindari pembentukan \\(A^TA\\), yang akan menaikkan condition number kira-kira menjadi kuadratnya. Ini memberi alasan numerik dan struktur-spesifik untuk memakai QR, sementara solusi tetap diperoleh dari sistem segitiga atas.

## Langkah algoritma

1. Salin \\(A\\) ke \\(R\\), dan salin \\(b\\) ke \\(y\\).
2. Untuk setiap kolom \\(k\\), bentuk vektor refleksi \\(v_k\\) dan \\(H_k=I-2v_kv_k^T/(v_k^Tv_k)\\); terapkan refleksi pada bagian aktif \\(R\\) dan \\(y\\).
3. Setelah \\(R\\) menjadi segitiga atas, selesaikan \\(R_{{1:6,1:6}}x_{{LS}}=y_{{1:6}}\\) dengan substitusi balik.

Solver tidak menggunakan fungsi pustaka untuk faktorisasi atau penyelesaian SPL. Implementasi menerapkan refleksi melalui perkalian vektor dan outer product.

## Refleksi pertama dan verifikasi

Untuk kolom pertama, \\(x=A_{{:,1}}\\), \\(\\alpha=-\\mathrm{{sign}}(x_1)\\|x\\|_2={alpha:.12g}\\). Dengan \\(v=x-\\alpha e_1\\), refleksi pertama adalah \\(H_1=I-\\beta vv^T\\), dengan \\(\\beta=2/(v^Tv)={beta:.12g}\\). Matriks penuh berukuran \\({m}\\times{m}\\) tersedia di `H1_iv.csv`; perkalian matriksnya tersedia di `H1A_iv.csv`.

Perkalian \\(H_1A\\) menghasilkan struktur berikut pada delapan baris pertama (seluruh matriks tersedia di `H1A_iv.csv`):

| Kolom 1 | Kolom 2 | Kolom 3 | Kolom 4 | Kolom 5 | Kolom 6 |
|---:|---:|---:|---:|---:|---:|
{preview_rows}

Norma seluruh elemen subdiagonal kolom pertama, \\(\\|(H_1A)_{{2:m,1}}\\|_2\\), adalah **{subdiag_norm:.6e}** (toleransi verifikasi {tolerance:.2e}). Elemen pertama kolom itu menjadi \\(\\alpha\\), sedangkan elemen di bawahnya nol hingga galat pembulatan.

## Solusi

| Parameter | \\(x_{{LS}}\\) |
|---|---:|
{parameter_rows}

Ukuran sistem: \\(A\\in\\mathbb{{R}}^{{{m}\\times 6}}\\), \\(b\\in\\mathbb{{R}}^{{{m}}}\\), \\(x_{{LS}}\\in\\mathbb{{R}}^6\\). Norma residual \\(\\|Ax_{{LS}}-b\\|_2\\) = **{residual_norm:.8g}**. Norma bagian bawah-diagonal \\(R\\) setelah QR = {factorization_error:.6e}.
"""
    (OUTPUT / "laporan_iv.md").write_text(report, encoding="utf-8")

    print(f"A: {A.shape}, b: {b.shape}, x_LS: {x_ls.shape}")
    print(f"|| (H1 A)[2:m,1] ||_2 = {subdiag_norm:.6e} (toleransi {tolerance:.2e})")
    print(f"|| A x_LS - b ||_2 = {residual_norm:.8g}")
    print("Koefisien x_LS:")
    for name, value in zip(PARAMETERS, x_ls):
        print(f"  {name} = {value:.12g}")
    print(f"Laporan: {OUTPUT / 'laporan_iv.md'}")


if __name__ == "__main__":
    main()
