# =====================================================================
# iii) Penyelesaian via Persamaan Normal
# =====================================================================
"""
Nomor 2 Bagian iii - Penyelesaian Persamaan Normal
- Menyelesaikan least squares A*x = b melalui persamaan normal (A^T * A * x = A^T * b)
- Menggunakan metode Eliminasi Gauss dengan Partial Pivoting 
- Menghitung condition number menggunakan power iteration berbasis fungsi SPL
"""

import numpy as np
from pathlib import Path
import csv

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "hasil" / "iii"
PARAMETERS = ("alpha1", "phi11", "phi12", "alpha2", "phi21", "phi22")
def solve_spl(M_input: np.ndarray, y_input: np.ndarray) -> np.ndarray:
    """
    Fungsi Solver Utama: Menyelesaikan M*x = y menggunakan 
    Eliminasi Gauss dan Partial Pivoting.
    """
    M = np.asarray(M_input, dtype=float).copy()
    y = np.asarray(y_input, dtype=float).copy()
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M harus berupa matriks persegi")
    if y.ndim != 1 or y.shape[0] != M.shape[0]:
        raise ValueError("Panjang y harus sama dengan dimensi M")
    if not np.all(np.isfinite(M)) or not np.all(np.isfinite(y)):
        raise ValueError("M dan y harus berisi nilai finite")
    n = M.shape[0]
    scale = float(np.max(np.abs(M)))
    if scale == 0.0:
        raise np.linalg.LinAlgError("Matriks koefisien adalah matriks nol")
    pivot_tol = np.finfo(float).eps * max(1, n) * scale
    
    # 1. Partial Pivoting & Gauss Elimination
    for i in range(n):
        max_row = i + np.argmax(np.abs(M[i:n, i]))
        
        if max_row != i:
            M[[i, max_row]] = M[[max_row, i]]
            y[[i, max_row]] = y[[max_row, i]]
            
        if abs(M[i, i]) <= pivot_tol:
            raise np.linalg.LinAlgError(
                f"Pivot terlalu kecil pada langkah {i + 1}: {M[i, i]:.3e}"
            )
        for j in range(i + 1, n):
            factor = M[j, i] / M[i, i]
            M[j, i:n] = M[j, i:n] - factor * M[i, i:n]
            y[j] = y[j] - factor * y[i]
            
    # 2. Back Substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(M[i, i]) <= pivot_tol:
            raise np.linalg.LinAlgError(
                f"Pivot substitusi balik terlalu kecil pada baris {i + 1}: {M[i, i]:.3e}"
            )
        sum_val = 0.0
        for j in range(i + 1, n):
            sum_val += M[i, j] * x[j]
        x[i] = (y[i] - sum_val) / M[i, i]
        
    return x

def solve_normal_equations(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Menyelesaikan least squares dengan memanggil fungsi solver SPL."""
    M = np.dot(A.T, A)  
    y = np.dot(A.T, b)  
    return solve_spl(M, y)

def calculate_condition_numbers(A: np.ndarray, iterasi=100) -> tuple[float, float]:
    """
    Menghitung kappa_2(AtA) = lambda_max / lambda_min.
    """
    M = np.dot(A.T, A)
    n = M.shape[0]
    
    v = np.ones(n)
    for _ in range(iterasi):
        v_new = np.dot(M, v)
        v = v_new / np.sqrt(np.sum(v_new**2)) 
    lambda_max = np.dot(v.T, np.dot(M, v)) / np.dot(v.T, v)
    
    u = np.ones(n)
    for _ in range(iterasi):
        u_new = solve_spl(M, u) 
        u = u_new / np.sqrt(np.sum(u_new**2))
    lambda_min = np.dot(u.T, np.dot(M, u)) / np.dot(u.T, u)
    
    # Condition Number
    if lambda_min <= 0.0 or not np.isfinite(lambda_min):
        raise np.linalg.LinAlgError("Estimasi eigenvalue minimum A^T A tidak positif/finite")
    cond_AtA = lambda_max / lambda_min
    cond_A = np.sqrt(cond_AtA)
    
    return cond_A, cond_AtA

if __name__ == "__main__":
    import i_formulate_overdetermined_matrix as SETAR
    
    file_path = HERE / "stock_train.csv"
    prices = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=1)
    
    returns = SETAR.return_price(prices)
    A, b = SETAR.create_matrix_SETAR(returns)
    x_LS = solve_normal_equations(A, b)
    cond_A, cond_AtA = calculate_condition_numbers(A)
    
    print("x_LS:")
    print(f"Alpha_1 (Bullish)   : {x_LS[0]:.6f}")
    print(f"Phi_1,1             : {x_LS[1]:.6f}")
    print(f"Phi_1,2             : {x_LS[2]:.6f}")
    print(f"Alpha_2 (Bearish)   : {x_LS[3]:.6f}")
    print(f"Phi_2,1             : {x_LS[4]:.6f}")
    print(f"Phi_2,2             : {x_LS[5]:.6f}")
    
    print("\n--- Analisis Stabilitas Numerik (Manual) ---")
    print(f"Condition Number Matriks A     : {cond_A:e}")
    print(f"Condition Number Matriks A^T*A : {cond_AtA:e}")
    print(f"Rasio Peningkatan Instabilitas : {cond_AtA / cond_A:.2f}x")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "koefisien_normal_iii.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Parameter", "Nilai"])
        writer.writerows(zip(PARAMETERS, x_LS))
    residual = float(np.linalg.norm(A @ x_LS - b))
    report = f"""# Nomor 2 bagian iii — Penyelesaian via persamaan normal

Least squares \\(Ax\\approx b\\) diselesaikan dengan \\(A^TAx=A^Tb\\). Implementasi membentuk kedua ruas dan menyelesaikan sistem 6x6 menggunakan eliminasi Gauss dengan partial pivoting, lalu substitusi balik. Algoritma utama tidak memanggil solver pustaka.

Pseudocode: bentuk \\(M=A^TA\\), \\(y=A^Tb\\); untuk tiap kolom pilih baris dengan nilai pivot absolut terbesar, tukar baris, eliminasi elemen di bawah pivot; periksa pivot relatif terhadap skala matriks; lakukan substitusi balik.

| Parameter | Nilai |
|---|---:|
{chr(10).join(f"| {name} | {value:.12g} |" for name, value in zip(PARAMETERS, x_LS))}

Condition number 2-norm hasil power iteration: \\(\\kappa_2(A)={cond_A:.10g}\\), \\(\\kappa_2(A^TA)={cond_AtA:.10g}\\). Karena \\(\\kappa_2(A^TA)\\approx\\kappa_2(A)^2\\), pembentukan persamaan normal memperburuk sensitivitas terhadap pembulatan. Untuk dataset ini rank(A) = {np.linalg.matrix_rank(A)} dari 6 dan residual \\(\\|Ax-b\\|_2={residual:.10g}\\). Solusi normal tersedia di `koefisien_normal_iii.csv`.
"""
    (OUTPUT / "laporan_iii.md").write_text(report, encoding="utf-8")
    print(f"Residual ||Ax-b||_2: {residual:.10g}")
    print(f"Laporan tersimpan di {OUTPUT / 'laporan_iii.md'}")
