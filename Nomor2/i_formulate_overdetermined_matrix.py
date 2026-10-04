# =====================================================================
# i) FORMULASI MATRIKS OVERDETERMINED
# =====================================================================
"""
Nomor 2 Bagian i - Formulasi Matriks Overdetermined
Diawali dengan membaca file stock_train.csv berukuran 303x1
Yang diproses kembali menggunakan fungsi:
    - return_price -> Mencari hasil return dari data harga penutupan
    - bangun_matriks_SETAR -> Untuk membentuk matriks return_price sebelumnya menjadi
                              menjadi matriks yang sudah disesuaikan dengan konsep SETAR
"""

import numpy as np
from pathlib import Path
import csv

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "hasil" / "i"

def return_price(prices: np.ndarray) -> np.ndarray:
    """
    Menghitung deret return harian Rt = (Pt - Pt-1) / Pt-1
    Jika panjang prices adalah N, maka panjang returns adalah N-1.
    """
    returns = np.zeros(len(prices) - 1)
    for i in range(1, len(prices)):
        returns[i-1] = (prices[i] - prices[i-1]) / prices[i-1]
    return returns

def create_matrix_SETAR(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Menyusun sistem persamaan linear overdetermined A*x = b. Menggunakan lag order p = 2
    """
    N = len(returns)
    n_baris = N - 2  #Caused by Lag Order
    
    A = np.zeros((n_baris, 6))
    b = np.zeros(n_baris)
    
    for i in range(2, N):
        Rt = returns[i]
        Rt_1 = returns[i-1]
        Rt_2 = returns[i-2]
        
        b[i-2] = Rt
        
        if Rt_1 >= 0: # Rezim 1: Bullish
            A[i-2, 0] = 1.0    
            A[i-2, 1] = Rt_1   
            A[i-2, 2] = Rt_2    
        else: # Rezim 2: Bearish
            A[i-2, 3] = 1.0     
            A[i-2, 4] = Rt_1    
            A[i-2, 5] = Rt_2    
            
    return A, b

if __name__ == "__main__":
    file_path = HERE / "stock_train.csv"
    prices = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=1)
    
    returns = return_price(prices)
    A, b = create_matrix_SETAR(returns)
    
    #Output tester <Range/Size>
    print(f"Jumlah Harga (P): {len(prices)}") #Current P 
    print(f"Jumlah Return (R): {len(returns)}") #Current R
    print(f"Dimensi Matriks A: {A.shape}") #Current M x 6 (lag order, p=2)
    print(f"Dimensi vektor solusi x: (6, 1)")
    print(f"Dimensi vektor target b: {b.shape}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUTPUT / "return_train_i.csv", returns, delimiter=",", header="Return", comments="")
    np.savetxt(OUTPUT / "matriks_A_i.csv", A, delimiter=",", header=",".join(f"x{j + 1}" for j in range(A.shape[1])), comments="")
    np.savetxt(OUTPUT / "vektor_b_i.csv", b, delimiter=",", header="Return_target", comments="")
    bullish = int(np.count_nonzero(A[:, 0]))
    bearish = int(np.count_nonzero(A[:, 3]))
    report = f"""# Nomor 2 bagian i — Formulasi matriks overdetermined

Harga Close train berjumlah {len(prices)}. Return harian dihitung sebagai \\(R_t=(P_t-P_{{t-1}})/P_{{t-1}}\\), sehingga terdapat {len(returns)} return. Untuk lag order 2, baris matriks dimulai pada indeks return ke-3. Rezim ditentukan dari tanda return lag-1: \\(R_{{t-1}}\\ge0\\) bullish dan \\(R_{{t-1}}<0\\) bearish.

Baris desain menggunakan urutan parameter \\(x=[\\alpha_1,\\phi_{{1,1}},\\phi_{{1,2}},\\alpha_2,\\phi_{{2,1}},\\phi_{{2,2}}]^T\\). Untuk rezim bullish barisnya \\([1,R_{{t-1}},R_{{t-2}},0,0,0]\\); untuk bearish \\([0,0,0,1,R_{{t-1}},R_{{t-2}}]\\). Target adalah \\(b=R_t\\).

Dimensi sistem: \\(A\\in\\mathbb{{R}}^{{{A.shape[0]}\\times{A.shape[1]}}}\\), \\(x\\in\\mathbb{{R}}^6\\), dan \\(b\\in\\mathbb{{R}}^{{{b.shape[0]}}}\\). Baris bullish: {bullish}; baris bearish: {bearish}. Sistem overdetermined karena {A.shape[0]} > 6.

Data yang digunakan ulang tersedia di `return_train_i.csv`, `matriks_A_i.csv`, dan `vektor_b_i.csv`.
"""
    (OUTPUT / "laporan_i.md").write_text(report, encoding="utf-8")
