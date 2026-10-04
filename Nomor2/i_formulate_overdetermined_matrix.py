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
    file_path = "Nomor2/stock_train.csv"
    prices = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=1)
    
    returns = return_price(prices)
    A, b = create_matrix_SETAR(returns)
    
    print(f"Jumlah Harga (P): {len(prices)}") #Current P 
    print(f"Jumlah Return (R): {len(returns)}") #Current R
    print(f"Dimensi Matriks A: {A.shape}") #Current M x 6 (lag order, p=2)
    print(f"Dimensi vektor solusi x: (6, 1)")
    print(f"Dimensi vektor target b: {b.shape}")