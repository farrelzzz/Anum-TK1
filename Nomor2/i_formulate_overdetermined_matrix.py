# =====================================================================
# i) FORMULASI MATRIKS OVERDETERMINED
# =====================================================================
"""
Nomor 2 Bagian i - Formulasi Matriks Overdetermined
Diawali dengan membaca matriks 
"""
def hitung_return(prices: np.ndarray) -> np.ndarray:
    """
    Menghitung deret return harian Rt = (Pt - Pt-1) / Pt-1
    Jika panjang prices adalah N, maka panjang returns adalah N-1.
    """
    returns = np.zeros(len(prices) - 1)
    for i in range(1, len(prices)):
        returns[i-1] = (prices[i] - prices[i-1]) / prices[i-1]
    return returns

def bangun_matriks_SETAR(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        
        if Rt_1 >= 0:
            # Rezim 1: Bullish
            A[i-2, 0] = 1.0    
            A[i-2, 1] = Rt_1   
            A[i-2, 2] = Rt_2    
        else:
            # Rezim 2: Bearish
            A[i-2, 3] = 1.0     
            A[i-2, 4] = Rt_1    
            A[i-2, 5] = Rt_2    
            
    return A, b

def hitung_dan_visualisasi_return(filepath: str):
    df = pd.read_csv(filepath)
    prices = df.iloc[:, 1].values 
    returns = hitung_return(prices)
    
    df_returns = pd.DataFrame(returns, columns=['Return Harian'])
    print("--- Statistik Return Harian ---")
    print(df_returns.describe().round(6))
    
    plt.figure(figsize=(12, 5))
    plt.plot(df_returns.index, df_returns['Return Harian'], color='blue', linewidth=1.2, label='Actual Return')
    plt.axhline(0, color='red', linestyle='--', linewidth=1, label='Batas Nol ($R=0$)')
    
    plt.title(f'Return Harian')
    plt.xlabel('Hari ke-t')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Menampilkan plot
    plt.show()

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    file_path = "stock_train.csv"
    prices = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=1)
    
    returns = hitung_return(prices)
    A, b = bangun_matriks_SETAR(returns)
    
    print(f"Jumlah Harga (P): {len(prices)}") #Current P 
    print(f"Jumlah Return (R): {len(returns)}") #Current R
    hitung_dan_visualisasi_return("stock_train.csv") #Laporan Visualisasi
    print(f"Dimensi Matriks A: {A.shape}") #Current M x 6 (lag order, p=2)
    print(f"Dimensi vektor solusi x: (6, 1)")
    print(f"Dimensi vektor target b: {b.shape}")