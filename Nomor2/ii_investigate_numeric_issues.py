import numpy as np

def investigate_numeric_issues(A):
    """
    Menghitung tingkat sparsity (persentase elemen nol dalam matriks)
    """
    zeros = np.count_nonzero(A == 0)
    zero_percentage = (zeros / A.size) * 100
    
    """
    Mengecek ketimpangan skala antar kolom (intersep vs return lag)
    Kolom 0 dan 3 adalah intersep, sisanya adalah lag return
    """
    drift_col = A[:, [0, 3]]
    lag_col = A[:, [1, 2, 4, 5]]
    
    max_drift_scale = np.max(np.abs(drift_col))
    max_lag_scale = np.max(np.abs(lag_col))
    
    scale_ratio = max_drift_scale / max_lag_scale if max_lag_scale != 0 else 0
    
    return zero_percentage, scale_ratio

if __name__ == "__main__":
    import i_formulate_overdetermined_matrix as SETAR
            
    file_path = "Nomor2/stock_train.csv"
    prices = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=1)
    
    # Memanggil fungsi matriks model SETAR
    returns = SETAR.return_price(prices)
    A, b = SETAR.create_matrix_SETAR(returns)
            
    # Menjalankan fungsi investigasi isu numerik
    zero_percentage, scale_ratio = investigate_numeric_issues(A)
    
    print("--- ii. Investigasi Isu Numerik ---")
    print(f"Tingkat Sparsity (Elemen Nol) : {zero_percentage:.2f}%")
    print(f"Rasio Skala (Intersep vs Lag) : {scale_ratio:.2f} kali lipat")
    print("-" * 35)