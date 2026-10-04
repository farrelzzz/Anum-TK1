# =====================================================================
# ii) Investigasi Isu Matriks
# =====================================================================
"""
Nomor 2 Bagian ii - Investigasi Isu Matriks
Memeriksa isu yang mungkin terjadi dalam menjalankan perhitungan matriks pada data stock_train.csv
    - Mencari risiko pivot nol (Isu pembagian oleh nol)
    - Mencari isu keterbatasan presisi floating point yang mendekati nol
"""

import numpy as np
import i_formulate_overdetermined_matrix as SETAR

def trigger_precision_loss(A):
    A_32 = A.astype(np.float32)
    A_64 = A.astype(np.float64)

    AtA_32 = np.dot(A_32.T, A_32)
    AtA_64 = np.dot(A_64.T, A_64)

    # Jika matriks stabil, error_diff seharusnya bernilai 0.
    error_diff = np.max(np.abs(AtA_64 - AtA_32))
    
    return error_diff

if __name__ == "__main__":
    file_path = "Nomor2/stock_train.csv"
    
    prices = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=1)
    returns = SETAR.return_price(prices)
    A, b = SETAR.create_matrix_SETAR(returns)
    max_error = trigger_precision_loss(A)

    print(f"Galat Presisi (Round-off Error): {max_error:.10f}")