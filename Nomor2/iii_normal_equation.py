# =====================================================================
# iii) Penyelesaian via Persamaan Normal
# =====================================================================
"""
Nomor 2 Bagian iii - Penyelesaian Persamaan Normal
- Menyelesaikan least squares A*x = b melalui persamaan normal (A^T * A * x = A^T * b)
- Menggunakan metode Eliminasi Gauss dengan Partial Pivoting
- Menghitung condition number matriks A dan A^T * A
"""

import numpy as np

def solve_normal_equations(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Menyelesaikan A*x = b menggunakan Persamaan Normal A^T*A*x = A^T*b
    Dilengkapi dengan Eliminasi Gauss dan Partial Pivoting.
    """
    # 1. Bentuk sistem persamaan normal (M = A^T * A, y = A^T * b)
    M = np.dot(A.T, A)  
    y = np.dot(A.T, b)  
    
    n = M.shape[0]
    
    # 2. Partial Pivoting & Gauss Elimination
    for i in range(n):
        max_row = i + np.argmax(np.abs(M[i:n, i]))
        
        if max_row != i:
            M[[i, max_row]] = M[[max_row, i]]
            y[[i, max_row]] = y[[max_row, i]]
            
        for j in range(i + 1, n):
            if M[i, i] == 0:
                continue #division by zero!
            factor = M[j, i] / M[i, i]
            M[j, i:n] = M[j, i:n] - factor * M[i, i:n]
            y[j] = y[j] - factor * y[i]
            
    # 3. Back Substitution
    x_LS = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_LS[i] = (y[i] - np.dot(M[i, i+1:], x_LS[i+1:])) / M[i, i]
        
    return x_LS

def calculate_condition_numbers(A: np.ndarray) -> tuple[float, float]:
    """
    Menghitung dan membandingkan condition number (L2 norm) dari A dan A^T * A.
    """
    M = np.dot(A.T, A)
    
    cond_A = np.linalg.cond(A, p=2)
    cond_AtA = np.linalg.cond(M, p=2)
    
    return cond_A, cond_AtA

if __name__ == "__main__":
    import i_formulate_overdetermined_matrix as SETAR
    
    file_path = "Nomor2/stock_train.csv"
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
    
    print("\n--- Analisis Stabilitas Numerik ---")
    print(f"Condition Number Matriks A : {cond_A:e}")
    print(f"Condition Number Matriks A^T*A : {cond_AtA:e}")
    print(f"Rasio Peningkatan Instabilitas : {cond_AtA / cond_A:.2f}x")