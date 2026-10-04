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
def solve_spl(M_input: np.ndarray, y_input: np.ndarray) -> np.ndarray:
    """
    Fungsi Solver Utama: Menyelesaikan M*x = y menggunakan 
    Eliminasi Gauss dan Partial Pivoting.
    """
    M = M_input.copy()
    y = y_input.copy()
    n = M.shape[0]
    
    # 1. Partial Pivoting & Gauss Elimination
    for i in range(n):
        max_row = i + np.argmax(np.abs(M[i:n, i]))
        
        if max_row != i:
            M[[i, max_row]] = M[[max_row, i]]
            y[[i, max_row]] = y[[max_row, i]]
            
        for j in range(i + 1, n):
            if M[i, i] == 0:
                continue # division by zero!
            factor = M[j, i] / M[i, i]
            M[j, i:n] = M[j, i:n] - factor * M[i, i:n]
            y[j] = y[j] - factor * y[i]
            
    # 2. Back Substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
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
    cond_AtA = lambda_max / lambda_min
    cond_A = np.sqrt(cond_AtA)
    
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
    
    print("\n--- Analisis Stabilitas Numerik (Manual) ---")
    print(f"Condition Number Matriks A     : {cond_A:e}")
    print(f"Condition Number Matriks A^T*A : {cond_AtA:e}")
    print(f"Rasio Peningkatan Instabilitas : {cond_AtA / cond_A:.2f}x")