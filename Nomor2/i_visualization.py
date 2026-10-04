def hitung_dan_visualisasi_return(filepath: str):
    df = pd.read_csv(filepath)
    prices = df.iloc[:, 1].values 
    returns = ifom.return_price(prices)
    
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
    import pandas as pd
    import matplotlib.pyplot as plt
    import i_formulate_overdetermined_matrix as ifom
    
    hitung_dan_visualisasi_return("Nomor2/stock_train.csv") #Laporan Visualisasi