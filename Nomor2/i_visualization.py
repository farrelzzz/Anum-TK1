"""Visualisasi return train untuk mendukung laporan Nomor 2 bagian i."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from i_formulate_overdetermined_matrix import return_price


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "hasil" / "i"


def main() -> None:
    prices = np.genfromtxt(HERE / "stock_train.csv", delimiter=",", skip_header=1, usecols=1)
    returns = return_price(prices)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(np.arange(1, len(returns) + 1), returns, color="tab:blue", linewidth=1.1,
            label="Return aktual")
    ax.axhline(0.0, color="tab:red", linestyle="--", linewidth=1, label="Ambang rezim")
    ax.set_title("Return harian data train")
    ax.set_xlabel("Urutan return")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "return_train_i.png", dpi=160)
    plt.close(fig)
    print(f"Grafik return tersimpan di {OUTPUT / 'return_train_i.png'}")


if __name__ == "__main__":
    main()
