# Nomor 2 bagian vii — Interpretasi finansial dan visualisasi

Metode yang digunakan adalah **QR Householder**, dipilih memakai aturan evaluasi bagian vi. RMSE train = 0.0088321341; RMSE test = 0.01196708.

Model SETAR hasil estimasi:

- **Bullish** (\(R_{t-1}\ge 0\)): \(\widehat{R}_t=0.001513374224+(0.1721528973)R_{t-1}+(0.166072851)R_{t-2}\).
- **Bearish** (\(R_{t-1}<0\)): \(\widehat{R}_t=-0.001212688554+(-0.3603768239)R_{t-1}+(-0.1098080385)R_{t-2}\).

Interpretasi koefisien lag:

| Rezim | Koefisien | Nilai | Interpretasi tanda |
|---|---|---:|---|
| Bullish | \(\phi_{1,1}\) | 0.1721528973 | positif: pengaruh searah dengan return lag (kecenderungan momentum) |
| Bullish | \(\phi_{1,2}\) | 0.166072851 | positif: pengaruh searah dengan return lag (kecenderungan momentum) |
| Bearish | \(\phi_{2,1}\) | -0.3603768239 | negatif: pengaruh berlawanan arah dengan return lag (kecenderungan mean-reverting) |
| Bearish | \(\phi_{2,2}\) | -0.1098080385 | negatif: pengaruh berlawanan arah dengan return lag (kecenderungan mean-reverting) |

Tanda koefisien menyatakan arah pengaruh linear parsial dari return lag, bukan jaminan bahwa tren pasar akan berlanjut atau berbalik. Grafik `overlay_train_vii.png` membandingkan aktual dan estimasi train. `overlay_train_test_vii.png` menggabungkan kedua segmen waktu dengan garis batas yang jelas.
