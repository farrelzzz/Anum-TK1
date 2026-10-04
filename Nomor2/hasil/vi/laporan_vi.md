# Nomor 2 bagian vi — Evaluasi out-of-sample

Model dilatih hanya dengan data train (\(A\in\mathbb{R}^{300\times 6}\)). Setiap prediksi test adalah one-step ahead: lag awal menggunakan dua return train terakhir, target pertama memasukkan perubahan dari harga train terakhir ke harga test pertama, dan target berikutnya menggunakan return aktual sebelumnya. Dengan demikian 103 harga test menghasilkan 103 return target test.

| Metode | RMSE train | RMSE test | Residual train \(L_2\) |
|---|---:|---:|---:|
| QR Householder | 0.00883213409479 | 0.0119670801055 | 0.152977049914 |
| Persamaan Normal | 0.00883213409479 | 0.0119670801055 | 0.152977049914 |

Metode dengan RMSE test terkecil adalah **QR Householder**. Jika RMSE kedua metode berbeda tidak lebih dari toleransi numerik \(10^{-12}\), QR dipilih karena lebih stabil secara numerik daripada persamaan normal. Pilihan ini dan semua prediksi per tanggal tersimpan dalam berkas CSV bagian vi.
