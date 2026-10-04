# Nomor 2 bagian iii — Penyelesaian via persamaan normal

Least squares \(Ax\approx b\) diselesaikan dengan \(A^TAx=A^Tb\). Implementasi membentuk kedua ruas dan menyelesaikan sistem 6x6 menggunakan eliminasi Gauss dengan partial pivoting, lalu substitusi balik. Algoritma utama tidak memanggil solver pustaka.

Pseudocode: bentuk \(M=A^TA\), \(y=A^Tb\); untuk tiap kolom pilih baris dengan nilai pivot absolut terbesar, tukar baris, eliminasi elemen di bawah pivot; periksa pivot relatif terhadap skala matriks; lakukan substitusi balik.

| Parameter | Nilai |
|---|---:|
| alpha1 | 0.00151337422439 |
| phi11 | 0.172152897275 |
| phi12 | 0.166072850953 |
| alpha2 | -0.00121268855387 |
| phi21 | -0.360376823937 |
| phi22 | -0.109808038547 |

Condition number 2-norm hasil power iteration: \(\kappa_2(A)=192.6054102\), \(\kappa_2(A^TA)=37096.84403\). Karena \(\kappa_2(A^TA)\approx\kappa_2(A)^2\), pembentukan persamaan normal memperburuk sensitivitas terhadap pembulatan. Untuk dataset ini rank(A) = 6 dari 6 dan residual \(\|Ax-b\|_2=0.1529770499\). Solusi normal tersedia di `koefisien_normal_iii.csv`.
