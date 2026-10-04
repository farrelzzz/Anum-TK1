# Nomor 2 bagian v — Perbandingan performa dan pengaruh outlier

Matriks train berukuran **300 x 6**. Penyimpanan masukan bersama \(A,b\) adalah 16,800 byte pada float64 dan tidak dihitung sebagai workspace tambahan.

| Metode | FLOPs perkiraan | Workspace tanpa \(Q\) eksplisit (byte) | Workspace dengan \(Q\) eksplisit (byte) | Residual train \(L_2\) |
|---|---:|---:|---:|---:|
| QR Householder | 25056 | 19,248 | 739,248 | 0.152977049914 |
| Persamaan Normal | 21744 | 720 | 720 | 0.152977049914 |

Perkiraan QR menggunakan \(2mn^2-\frac{2}{3}n^3\) untuk faktorisasi dan sekitar \(2mn\) untuk mentransformasikan ruas kanan. Persamaan Normal menggunakan sekitar \(2mn^2\) untuk membentuk \(A^TA\), lalu \(\frac{2}{3}n^3\) untuk eliminasi pada sistem kecil. Estimasi workspace memperhitungkan array kerja solver (tidak termasuk masukan bersama A dan b dan temporary internal NumPy); penyimpanan matriks ortogonal penuh menambah \(m^2\) elemen (720,000 byte). Implementasi solver QR di bagian iv menerapkan refleksi secara implisit dan hanya membentuk \(H_1\) eksplisit untuk verifikasi.

Condition number yang dihitung oleh prosedur bagian iii: \(\kappa_2(A)=192.60541\) dan \(\kappa_2(A^TA)=37096.844\). Selisih residual kedua metode adalah 0.000000e+00. QR secara umum lebih stabil karena tidak membentuk \(A^TA\), yang menguadratkan condition number.

Outlier ekstrem pada return berpengaruh kuadrat terhadap fungsi objektif least squares. Satu observasi ekstrem dapat menaikkan residual dan menggeser koefisien agar lebih mengikuti observasi tersebut. Karena itu residual besar dapat mencerminkan shock pada data, bukan hanya kekurangan solver; pemeriksaan return ekstrem dan evaluasi out-of-sample tetap diperlukan.
