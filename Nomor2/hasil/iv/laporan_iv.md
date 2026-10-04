# Nomor 2 bagian iv — Penyelesaian LSP dengan QR Householder

## Metode dan alasan pemilihan

Kelompok C9 bernomor ganjil, sehingga metode QR yang ditugaskan adalah Householder Reflections. Matriks desain SETAR berukuran \(m\times 6\), dengan \(m=300\) dan enam koefisien. Investigasi bagian ii menunjukkan rank penuh 6, condition number 2-norm \(\kappa_2(A)=192.60541\), dan norma kolom berkisar 0.08897615 hingga 13.96424. Matriksnya tall dan full rank, tetapi kolom-kolom prediktor memiliki skala berbeda. Refleksi Householder menghilangkan elemen subdiagonal secara ortogonal, mempertahankan norma 2, dan menghindari pembentukan \(A^TA\), yang akan menaikkan condition number kira-kira menjadi kuadratnya. Ini memberi alasan numerik dan struktur-spesifik untuk memakai QR, sementara solusi tetap diperoleh dari sistem segitiga atas.

## Langkah algoritma

1. Salin \(A\) ke \(R\), dan salin \(b\) ke \(y\).
2. Untuk setiap kolom \(k\), bentuk vektor refleksi \(v_k\) dan \(H_k=I-2v_kv_k^T/(v_k^Tv_k)\); terapkan refleksi pada bagian aktif \(R\) dan \(y\).
3. Setelah \(R\) menjadi segitiga atas, selesaikan \(R_{1:6,1:6}x_{LS}=y_{1:6}\) dengan substitusi balik.

Solver tidak menggunakan fungsi pustaka untuk faktorisasi atau penyelesaian SPL. Implementasi menerapkan refleksi melalui perkalian vektor dan outer product.

## Refleksi pertama dan verifikasi

Untuk kolom pertama, \(x=A_{:,1}\), \(\alpha=-\mathrm{sign}(x_1)\|x\|_2=-13.9642400438\). Dengan \(v=x-\alpha e_1\), refleksi pertama adalah \(H_1=I-\beta vv^T\), dengan \(\beta=2/(v^Tv)=0.00512820512821\). Matriks penuh berukuran \(300\times300\) tersedia di `H1_iv.csv`; perkalian matriksnya tersedia di `H1A_iv.csv`.

Perkalian \(H_1A\) menghasilkan struktur berikut pada delapan baris pertama (seluruh matriks tersedia di `H1A_iv.csv`):

| Kolom 1 | Kolom 2 | Kolom 3 | Kolom 4 | Kolom 5 | Kolom 6 |
|---:|---:|---:|---:|---:|---:|
| -1.39642400e+01 | -1.03522935e-01 | -4.47421625e-02 | 0.00000000e+00 | -1.73472348e-18 | 0.00000000e+00 |
| 0.00000000e+00 | -3.58891009e-05 | -1.15237693e-02 | -7.16114874e-02 | 5.95787276e-04 | -1.74588806e-04 |
| 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 | 1.00000000e+00 | -1.30962462e-02 | 7.37754225e-03 |
| 0.00000000e+00 | -3.83057282e-03 | -1.63002990e-02 | -7.16114874e-02 | 5.95787276e-04 | -1.74588806e-04 |
| 0.00000000e+00 | -6.46168221e-03 | 3.78805728e-04 | -7.16114874e-02 | 5.95787276e-04 | -1.74588806e-04 |
| 0.00000000e+00 | 1.71729159e-03 | -2.25230367e-03 | -7.16114874e-02 | 5.95787276e-04 | -1.74588806e-04 |
| 0.00000000e+00 | -3.00665880e-03 | 5.92667014e-03 | -7.16114874e-02 | 5.95787276e-04 | -1.74588806e-04 |
| 0.00000000e+00 | -1.59003670e-04 | 1.20271975e-03 | -7.16114874e-02 | 5.95787276e-04 | -1.74588806e-04 |

Norma seluruh elemen subdiagonal kolom pertama, \(\|(H_1A)_{2:m,1}\|_2\), adalah **0.000000e+00** (toleransi verifikasi 1.40e-09). Elemen pertama kolom itu menjadi \(\alpha\), sedangkan elemen di bawahnya nol hingga galat pembulatan.

## Solusi

| Parameter | \(x_{LS}\) |
|---|---:|
| alpha1 | 0.00151337422439 |
| phi11 | 0.172152897275 |
| phi12 | 0.166072850953 |
| alpha2 | -0.00121268855387 |
| phi21 | -0.360376823937 |
| phi22 | -0.109808038547 |

Ukuran sistem: \(A\in\mathbb{R}^{300\times 6}\), \(b\in\mathbb{R}^{300}\), \(x_{LS}\in\mathbb{R}^6\). Norma residual \(\|Ax_{LS}-b\|_2\) = **0.15297705**. Norma bagian bawah-diagonal \(R\) setelah QR = 1.704147e-16.
