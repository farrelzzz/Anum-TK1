# Nomor 2 bagian i — Formulasi matriks overdetermined

Harga Close train berjumlah 303. Return harian dihitung sebagai \(R_t=(P_t-P_{t-1})/P_{t-1}\), sehingga terdapat 302 return. Untuk lag order 2, baris matriks dimulai pada indeks return ke-3. Rezim ditentukan dari tanda return lag-1: \(R_{t-1}\ge0\) bullish dan \(R_{t-1}<0\) bearish.

Baris desain menggunakan urutan parameter \(x=[\alpha_1,\phi_{1,1},\phi_{1,2},\alpha_2,\phi_{2,1},\phi_{2,2}]^T\). Untuk rezim bullish barisnya \([1,R_{t-1},R_{t-2},0,0,0]\); untuk bearish \([0,0,0,1,R_{t-1},R_{t-2}]\). Target adalah \(b=R_t\).

Dimensi sistem: \(A\in\mathbb{R}^{300\times6}\), \(x\in\mathbb{R}^6\), dan \(b\in\mathbb{R}^{300}\). Baris bullish: 195; baris bearish: 105. Sistem overdetermined karena 300 > 6.

Data yang digunakan ulang tersedia di `return_train_i.csv`, `matriks_A_i.csv`, dan `vektor_b_i.csv`.
