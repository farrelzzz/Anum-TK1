# Nomor 2 bagian ii — Investigasi isu numerik

Matriks desain berukuran \(300\times6\), berpangkat 6 dari 6 parameter. Condition number 2-norm \(\kappa_2(A)\) adalah 192.60541. Norma kolom beragam: minimum 0.08897615, maksimum 13.96424; perbedaan skala ini dapat membuat pengujian pivot berdasarkan ambang absolut saja kurang andal.

Membandingkan pembentukan \(A^TA\) dengan float32 dan float64 menghasilkan galat absolut maksimum 1.6104009e-07 dan galat relatif maksimum 8.2584661e-10. Galat float32/float64 tidak seharusnya diharapkan nol; perbedaan tersebut adalah akibat pembulatan floating point. Risiko yang lebih besar muncul ketika persamaan normal membentuk \(A^TA\), sebab \(\kappa_2(A^TA)\approx\kappa_2(A)^2\), sehingga galat pembulatan lebih kuat memengaruhi solusi. Rank penuh pada data ini menunjukkan tidak ada kolom yang redundan secara numerik pada toleransi rank NumPy.

Mitigasi teknis yang disarankan:

1. Simpan harga, return, dan matriks dalam float64; jangan menurunkan presisi sebelum membentuk matriks.
2. Gunakan QR Householder sebagai solver utama karena transformasi ortogonal menghindari pembentukan \(A^TA\).
3. Bila persamaan normal dipakai sebagai pembanding, gunakan partial pivoting dan tolak pivot yang nol atau terlalu kecil terhadap skala matriks.
4. Periksa rank, condition number, norma kolom, serta residual solusi; penskalaan kolom dapat dipertimbangkan bila rentang skala menghambat perhitungan, dengan parameter dikembalikan ke skala semula.

Metrik rinci tersedia dalam `diagnostik_ii.csv`.
