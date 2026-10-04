# TK1 Analisis Numerik:  Sistem Persamaan Linear dan Least Square Problem

**Kelompok C9**  
- Clarence Grady  
- Fakhri Husaini Romza  
- Muhammad Farrel Rajendra  
- Nicholas  

## Prasyarat (prerequisite)
| Kebutuhan |	Versi yang dipakai saat development	Keterangan |  
| --------- | ------------------------------------------------ |
| Python	| disarankan 3.10+  |
| numpy	| disarankan 1.24+ (array, I/O CSV, operasi minor, bukan solver utama) |
| Matplotlib | 3.x+ | Grafik bagian vii Nomor 2 |
  
Jika belum ada `numpy`, bisa dipasang dengan perintah ini:  
```
pip install numpy matplotlib
```    
  
clone repository ini dengan jalankan command ini di cmd:  
```  
git clone https://github.com/farrelzzz/Anum-TK1.git
```  
  
  
## Struktur Folder

```
Anum-TK1/
|-- Nomor1/   # solver sistem linear dan dataset T
|-- Nomor2/   # analisis SETAR dan dataset saham
`-- README.md
```

## Nomor 1    
  
### Struktur Nomor 1  
```
Nomor1/                     
├── T_16.csv  
├── T_32.csv  
├── T_64.csv  
├── T_128.csv  
├── T_256.csv  
├── T_512.csv  
├── i_validasi_data.py        (bagian i)  
├── ii_formulasi.py             (bagian ii)  
├── iii_struktur_matriks.py      (bagian iii)  
├── iv_a_solver_dense.py          (bagian iv-a)  
├── iv_b_solver_banded.py         (bagian iv-b)  
└── iv_reference_check.py       (HANYA alat bantu verifikasi, bukan solver laporan)     
``` 

### Dataset
Karena ini kelompok C9, berdasarkan ketentuan di soal, kelompok bernomor ganjil menggunakan kode A.
  
### Kode    
Note: kode juga bisa dieksekusi dengan command `python3`, tidak harus `python`, atau bisa juga klik tombol Run via GUI VS Code di file yang ingin dieksekusi, menyesuaikan saja    

Eksekusi kode untuk nomor 1 dengan jalankan command di bawah ini:      
  
**Wajib Berada di Direktori Nomor1 Sebelum Eksekusi Program**
```
cd Anum-TK1/Nomor1  
```  


#### **Bagian i - Validasi Data**
```
python i_validasi_data.py  
```  
**Ekspektasi Output**  
|file        | N   |  square | nonneg | rows=1  |   unique_ss|
|------------|-----|---------|--------|---------|------------|
|T_16.csv    | 16  |  True   | True   | True    |   True|
|T_32.csv    | 32  |  True   | True   | True    |   True|
|T_64.csv    | 64  |  True   | True   | True    |   True|
|T_128.csv   | 128 |   True  |  True  |  True   |    True|
|T_256.csv   | 256 |   True  |  True  |  True   |    True|
|T_512.csv   | 512 |   True  |  True  |  True   |    True|


#### **Bagian ii - Formulasi dan Penanganan Singularitas** 
```
python ii_formulasi.py  
```  
**Ekspektasi Output**    
|file        | N   |  rank=N | cond(B) | 
|------------|-----|---------|--------|
| T_16.csv   | 16  |  True   |  6.376e+02 |
| T_32.csv   | 32  |  True   |  2.457e+03|
| T_64.csv   | 64  |  True   |  9.660e+03|
| T_128.csv  | 128 |   True  |   3.832e+04|
| T_256.csv  | 256 |   True  |   7.567e+08|
| T_512.csv  | 512 |   True  |   6.095e+05|

#### **Bagian iii - Identifikasi Struktur Matriks** 
```
python iii_struktur_matriks.py  
```    
**Ekspektasi Output** 
|file        | N   |  p | p | band_ok  | dense KB| banded KB|
|------------|-----|----|----|---------|----------|--------|
| T_16.csv   | 16  |  1 |  2 |    True |      2.0 |     0.5|
| T_32.csv   | 32  |  1 |  2 |    True |      8.0 |     1.0|
| T_64.csv   | 64  |  1 |  2 |    True |     32.0 |     2.0|
| T_128.csv  | 128 |   1|   2|     True|     128.0|     4.0|
| T_256.csv  | 256 |   1|   2|     True|     512.0|     8.0|
| T_512.csv  | 512 |   1|   2|     True|    2048.0|    16.0|

#### **Bagian iv-a - Solver Dense Berbasis Faktorisasi LU** 
```
python iv_a_solver_dense.py  
```    
**Ekspektasi Output**   
```  
  --- N=16 (solver dense PB=LU manual) ---
sum(pi) = 1.0
min(pi) =  0.03128143631647254
max|pi_manual - pi_numpy_ref| = 2.7755575615628914e-17
residual r = 2.0816681711721685e-17
e_norm = 0.0

--- N=32 (solver dense PB=LU manual) ---
sum(pi) = 1.0
min(pi) =  0.015590010077180366
max|pi_manual - pi_numpy_ref| = 3.0531133177191805e-16
residual r = 1.2018516789897274e-17
e_norm = 0.0

--- N=64 (solver dense PB=LU manual) ---
sum(pi) = 1.0
min(pi) =  0.007798834732920434
max|pi_manual - pi_numpy_ref| = 4.2327252813834093e-16
residual r = 1.274756208291111e-17
e_norm = 0.0

--- N=128 (solver dense PB=LU manual) ---
sum(pi) = 1.0
min(pi) =  0.0039022447321592495
max|pi_manual - pi_numpy_ref| = 3.642919299551295e-16
residual r = 7.949501638480627e-18
e_norm = 1.1102230246251565e-16

--- N=256 (solver dense PB=LU manual) ---
sum(pi) = 1.0
min(pi) =  0.0019520516270901413
max|pi_manual - pi_numpy_ref| = 6.379514971843747e-13
residual r = 5.822478602598276e-18
e_norm = 0.0

--- N=512 (solver dense PB=LU manual) ---
sum(pi) = 1.0
min(pi) =  0.0009762852453183952
max|pi_manual - pi_numpy_ref| = 1.3040783730655647e-15
residual r = 3.860734174794557e-18
e_norm = 0.0
```  


#### **Bagian iv-b - Solver Banded seperti Algoritma Thomas** 
```
python iv_b_solver_banded.py  
```  
**Ekspektasi Output**     
```  
--- N=16, p=1, q=2 (solver banded manual) ---
sum(pi) = 1.0
min(pi) = 0.03128143631647253
max|pi_banded - pi_dense_manual| =  2.7755575615628914e-17
max|pi_banded - pi_numpy_ref|    =  2.7755575615628914e-17
residual r = 1.962615573354719e-17
e_norm = 0.0

--- N=32, p=1, q=2 (solver banded manual) ---
sum(pi) = 1.0
min(pi) = 0.015590010077180364
max|pi_banded - pi_dense_manual| =  1.3877787807814457e-17
max|pi_banded - pi_numpy_ref|    =  3.0531133177191805e-16
residual r = 1.4304896245381992e-17
e_norm = 0.0

--- N=64, p=1, q=2 (solver banded manual) ---
sum(pi) = 1.0
min(pi) = 0.007798834732920435
max|pi_banded - pi_dense_manual| =  1.0408340855860843e-17
max|pi_banded - pi_numpy_ref|    =  4.2674197509029455e-16
residual r = 1.214306433183765e-17
e_norm = 0.0

--- N=128, p=1, q=2 (solver banded manual) ---
sum(pi) = 1.0
min(pi) = 0.0039022447321592473
max|pi_banded - pi_dense_manual| =  8.673617379884035e-18
max|pi_banded - pi_numpy_ref|    =  3.608224830031759e-16
residual r = 6.829613154537259e-18
e_norm = 0.0

--- N=256, p=1, q=2 (solver banded manual) ---
sum(pi) = 1.0
min(pi) = 0.0019520516270901415
max|pi_banded - pi_dense_manual| =  3.469446951953614e-18
max|pi_banded - pi_numpy_ref|    =  6.379514971843747e-13
residual r = 5.346776843295223e-18
e_norm = 0.0

--- N=512, p=1, q=2 (solver banded manual) ---
sum(pi) = 1.0
min(pi) = 0.0009762852453183946
max|pi_banded - pi_dense_manual| =  3.0357660829594124e-18
max|pi_banded - pi_numpy_ref|    =  1.3049457348035531e-15
residual r = 3.903127820947816e-18
e_norm = 0.0
```

##### **Alat Bantu Verifikasi (numpy.linalg.solve, HANYA sebagai pembanding, bukan solver laporan)** 
```
python iv_reference_check.py   
```  
**Ekspektasi Output**        
```    
--- N=16 ---
valid steady state indication:  True
sum(pi) =  1.0000000000000002
min(pi) =  0.031281436316472545
residual r =  2.0816681711721685e-17
e_norm =  0.0
pi[:5] =  [0.07120743 0.07694907 0.07705396 0.07501655 0.07438765]

--- N=32 ---
valid steady state indication:  True
sum(pi) =  1.0
min(pi) =  0.015590010077180281
residual r =  1.5612511283791264e-17
e_norm =  0.0
pi[:5] =  [0.03576983 0.03758315 0.03861181 0.03894369 0.03875832]

--- N=64 ---
valid steady state indication:  True
sum(pi) =  1.0
min(pi) =  0.007798834732920535
residual r =  1.1765467043548785e-17
e_norm =  1.1102230246251565e-16
pi[:5] =  [0.01792673 0.01842361 0.0188242  0.01912861 0.01934042]

--- N=128 ---
valid steady state indication:  True
sum(pi) =  0.9999999999999998
min(pi) =  0.0039022447321593284
residual r =  7.217888169571747e-18
e_norm =  2.220446049250313e-16
pi[:5] =  [0.00897386 0.0091032  0.00922082 0.00932655 0.00942034]

--- N=256 ---
valid steady state indication:  True
sum(pi) =  1.0
min(pi) =  0.0019520516268127746
residual r =  5.416672317624936e-18
e_norm =  0.0
pi[:5] =  [0.00448956 0.00452251 0.00455402 0.00458408 0.00461267]

--- N=512 ---
valid steady state indication:  True
sum(pi) =  0.9999999999999998
min(pi) =  0.0009762852453180921
residual r =  3.7021972030927625e-18
e_norm =  2.220446049250313e-16
pi[:5] =  [0.00224544 0.00225375 0.00226189 0.00226984 0.00227762]
```  

  
## Nomor 2  

## Struktur Nomor 2

```
Nomor2/
|-- i_formulate_overdetermined_matrix.py   # formulasi (bagian i)
|-- i_visualization.py                     # grafik return train
|-- ii_investigate_numeric_issues.py       # isu numerik (bagian ii)
|-- iii_normal_equation.py                 # persamaan normal (bagian iii)
|-- iv_qr_householder.py                   # QR Householder (bagian iv)
|-- setar_analysis_common.py               # utilitas data/prediksi bersama
|-- v_performance_comparison.py            # performa (bagian v)
|-- vi_test_evaluation.py                  # RMSE out-of-sample (bagian vi)
|-- vii_interpretation_visualization.py    # interpretasi/grafik (bagian vii)
|-- build_report.py                        # gabungkan laporan i-vii
|-- stock_train.csv
|-- stock_test.csv
`-- hasil/
    |-- i/
    |-- ii/
    |-- iii/
    |-- iv/
    |-- v/
    |-- vi/
    |-- vii/
    `-- laporan_nomor2.md                  # draf laporan gabungan
```


### Bagian i-vii - SETAR
Dari direktori utama proyek, jalankan seluruh bagian berurutan:
```
python Nomor2/i_formulate_overdetermined_matrix.py
python Nomor2/i_visualization.py
python Nomor2/ii_investigate_numeric_issues.py
python Nomor2/iii_normal_equation.py
python Nomor2/iv_qr_householder.py
python Nomor2/v_performance_comparison.py
python Nomor2/vi_test_evaluation.py
python Nomor2/vii_interpretation_visualization.py
python Nomor2/build_report.py
```
Kelompok C9 bernomor ganjil, sehingga bagian iv menggunakan Householder. Modul `setar_analysis_common.py` menyediakan pembacaan dataset, prediksi one-step, metrik, dan utilitas bersama; solver Persamaan Normal bagian iii tetap dipakai sebagai pembanding. Kebutuhan paket: NumPy dan Matplotlib.

Setiap bagian menulis laporan dan tabel ke subfoldernya masing-masing di `Nomor2/hasil/`. Return test pertama mencakup perubahan harga dari penutupan train terakhir ke harga test pertama, sehingga 103 harga test menghasilkan 103 return target. Perintah terakhir menggabungkan laporan bagian i-vii dan menyertakan grafik ke `Nomor2/hasil/laporan_nomor2.md`.

## Ketentuan Tentang README  
Kelompok Anda diharapkan juga melampirkan README untuk menjalankan
code yang digunakan, sehingga mendapatkan hasil yang sama dengan eksperimen
kelompok Anda    
  
Pada laporan, sertakan pseudocode atau penjelasan langkah utama setiap algoritma, hasil
uji pada ukuran kecil, tabel eksperimen, grafik, dan README yang menjelaskan cara menjalankan kode.  

Berkas README.md berisi instruksi eksekusi kode.    
