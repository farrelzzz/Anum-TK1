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
### Dataset
Karena ini kelompok C9, berdasarkan ketentuan di soal, kelompok bernomor ganjil menggunakan kode A.
  
### Kode    
Note: kode juga bisa dieksekusi dengan command `python3`, tidak harus `python`, atau bisa juga klik tombol Run via GUI VS Code di file yang ingin dieksekusi, menyesuaikan saja    

Eksekusi kode untuk nomor 1 dengan jalankan command di bawah ini:      
  
**Wajib Berada di Direktori Nomor1 Sebelum Eksekusi Program**
```
cd Anum-TK1/Nomor1  
```  


**Bagian i - Validasi Data**
```
python i_validasi_data.py  
```

**Bagian ii - Formulasi dan Penanganan Singularitas** 
```
python ii_formulasi.py  
```

**Bagian iii - Identifikasi Struktur Matriks** 
```
python iii_struktur_matriks.py  
```

**Bagian iv-a - Solver Dense Berbasis Faktorisasi LU** 
```
python iv_a_solver_dense.py  
```  


**Bagian iv-b - Solver Banded seperti Algoritma Thomas** 
```
python iv_b_solver_banded.py  
```

**Alat Bantu Verifikasi (numpy.linalg.solve, HANYA sanity-check, bukan solver laporan)** 
```
python iv_reference_check.py   
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
