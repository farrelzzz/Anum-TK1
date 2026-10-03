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
| numpy	| disarankan 1.24+ (Dipakai HANYA untuk array container, I/O CSV, dan operasi minor (rank, cond, norm), bukan untuk numpy.linalg.solve sebagai solver utama) |
  
Jika belum ada `numpy`, bisa download dahulu dengan jalnakn command ini di cmd:  
```
pip install numpy
```    
  
clone repository ini dengan jalankan command ini di cmd:  
```  
git clone https://github.com/farrelzzz/Anum-TK1.git
```  
  
  
## Struktur Folder    
```
Anum-TK1/  
├── Nomor1/                     
│   ├── T_16.csv  
│   ├── T_32.csv  
│   ├── T_64.csv  
│   ├── T_128.csv  
│   ├── T_256.csv  
│   ├── T_512.csv  
│   ├── i_validasi_data.py        (bagian i)  
│   ├── ii_formulasi.py             (bagian ii)  
│   ├── iii_struktur_matriks.py      (bagian iii)  
│   ├── iv_a_solver_dense.py          (bagian iv-a)  
│   ├── iv_b_solver_banded.py         (bagian iv-b)  
│   └── iv_reference_check.py       (HANYA alat bantu verifikasi, bukan solver laporan)    
├── Nomor2/     
│   ├── stock_test.csv  
│   ├── stock_train.csv                  
│   ├──                      (bagian i dan seterusnya)    
│   └──        
└── README.md                (file ini)     
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
  

## Ketentuan Tentang README  
Kelompok Anda diharapkan juga melampirkan README untuk menjalankan
code yang digunakan, sehingga mendapatkan hasil yang sama dengan eksperimen
kelompok Anda    
  
Pada laporan, sertakan pseudocode atau penjelasan langkah utama setiap algoritma, hasil
uji pada ukuran kecil, tabel eksperimen, grafik, dan README yang menjelaskan cara menjalankan kode.  

Berkas README.md berisi instruksi eksekusi kode.    
