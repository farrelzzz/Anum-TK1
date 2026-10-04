"""Gabungkan laporan Markdown bagian i-vii menjadi satu draf laporan Nomor 2."""
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "hasil"
REPORTS = [
    RESULTS / "i" / "laporan_i.md",
    RESULTS / "ii" / "laporan_ii.md",
    RESULTS / "iii" / "laporan_iii.md",
    RESULTS / "iv" / "laporan_iv.md",
    RESULTS / "v" / "laporan_v.md",
    RESULTS / "vi" / "laporan_vi.md",
    RESULTS / "vii" / "laporan_vii.md",
]


def main() -> None:
    missing = [str(path.relative_to(HERE)) for path in REPORTS if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Jalankan skrip bagian i sampai vii terlebih dahulu; laporan belum ada: "
            + ", ".join(missing)
        )
    sections = [path.read_text(encoding="utf-8").strip() for path in REPORTS]
    section_paths = ["i/", "ii/", "iii/", "iv/", "v/", "vi/", "vii/"]
    artifacts = [
        ("return_train_i.csv", "i/return_train_i.csv"),
        ("matriks_A_i.csv", "i/matriks_A_i.csv"),
        ("vektor_b_i.csv", "i/vektor_b_i.csv"),
        ("diagnostik_ii.csv", "ii/diagnostik_ii.csv"),
        ("koefisien_normal_iii.csv", "iii/koefisien_normal_iii.csv"),
        ("H1_iv.csv", "iv/H1_iv.csv"),
        ("H1A_iv.csv", "iv/H1A_iv.csv"),
        ("x_ls_iv.csv", "iv/x_ls_iv.csv"),
        ("perbandingan_v.csv", "v/perbandingan_v.csv"),
        ("evaluasi_vi.csv", "vi/evaluasi_vi.csv"),
        ("model_terpilih.csv", "vi/model_terpilih.csv"),
        ("prediksi_vi.csv", "vi/prediksi_vi.csv"),
        ("koefisien_vii.csv", "vii/koefisien_vii.csv"),
    ]
    for i, section in enumerate(sections):
        for filename, relative_path in artifacts:
            if relative_path.startswith(section_paths[i]):
                section = section.replace(f"`{filename}`", f"`{relative_path}`")
        if i == 6:
            section = section.replace(
                "`overlay_train_vii.png`", "[overlay_train_vii.png](vii/overlay_train_vii.png)"
            ).replace(
                "`overlay_train_test_vii.png`",
                "[overlay_train_test_vii.png](vii/overlay_train_test_vii.png)",
            )
        sections[i] = section
    text = "# Laporan Teknis Nomor 2 — Model SETAR\n\n"
    text += "**Kelompok C9**\n\n" + "\n\n---\n\n".join(sections)
    text += """

## Grafik pendukung

![Return harian train](i/return_train_i.png)

![Overlay return aktual dan estimasi pada train](vii/overlay_train_vii.png)

![Overlay kontinu train dan test](vii/overlay_train_test_vii.png)
"""
    destination = RESULTS / "laporan_nomor2.md"
    destination.write_text(text, encoding="utf-8")
    print(f"Laporan gabungan tersimpan di {destination}")


if __name__ == "__main__":
    main()
