# Ders Veri Setleri

Bu klasördeki veriler [Ders Notu](../Ders_Notu.md)'ndaki kısa uygulamalarda kullanılır. Derse özel veriler **yapaydır (sentetik)** ama gerçekçidir. Nasıl üretildikleri [`codes/python/00_veri_setlerini_uret.py`](../codes/python/00_veri_setlerini_uret.py) dosyasında açıkça yazılıdır; betik sabit bir tohum (seed) kullandığı için her çalıştırıldığında aynı dosyaları üretir.

- **CSV** dosyaları Python (`pandas.read_csv`) ve R (`read.csv`) içindir.
- **ARFF** dosyaları WEKA içindir (Explorer → Preprocess → Open file…).
- Python ve R kodları veriyi doğrudan şu adresten okuyabilir (indirmeye gerek yoktur, Google Colab'da da çalışır):
  `https://raw.githubusercontent.com/erkanozhan/machinelearning/main/data/<dosya_adı>`

| Dosya | Satır | Sütunlar | Görev | Ders notu |
| :--- | :---: | :--- | :--- | :--- |
| `notlar.arff`, `notlar.csv` | 200 | `vize`, `final`, `durum` {gecti, kaldi} | Sınıflandırma. Kural: 0.4·vize + 0.6·final ≥ 60 → gecti (117 gecti, 83 kaldi) | 4.2, 5.2, 5.8, 7.4, 12.5 |
| `ev_fiyatlari.csv`, `.arff` | 240 | `metrekare`, `oda_sayisi`, `bina_yasi`, `kat`, `semt` {merkez, cevre, banliyo}, `otopark` {evet, hayir}, `fiyat_bin_tl` | Regresyon | 6.8, 9.4 |
| `kredi_basvurulari.csv`, `.arff` | 600 | `yas`, `aylik_gelir`, `borc_gelir_orani`, `calisma_yili`, `sehir`, `gecmis_gecikme`, `risk` {dusuk, yuksek} | Dengesiz sınıflandırma (~%20 yüksek risk). **Eksik değer içerir:** 30 gelir, 18 şehir | 5.7, 5.10, 9.3, 22.6 |
| `musteri_segmentleri.csv`, `.arff` | 300 | `yas`, `yillik_gelir_bin_tl`, `harcama_puani` | Kümeleme (etiket yok; 5 doğal grup) | 17.6 |
| `market_sepeti.csv`, `.arff` | 391 | 12 ürün (CSV: 1/0, ARFF: t/?) | Birliktelik kuralları | 18.5 |
| `weather.nominal.arff` | 14 | `outlook`, `temperature`, `humidity`, `windy`, `play` | Elle hesaplanan Naive Bayes ve karar ağacı örnekleri (Quinlan, 1986; WEKA ile de gelir) | 11, 12 |
| `weather_yeni_gun.arff` | 1 | (yukarıdakiyle aynı) | Naive Bayes ile yeni günü tahmin etmek | 11.5 |
| `iris_yeni_test.arff` | 5 | Iris öznitelikleri | Kaydedilmiş WEKA modeliyle tahmin | 15 |
| `insanlar.csv` | 50 | `ad`, `yas`, `boy`, `kilo`, `ayak_no` | Proje ödevi | 25 |
| `musteri_ticaret.csv` | 90 | Boy, kilo, yaş, harcama, eğitim, ödeme yöntemi | Proje ödevi (ek alıştırma) | 25 |

Iris, meme kanseri ve diyabet gibi klasik veri setleri kodlarda doğrudan `scikit-learn` içinden (`load_iris()` vb.) veya WEKA'nın kendi `data/` klasöründen kullanılır; bu yüzden burada ayrıca bulunmazlar.
