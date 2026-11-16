# Ek C: Kodlar ve Animasyonlar

**Python kodları** ([`codes/python/`](https://github.com/erkanozhan/machinelearning/tree/main/codes/python)). Kurulum: `pip install -r codes/requirements.txt`

| Dosya | Konu | Bölüm |
| :--- | :--- | :---: |
| [`00_veri_setlerini_uret.py`](https://github.com/erkanozhan/machinelearning/blob/main/codes/python/00_veri_setlerini_uret.py) | Derse özel yapay veri setlerini üretir (`data/` klasörü) | — |
| [`01_istatistik_ve_olceklendirme.py`](https://github.com/erkanozhan/machinelearning/blob/main/codes/python/01_istatistik_ve_olceklendirme.py) | Temel istatistikler, Min-Max, Z-skoru, Robust, onluk ölçekleme | 5 |
| [`02_lineer_regresyon.py`](https://github.com/erkanozhan/machinelearning/blob/main/codes/python/02_lineer_regresyon.py) | En küçük kareler, sıfırdan gradyan inişi, öğrenme oranı deneyi | 6 |
| [`03_siniflandirma_algoritmalari.py`](https://github.com/erkanozhan/machinelearning/blob/main/codes/python/03_siniflandirma_algoritmalari.py) | Lojistik regresyon, k-NN, Naive Bayes, karar ağacı, MLP | 7, 10–12, 14 |
| [`04_entropi_ve_naive_bayes_elle.py`](https://github.com/erkanozhan/machinelearning/blob/main/codes/python/04_entropi_ve_naive_bayes_elle.py) | Entropi, bilgi kazancı, Naive Bayes (kütüphanesiz) | 11–12 |
| [`05_model_degerlendirme_yontemleri.py`](https://github.com/erkanozhan/machinelearning/blob/main/codes/python/05_model_degerlendirme_yontemleri.py) | Holdout, üçlü ayırma, K-fold, LOOCV, bootstrap/OOB | 8 |
| [`06_siniflandirma_metrikleri.py`](https://github.com/erkanozhan/machinelearning/blob/main/codes/python/06_siniflandirma_metrikleri.py) | Karışıklık matrisi, tüm metrikler, ROC ve PR eğrileri | 9 |
| [`07_regresyon_metrikleri.py`](https://github.com/erkanozhan/machinelearning/blob/main/codes/python/07_regresyon_metrikleri.py) | MAE, MSE, RMSE, R², düzeltilmiş R², korelasyon | 9 |
| [`08_topluluk_ogrenmesi.py`](https://github.com/erkanozhan/machinelearning/blob/main/codes/python/08_topluluk_ogrenmesi.py) | Bagging, RF, AdaBoost, GB, Stacking (sınıflandırma + regresyon) | 16 |
| [`09_svm.py`](https://github.com/erkanozhan/machinelearning/blob/main/codes/python/09_svm.py) | Marj, C etkisi, çekirdekler, karar sınırları | 13 |
| [`10_kumeleme.py`](https://github.com/erkanozhan/machinelearning/blob/main/codes/python/10_kumeleme.py) | Sıfırdan K-Means, dirsek, siluet, hiyerarşik, DBSCAN | 17 |
| [`11_birliktelik_kurallari_apriori.py`](https://github.com/erkanozhan/machinelearning/blob/main/codes/python/11_birliktelik_kurallari_apriori.py) | Apriori (kütüphanesiz) | 18 |
| [`11b_apriori_ve_fpgrowth_mlxtend.py`](https://github.com/erkanozhan/machinelearning/blob/main/codes/python/11b_apriori_ve_fpgrowth_mlxtend.py) | `mlxtend` ile Apriori ve FP-Growth | 18 |
| [`12_pca_ve_oznitelik_secimi.py`](https://github.com/erkanozhan/machinelearning/blob/main/codes/python/12_pca_ve_oznitelik_secimi.py) | Sıfırdan PCA, filtre/sarmalayıcı/gömülü seçim | 19 |
| [`13_maliyete_duyarli_ogrenme.py`](https://github.com/erkanozhan/machinelearning/blob/main/codes/python/13_maliyete_duyarli_ogrenme.py) | Sınıf ağırlığı, oversampling, eşik ayarı | 22 |
| [`14_hiperparametre_optimizasyonu.py`](https://github.com/erkanozhan/machinelearning/blob/main/codes/python/14_hiperparametre_optimizasyonu.py) | Grid, random, nested CV, erken durdurma | 20 |
| [`15_veri_sizintisi.py`](https://github.com/erkanozhan/machinelearning/blob/main/codes/python/15_veri_sizintisi.py) | Sızıntı deneyi (yanlış vs doğru) | 21 |

**R kodları** ([`codes/R/`](https://github.com/erkanozhan/machinelearning/tree/main/codes/R)): `grid_ve_random_search.R`, `nested_cv.R`. Kurulum: `install.packages(c("caret", "mlbench"))`

**Veri dosyaları** ([`data/`](https://github.com/erkanozhan/machinelearning/tree/main/data)). Not içindeki kısa kodlar bu verileri `https://raw.githubusercontent.com/erkanozhan/machinelearning/main/data/` adresinden doğrudan okur; `codes/` klasöründeki betiklerin çoğu ise scikit-learn ile gelen hazır veri setlerini kullanır. Derse özel veriler yapaydır (sentetiktir); nasıl üretildikleri [`codes/python/00_veri_setlerini_uret.py`](https://github.com/erkanozhan/machinelearning/blob/main/codes/python/00_veri_setlerini_uret.py) dosyasında açıkça yazılıdır.

| Veri seti | Biçim | Satır | Görev | Kullanıldığı bölüm |
| :--- | :--- | :---: | :--- | :--- |
| [`data/notlar.arff`](https://github.com/erkanozhan/machinelearning/blob/main/data/notlar.arff), `notlar.csv` | ARFF, CSV | 200 | Sınıflandırma (gecti/kaldi) | 4.2, 5.2, 5.8, 7.4, 12.5 |
| [`data/ev_fiyatlari.csv`](https://github.com/erkanozhan/machinelearning/blob/main/data/ev_fiyatlari.csv), `.arff` | CSV, ARFF | 240 | Regresyon (fiyat) | 6.8, 9.4 |
| [`data/kredi_basvurulari.csv`](https://github.com/erkanozhan/machinelearning/blob/main/data/kredi_basvurulari.csv), `.arff` | CSV, ARFF | 600 | Dengesiz sınıflandırma, eksik değerli | 5.7, 5.10, 9.3, 22.6 |
| [`data/musteri_segmentleri.csv`](https://github.com/erkanozhan/machinelearning/blob/main/data/musteri_segmentleri.csv), `.arff` | CSV, ARFF | 300 | Kümeleme (etiketsiz) | 17.6 |
| [`data/market_sepeti.csv`](https://github.com/erkanozhan/machinelearning/blob/main/data/market_sepeti.csv), `.arff` | CSV, ARFF | 391 | Birliktelik kuralları | 18.6 |
| [`data/weather.nominal.arff`](https://github.com/erkanozhan/machinelearning/blob/main/data/weather.nominal.arff) | ARFF | 14 | Sınıflandırma (elle hesap) | 11.3, 11.5, 12.2, 12.5 |
| [`data/weather_yeni_gun.arff`](https://github.com/erkanozhan/machinelearning/blob/main/data/weather_yeni_gun.arff) | ARFF | 1 | Naive Bayes ile yeni tahmin | 11.5 |
| [`data/meme_kanseri.csv`](https://github.com/erkanozhan/machinelearning/blob/main/data/meme_kanseri.csv), `.arff` | CSV, ARFF | 569 | Sınıflandırma (gerçek veri; Python'da `load_breast_cancer()`) | 7.4, 8, 9.5, 12.5, 16.5, 19.3, 20.5, 20.6 |
| [`data/iris_yeni_test.arff`](https://github.com/erkanozhan/machinelearning/blob/main/data/iris_yeni_test.arff) | ARFF | 5 | Kaydedilmiş modelle tahmin | 15 |
| [`data/insanlar.csv`](https://github.com/erkanozhan/machinelearning/blob/main/data/insanlar.csv), [`data/musteri_ticaret.csv`](https://github.com/erkanozhan/machinelearning/blob/main/data/musteri_ticaret.csv) | CSV | 47, 90 | Proje ödevi | 25 |


**Etkileşimli animasyonlar** (tarayıcıda açılır):

| Animasyon | Konu |
| :--- | :--- |
| [Gradyan İnişi](https://erkanozhan.github.io/machinelearning/animation/gradyan_inisi_animasyonu.html) | Öğrenme oranının lineer regresyon eğitimine etkisi |
| [k-En Yakın Komşu (k-NN)](https://erkanozhan.github.io/machinelearning/animation/knn_animasyonu.html) | Mesafe hesabı, komşu seçimi, çoğunluk oyu, k'nın ve mesafe ölçüsünün etkisi, karar bölgeleri |
| [K-Means Şeffaf Algoritma Simülasyonu](https://erkanozhan.github.io/machinelearning/animation/k-means_interactive_animation.html) | Her noktanın merkezlere uzaklığı, atama ve merkez güncelleme adımları |
| [K-Means Deney Ortamı](https://erkanozhan.github.io/machinelearning/animation/kmeans_animasyonu.html) | K seçimi, rastgele vs k-means++ başlangıç, yerel minimum, SSE grafiği |
| [ROC Eğrisi](https://erkanozhan.github.io/machinelearning/roc_animation.html) | Eşik değiştikçe ROC eğrisinin oluşumu |
| [Nested Cross-Validation](https://erkanozhan.github.io/machinelearning/animation/nested_cv_animation.html) | İç ve dış döngülerin işleyişi |

---

[← Ek B: Türkçe–İngilizce Terimler](ek-b-terimler-sozlugu.md) · [İçindekiler](../Ders_Notu.md) · [Kaynaklar →](kaynaklar.md)
