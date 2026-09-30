# Ders Kodları

Bu klasör, [Ders Notu](../Ders_Notu.md)'nda anlatılan konuların uygulama dersinde çalıştırılacak kodlarını içerir. Her satırda Türkçe açıklama bulunur.

## Kurulum

```bash
pip install -r requirements.txt     # numpy, scipy, scikit-learn, matplotlib
```

Kurulum yapmadan çalıştırmak için dosya içeriğini [Google Colab](https://colab.research.google.com)'a yapıştırabilirsiniz.

## Python (`python/`)

| Dosya | Konu | Ders notu bölümü |
| :--- | :--- | :---: |
| `01_istatistik_ve_olceklendirme.py` | Ortalama, varyans, std; Min-Max, Z-skoru, Robust, onluk ölçekleme | 5 |
| `02_lineer_regresyon.py` | En küçük kareler, sıfırdan gradyan inişi, öğrenme oranı deneyi | 6 |
| `03_siniflandirma_algoritmalari.py` | Lojistik regresyon, k-NN, Naive Bayes, karar ağacı, MLP karşılaştırması | 7–12 |
| `04_entropi_ve_naive_bayes_elle.py` | Entropi, bilgi kazancı, kazanç oranı, Naive Bayes (kütüphanesiz) | 9–10 |
| `05_model_degerlendirme_yontemleri.py` | Holdout, üçlü ayırma, K-fold, LOOCV, bootstrap/OOB | 13 |
| `06_siniflandirma_metrikleri.py` | Karışıklık matrisi, accuracy…MCC, ROC ve PR eğrileri | 14 |
| `07_regresyon_metrikleri.py` | MAE, MSE, RMSE, R², düzeltilmiş R², korelasyon | 14 |
| `08_topluluk_ogrenmesi.py` | Bagging, Random Forest, AdaBoost, Gradient Boosting, Stacking | 16 |
| `09_svm.py` | Marj, C parametresi, çekirdekler, karar sınırları | 11 |
| `10_kumeleme.py` | Sıfırdan K-Means, dirsek, siluet, hiyerarşik, DBSCAN | 17 |
| `11_birliktelik_kurallari_apriori.py` | Apriori algoritması (kütüphanesiz) | 18 |
| `12_pca_ve_oznitelik_secimi.py` | Sıfırdan PCA, filtre / sarmalayıcı / gömülü öznitelik seçimi | 19 |
| `13_maliyete_duyarli_ogrenme.py` | Sınıf ağırlığı, oversampling, maliyete göre eşik | 20 |
| `14_hiperparametre_optimizasyonu.py` | Grid Search, Random Search, Nested CV, erken durdurma | 21 |
| `15_veri_sizintisi.py` | Veri sızıntısı deneyi: yanlış ve doğru yöntem | 22 |

Çalıştırma örneği:

```bash
cd python
python 01_istatistik_ve_olceklendirme.py
```

## R (`R/`)

| Dosya | Konu |
| :--- | :--- |
| `grid_ve_random_search.R` | `caret` ile Grid Search ve Random Search (glmnet) |
| `nested_cv.R` | `caret` ile iç içe çapraz doğrulama |

Gerekli paketler: `install.packages(c("caret", "glmnet", "mlbench"))`
