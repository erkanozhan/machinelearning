# -*- coding: utf-8 -*-
"""
03 - Temel Sınıflandırma Algoritmalarının Karşılaştırılması
===========================================================
Ders notu bölümleri: "7. Lojistik Regresyon", "10. k-En Yakın Komşu",
                     "11. Naive Bayes", "12. Karar Ağaçları", "14. Yapay Sinir Ağları"

Iris veri seti üzerinde beş algoritmayı AYNI 10 katlı tabakalı çapraz doğrulama
bölmeleriyle karşılaştırır. Ayrıca:
  - Lojistik regresyonun sigmoid çıktısını (olasılık) gösterir
  - k-NN'de k değerinin etkisini gösterir
  - Karar ağacının öğrendiği kuralları metin olarak yazdırır

Çalıştırma:  python 03_siniflandirma_algoritmalari.py
"""

import numpy as np                                            # Sayısal işlemler
from sklearn.datasets import load_iris                        # Iris veri seti (150 çiçek, 4 öznitelik, 3 sınıf)
from sklearn.model_selection import StratifiedKFold, cross_val_score   # Tabakalı K-katlı CV ve skorlama
from sklearn.pipeline import make_pipeline                    # Ölçekleme + model zinciri (veri sızıntısını önler)
from sklearn.preprocessing import StandardScaler              # Z-skoru ölçekleyici
from sklearn.linear_model import LogisticRegression           # Lojistik regresyon
from sklearn.neighbors import KNeighborsClassifier            # k-En Yakın Komşu
from sklearn.naive_bayes import GaussianNB                    # Sayısal öznitelikler için Naive Bayes
from sklearn.tree import DecisionTreeClassifier, export_text  # Karar ağacı ve kurallarını yazdırma
from sklearn.neural_network import MLPClassifier              # Çok katmanlı algılayıcı (yapay sinir ağı)

# ---------------------------------------------------------------------------
# 1) VERİ
# ---------------------------------------------------------------------------
iris = load_iris()                                   # Veri setini yükle
X, y = iris.data, iris.target                        # X: öznitelik matrisi (150x4), y: sınıf etiketleri (0,1,2)
oznitelik_adlari = iris.feature_names                # ['sepal length (cm)', ...]

# Tüm modeller aynı bölmelerle test edilsin diye CV nesnesini tek sefer tanımlıyoruz
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ---------------------------------------------------------------------------
# 2) MODELLER
#    Mesafe veya gradyan kullanan modellerin (LR, k-NN, MLP) önüne StandardScaler koyuyoruz.
#    Pipeline sayesinde ölçekleyici her katmanda SADECE eğitim kısmıyla fit edilir.
# ---------------------------------------------------------------------------
modeller = {
    "Lojistik Regresyon": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "k-NN (k=5)":         make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Naive Bayes":        GaussianNB(),                                   # Ölçeklemeye duyarlı değil
    "Karar Ağacı":        DecisionTreeClassifier(criterion="entropy", random_state=42),   # Ölçeklemeye duyarlı değil
    "Yapay Sinir Ağı":    make_pipeline(StandardScaler(),
                                        MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000, random_state=42)),
}

print("=== 1) 10 katlı tabakalı CV ile doğruluk (ortalama ± std) ===")
for ad, model in modeller.items():                   # Her modeli sırayla değerlendir
    skorlar = cross_val_score(model, X, y, cv=cv, scoring="accuracy")   # 10 ayrı test skoru
    print(f"{ad:<20}: {skorlar.mean():.3f} ± {skorlar.std():.3f}")
print()

# ---------------------------------------------------------------------------
# 3) LOJİSTİK REGRESYON: sigmoid ve olasılık çıktısı
# ---------------------------------------------------------------------------
def sigmoid(z):
    """σ(z) = 1 / (1 + e^(-z)) : Her gerçek sayıyı (0, 1) aralığına sıkıştırır."""
    return 1 / (1 + np.exp(-z))

print("=== 2) Sigmoid fonksiyonu ===")
for z in [-5, -2, 0, 2, 5]:                          # Birkaç örnek z değeri
    print(f"σ({z:>2}) = {sigmoid(z):.4f}")
print()

lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)
ornek = X[[0, 60, 120]]                              # Her sınıftan birer çiçek
olasiliklar = lr.predict_proba(ornek)                # Her sınıf için olasılık (satır toplamı = 1)
print("=== 3) Lojistik regresyonun olasılık çıktıları ===")
for i, p in enumerate(olasiliklar):
    print(f"Örnek {i}: " + ", ".join(f"{iris.target_names[k]}={p[k]:.3f}" for k in range(3)))
print()

# ---------------------------------------------------------------------------
# 4) k-NN: k değerinin etkisi
# ---------------------------------------------------------------------------
print("=== 4) k-NN'de k değerinin etkisi ===")
for k in [1, 3, 5, 15, 51, 101]:                     # Çok küçük k → ezber, çok büyük k → aşırı basit
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    s = cross_val_score(knn, X, y, cv=cv).mean()
    print(f"k = {k:<4}: doğruluk = {s:.3f}")
print()

# ---------------------------------------------------------------------------
# 5) KARAR AĞACI: öğrenilen kurallar
# ---------------------------------------------------------------------------
agac = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42).fit(X, y)
print("=== 5) Karar ağacının kuralları (max_depth=3) ===")
print(export_text(agac, feature_names=list(oznitelik_adlari)))   # Ağacı okunabilir if-else kuralları olarak yazdırır
print("Öznitelik önemleri:")
for ad, onem in zip(oznitelik_adlari, agac.feature_importances_):
    print(f"  {ad:<20}: {onem:.3f}")
