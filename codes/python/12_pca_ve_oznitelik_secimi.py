# -*- coding: utf-8 -*-
"""
12 - Boyut Azaltma: PCA ve Öznitelik Seçimi
===========================================
Ders notu bölümü: "19. Boyut Azaltma"

  1) PCA'yı sıfırdan yazar: standartlaştır → kovaryans matrisi → özdeğer/özvektör → izdüşüm
  2) Aynı sonucu sklearn PCA ile doğrular; açıklanan varyans oranlarını yazdırır
  3) PCA'lı ve PCA'sız sınıflandırma başarısını (Pipeline içinde, sızıntısız) karşılaştırır
  4) Filtre yöntemi (karşılıklı bilgi / mutual information) ile öznitelik puanlar
  5) Sarmalayıcı (wrapper) yöntem: RFE (Recursive Feature Elimination)
  6) Gömülü (embedded) yöntem: L1 (Lasso) cezalı lojistik regresyonun sıfırladığı katsayılar

Çalıştırma:  python 12_pca_ve_oznitelik_secimi.py
"""

import numpy as np                                            # Sayısal işlemler
import matplotlib.pyplot as plt                               # Grafik
from sklearn.datasets import load_iris, load_breast_cancer    # Veri setleri
from sklearn.preprocessing import StandardScaler              # PCA öncesi standartlaştırma ŞART
from sklearn.decomposition import PCA                         # Temel bileşen analizi
from sklearn.pipeline import make_pipeline                    # Sızıntısız zincir
from sklearn.tree import DecisionTreeClassifier               # Karşılaştırma için sınıflandırıcı
from sklearn.linear_model import LogisticRegression           # RFE ve L1 için
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif, RFE   # Filtre ve sarmalayıcı yöntemler

iris = load_iris()
X, y = iris.data, iris.target

# ---------------------------------------------------------------------------
# 1) PCA'YI ELLE HESAPLAMAK
# ---------------------------------------------------------------------------
Xs = (X - X.mean(axis=0)) / X.std(axis=0)                     # Z-skoru: her sütun ortalama 0, std 1
kovaryans = np.cov(Xs, rowvar=False)                          # 4x4 kovaryans matrisi (sütunlar değişken)
ozdegerler, ozvektorler = np.linalg.eigh(kovaryans)           # Simetrik matris için özdeğer ayrışımı
sira = np.argsort(ozdegerler)[::-1]                           # Büyükten küçüğe sırala
ozdegerler, ozvektorler = ozdegerler[sira], ozvektorler[:, sira]
aciklanan = ozdegerler / ozdegerler.sum()                     # Her bileşenin açıkladığı varyans oranı
Z_elle = Xs @ ozvektorler[:, :2]                              # İlk 2 bileşene izdüşüm (projeksiyon)

print("=== 1) PCA (elle) ===")
print("Özdeğerler (λ):", np.round(ozdegerler, 3))
print("Açıklanan varyans oranı:", np.round(aciklanan, 4))
print("Kümülatif:", np.round(np.cumsum(aciklanan), 4))
print("PC1 yükleri (loadings):", {ad: round(float(v), 3) for ad, v in zip(iris.feature_names, ozvektorler[:, 0])})
print()

# ---------------------------------------------------------------------------
# 2) sklearn ile doğrulama
# ---------------------------------------------------------------------------
pca = PCA(n_components=2).fit(StandardScaler().fit_transform(X))
print("=== 2) sklearn PCA ===")
print("explained_variance_ratio_:", np.round(pca.explained_variance_ratio_, 4),
      "toplam:", round(pca.explained_variance_ratio_.sum(), 4))
print()

# ---------------------------------------------------------------------------
# 3) PCA sınıflandırmayı nasıl etkiliyor? (Pipeline → PCA her katmanda sadece eğitimle fit edilir)
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
print("=== 3) J48 benzeri karar ağacı: PCA'sız vs PCA'lı ===")
s_ham = cross_val_score(DecisionTreeClassifier(random_state=42), X, y, cv=cv)
print(f"4 orijinal öznitelik     : {s_ham.mean():.3f} ± {s_ham.std():.3f}")
for k in [1, 2, 3]:
    pipe = make_pipeline(StandardScaler(), PCA(n_components=k), DecisionTreeClassifier(random_state=42))
    s = cross_val_score(pipe, X, y, cv=cv)
    print(f"PCA ile {k} bileşen         : {s.mean():.3f} ± {s.std():.3f}")
print()

# ---------------------------------------------------------------------------
# 4) FİLTRE YÖNTEMİ: Karşılıklı bilgi (bilgi kazancının sürekli versiyonu)
# ---------------------------------------------------------------------------
mi = mutual_info_classif(X, y, random_state=0)
print("=== 4) Filtre: karşılıklı bilgi puanları ===")
for ad, puan in sorted(zip(iris.feature_names, mi), key=lambda t: -t[1]):
    print(f"  {ad:<20}: {puan:.3f}")
print()

# ---------------------------------------------------------------------------
# 5) SARMALAYICI YÖNTEM: RFE (modeli eğit → en zayıf özniteliği at → tekrarla)
# ---------------------------------------------------------------------------
Xb, yb = load_breast_cancer(return_X_y=True)                  # 30 öznitelikli veri
Xb_s = StandardScaler().fit_transform(Xb)
rfe = RFE(LogisticRegression(max_iter=5000), n_features_to_select=5).fit(Xb_s, yb)
adlar = load_breast_cancer().feature_names
print("=== 5) Sarmalayıcı: RFE ile seçilen 5 öznitelik (meme kanseri, 30 öznitelikten) ===")
print("  ", [str(a) for a in adlar[rfe.support_]])
print()

# ---------------------------------------------------------------------------
# 6) GÖMÜLÜ YÖNTEM: L1 cezası bazı katsayıları tam sıfır yapar
# ---------------------------------------------------------------------------
l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.05).fit(Xb_s, yb)
sifir_olmayan = np.flatnonzero(l1.coef_[0])
print("=== 6) Gömülü: L1 lojistik regresyon (C=0.05) ===")
print(f"Sıfır olmayan katsayı sayısı: {len(sifir_olmayan)} / 30")
print("  ", [str(a) for a in adlar[sifir_olmayan]])

# --- Grafik: Iris'in 2 boyutlu PCA izdüşümü ---
Z = pca.transform(StandardScaler().fit_transform(X))
for k, renk in zip(range(3), ["tab:blue", "tab:orange", "tab:green"]):
    plt.scatter(Z[y == k, 0], Z[y == k, 1], c=renk, label=iris.target_names[k], s=18)
plt.xlabel(f"PC1 (%{pca.explained_variance_ratio_[0] * 100:.1f})")
plt.ylabel(f"PC2 (%{pca.explained_variance_ratio_[1] * 100:.1f})")
plt.title("Iris veri setinin PCA ile 2 boyuta indirgenmesi")
plt.legend()
plt.show()
