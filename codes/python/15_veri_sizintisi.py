# -*- coding: utf-8 -*-
"""
15 - Veri Sızıntısı (Data Leakage) Deneyi
=========================================
Ders notu bölümü: "21. Veri Sızıntısı (Data Leakage)"

Klasik ve çarpıcı bir deney:
  - Tamamen RASTGELE (hiçbir anlamı olmayan) 10.000 öznitelik ve rastgele etiketler üretiyoruz.
  - Gerçek başarı %50 olmalı (yazı-tura).
  - YANLIŞ: Öznitelik seçimini TÜM veride yapıp sonra CV uygularsak → %80-90 gibi sahte başarı!
  - DOĞRU: Seçimi Pipeline içine koyarsak (her katmanda sadece eğitim kısmına bakar) → ~%50.

Çalıştırma:  python 15_veri_sizintisi.py
"""

import numpy as np                                            # Rastgele veri üretimi
from sklearn.feature_selection import SelectKBest, f_classif  # En iyi k özniteliği seçen filtre
from sklearn.linear_model import LogisticRegression           # Sınıflandırıcı
from sklearn.pipeline import make_pipeline                    # Sızıntıyı önleyen zincir
from sklearn.model_selection import cross_val_score, StratifiedKFold

rng = np.random.default_rng(0)                                # Tekrarlanabilir rastgelelik
X = rng.normal(size=(100, 10_000))                            # 100 örnek, 10.000 anlamsız öznitelik
y = rng.integers(0, 2, size=100)                              # Rastgele 0/1 etiketler (öğrenilecek hiçbir şey yok!)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# ---------------------------------------------------------------------------
# YANLIŞ: Önce tüm veride seçim, sonra CV
# ---------------------------------------------------------------------------
secici = SelectKBest(f_classif, k=20).fit(X, y)               # Etiketlere bakarak seçim yaptı — TEST katmanları dahil!
X_secili = secici.transform(X)
yanlis = cross_val_score(LogisticRegression(max_iter=1000), X_secili, y, cv=cv)
print(f"YANLIŞ yöntem (seçim CV'den önce): doğruluk = {yanlis.mean():.3f}  ← sahte başarı")

# ---------------------------------------------------------------------------
# DOĞRU: Seçim Pipeline içinde (her katmanda sadece eğitim kısmından öğrenilir)
# ---------------------------------------------------------------------------
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
dogru = cross_val_score(pipe, X, y, cv=cv)
print(f"DOĞRU yöntem (Pipeline içinde)    : doğruluk = {dogru.mean():.3f}  ← gerçek başarı ≈ şans")
print("\nAynı kural Normalize, Standardize, PCA, Discretize, SMOTE gibi TÜM veri-bağımlı ön işlemler için geçerlidir.")
print("Weka'da karşılığı: FilteredClassifier / AttributeSelectedClassifier kullanmak.")
