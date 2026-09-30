# -*- coding: utf-8 -*-
"""
05 - Model Değerlendirme Yöntemleri
===================================
Ders notu bölümü: "8. Model Değerlendirme Yöntemleri: Modelimiz Gerçekten Öğrendi mi?"

Aynı model (karar ağacı) ve aynı veri (meme kanseri teşhis verisi) ile:
  1) Eğitim verisinde test etmenin yanıltıcılığı (ezber / overfitting)
  2) Holdout: farklı rastgele bölmelerde skorun ne kadar oynadığı
  3) Eğitim / Doğrulama / Test (üçlü ayırma)
  4) K-katlı ve tabakalı K-katlı çapraz doğrulama
  5) Birini dışarıda bırak (LOOCV)
  6) Bootstrap ve torba dışı (OOB) örneklerin ~%36.8 olduğu

Çalıştırma:  python 05_model_degerlendirme_yontemleri.py
"""

import numpy as np                                            # Sayısal işlemler
from sklearn.datasets import load_breast_cancer               # 569 hasta, 30 öznitelik, 2 sınıf
from sklearn.tree import DecisionTreeClassifier               # Değerlendireceğimiz model
from sklearn.model_selection import (
    train_test_split,                                         # Holdout bölme
    KFold, StratifiedKFold, LeaveOneOut,                      # Çapraz doğrulama stratejileri
    cross_val_score,                                          # CV skorlarını tek satırda hesaplar
)
from sklearn.metrics import accuracy_score                    # Doğruluk ölçütü

X, y = load_breast_cancer(return_X_y=True)                    # X: öznitelikler, y: 0=kötü huylu, 1=iyi huylu
model = DecisionTreeClassifier(random_state=0)                # Sınırlandırılmamış ağaç: ezberlemeye çok yatkın

# ---------------------------------------------------------------------------
# 1) Eğitim verisinde test etmek (YANLIŞ yöntem)
# ---------------------------------------------------------------------------
model.fit(X, y)                                               # Tüm veriyle eğit
print("=== 1) Eğitim verisiyle test ===")
print(f"Eğitim doğruluğu: {accuracy_score(y, model.predict(X)):.3f}  ← %100: model veriyi ezberledi!")
print()

# ---------------------------------------------------------------------------
# 2) Holdout: bölme şansa bağlıdır
# ---------------------------------------------------------------------------
print("=== 2) Holdout (%70 eğitim / %30 test), 10 farklı rastgele bölme ===")
holdout_skorlari = []
for tohum in range(10):                                       # random_state (seed) her seferinde farklı
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=tohum, stratify=y)
    model.fit(X_tr, y_tr)                                     # Sadece eğitim kısmıyla eğit
    holdout_skorlari.append(accuracy_score(y_te, model.predict(X_te)))   # Görülmemiş test kısmında ölç
print("Skorlar:", np.round(holdout_skorlari, 3))
print(f"En düşük {min(holdout_skorlari):.3f}, en yüksek {max(holdout_skorlari):.3f}  ← tek bölmeye güvenmek risklidir")
print()

# ---------------------------------------------------------------------------
# 3) Üçlü ayırma: Eğitim %60 / Doğrulama %20 / Test %20
# ---------------------------------------------------------------------------
X_gecici, X_test, y_gecici, y_test = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y)
X_egitim, X_dogr, y_egitim, y_dogr = train_test_split(X_gecici, y_gecici, test_size=0.25,   # 0.25 * 0.80 = 0.20
                                                      random_state=1, stratify=y_gecici)
print("=== 3) Üçlü ayırma ile hiperparametre (max_depth) seçimi ===")
en_iyi_derinlik, en_iyi_skor = None, -1
for derinlik in [1, 2, 3, 4, 5, 8, None]:                     # Aday hiperparametre değerleri
    m = DecisionTreeClassifier(max_depth=derinlik, random_state=0).fit(X_egitim, y_egitim)
    skor = accuracy_score(y_dogr, m.predict(X_dogr))          # Seçimi DOĞRULAMA setine göre yap
    print(f"max_depth={str(derinlik):<5} → doğrulama doğruluğu {skor:.3f}")
    if skor > en_iyi_skor:
        en_iyi_derinlik, en_iyi_skor = derinlik, skor
son_model = DecisionTreeClassifier(max_depth=en_iyi_derinlik, random_state=0).fit(X_gecici, y_gecici)  # Eğitim+doğrulama ile yeniden eğit
print(f"Seçilen max_depth={en_iyi_derinlik}; TEST doğruluğu (tek sefer): {accuracy_score(y_test, son_model.predict(X_test)):.3f}")
print()

# ---------------------------------------------------------------------------
# 4) K-katlı ve tabakalı K-katlı çapraz doğrulama
# ---------------------------------------------------------------------------
print("=== 4) Çapraz doğrulama ===")
for k in [2, 5, 10]:
    kf = KFold(n_splits=k, shuffle=True, random_state=0)                  # Sınıf oranlarını gözetmez
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)       # Her katmanda sınıf oranı korunur
    s1 = cross_val_score(model, X, y, cv=kf)
    s2 = cross_val_score(model, X, y, cv=skf)
    print(f"K={k:<2}  KFold: {s1.mean():.3f} ± {s1.std():.3f}   StratifiedKFold: {s2.mean():.3f} ± {s2.std():.3f}")
print()

# ---------------------------------------------------------------------------
# 5) LOOCV (N = 569 kez eğitim!) — küçük veride kullanılır
# ---------------------------------------------------------------------------
kucuk_X, kucuk_y = X[:100], y[:100]                           # Süreyi kısaltmak için ilk 100 örnek
loo_skor = cross_val_score(model, kucuk_X, kucuk_y, cv=LeaveOneOut())   # Her skor 0 ya da 1 olur
print("=== 5) LOOCV (ilk 100 örnek) ===")
print(f"{len(loo_skor)} model eğitildi, ortalama doğruluk = {loo_skor.mean():.3f}")
print()

# ---------------------------------------------------------------------------
# 6) BOOTSTRAP ve OOB
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)                               # Tekrarlanabilir rastgele sayı üreteci
n = len(X)
oob_oranlari, oob_skorlari = [], []
for _ in range(200):                                          # 200 bootstrap örneklemi
    idx = rng.integers(0, n, size=n)                          # Yerine koyarak n adet indeks çek
    oob = np.setdiff1d(np.arange(n), idx)                     # Hiç seçilmeyenler = torba dışı (OOB)
    oob_oranlari.append(len(oob) / n)
    m = DecisionTreeClassifier(random_state=0).fit(X[idx], y[idx])       # Bootstrap örneğiyle eğit
    oob_skorlari.append(accuracy_score(y[oob], m.predict(X[oob])))       # OOB örneklerde test et
print("=== 6) Bootstrap ===")
print(f"Ortalama OOB oranı       : {np.mean(oob_oranlari):.3f}  (teori: (1-1/n)^n ≈ 1/e ≈ 0.368)")
print(f"Ortalama OOB doğruluğu   : {np.mean(oob_skorlari):.3f} ± {np.std(oob_skorlari):.3f}")
