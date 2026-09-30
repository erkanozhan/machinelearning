# -*- coding: utf-8 -*-
"""
13 - Dengesiz Veri ve Maliyete Duyarlı Öğrenme
==============================================
Ders notu bölümü: "18. Dengesiz Veri ve Maliyete Duyarlı Öğrenme"

Pozitif sınıf (1) nadir ve kaçırılması PAHALI olan durumdur (ör. dolandırıcılık, hastalık).
  C_FP = 1  : Gerçekte negatif olanı pozitif demek (gereksiz alarm)
  C_FN = 10 : Gerçekte pozitif olanı kaçırmak (çok pahalı)

Karşılaştırılan yaklaşımlar:
  0) Standart lojistik regresyon (tüm hatalar eşit)
  1) Sınıf ağırlıklandırma (class_weight) — algoritma seviyesi
  2) Rastgele aşırı örnekleme (oversampling) — veri seviyesi (sadece eğitim setine!)
  3) Karar eşiğini maliyete göre ayarlama (threshold moving)
     - Teorik eşik: t* = C_FP / (C_FP + C_FN)
     - Eşik, TEST setine göre değil DOĞRULAMA setine göre seçilir (aksi halde sızıntı olur)

Çalıştırma:  python 13_maliyete_duyarli_ogrenme.py
"""

import numpy as np                                            # Sayısal işlemler
from sklearn.datasets import make_classification              # Dengesiz yapay veri
from sklearn.model_selection import train_test_split          # Bölme
from sklearn.linear_model import LogisticRegression           # Temel model
from sklearn.metrics import confusion_matrix, recall_score, precision_score   # Değerlendirme

# ---------------------------------------------------------------------------
# 1) VERİ: %95 negatif, %5 pozitif
# ---------------------------------------------------------------------------
X, y = make_classification(n_samples=6000, n_features=20, n_informative=4, n_redundant=2,
                           weights=[0.95, 0.05], flip_y=0.01, class_sep=1.0, random_state=42)
# Eğitim %60, doğrulama %20, test %20 (tabakalı)
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=42)

C_FP, C_FN = 1.0, 10.0                               # Hata maliyetleri


def maliyet_raporu(ad, y_gercek, y_tahmin):
    """Karışıklık matrisini, toplam maliyeti ve recall/precision'ı yazdırır."""
    TN, FP, FN, TP = confusion_matrix(y_gercek, y_tahmin, labels=[0, 1]).ravel()   # 2x2 matrisi düzleştir
    toplam = FP * C_FP + FN * C_FN                   # Toplam maliyet = FP·C_FP + FN·C_FN
    print(f"{ad:<34} TN={TN:>4} FP={FP:>4} FN={FN:>3} TP={TP:>3} | "
          f"recall={recall_score(y_gercek, y_tahmin):.2f} precision={precision_score(y_gercek, y_tahmin, zero_division=0):.2f} | "
          f"MALİYET={toplam:>6.0f}")
    return toplam


print(f"Eğitimdeki pozitif oranı: {y_tr.mean():.3f}\n")
print("=== Test seti sonuçları ===")

# 0) Standart model
m0 = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
maliyet_raporu("0) Standart (eşik 0.5)", y_test, m0.predict(X_test))

# 1) Sınıf ağırlıklandırma: pozitif sınıfın hatası kayıp fonksiyonunda 10 kat sayılır
m1 = LogisticRegression(max_iter=2000, class_weight={0: C_FP, 1: C_FN}).fit(X_tr, y_tr)
maliyet_raporu("1) class_weight={0:1, 1:10}", y_test, m1.predict(X_test))

m1b = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_tr, y_tr)   # Ağırlık = n / (2·n_sınıf)
maliyet_raporu("1b) class_weight='balanced'", y_test, m1b.predict(X_test))

# 2) Rastgele aşırı örnekleme: pozitif örnekleri kopyalayarak dengele (SADECE eğitim setinde)
rng = np.random.default_rng(0)
pos = np.flatnonzero(y_tr == 1)                      # Pozitif örneklerin indeksleri
ek = rng.choice(pos, size=(y_tr == 0).sum() - len(pos), replace=True)   # Eksik kalan kadar yerine koyarak kopya çek
X_os, y_os = np.vstack([X_tr, X_tr[ek]]), np.concatenate([y_tr, y_tr[ek]])
m2 = LogisticRegression(max_iter=2000).fit(X_os, y_os)
maliyet_raporu("2) Oversampling (eğitimde)", y_test, m2.predict(X_test))
# (Daha gelişmiş sürüm: SMOTE → pip install imbalanced-learn)

# 3) Eşik ayarlama
t_teorik = C_FP / (C_FP + C_FN)                      # Olasılıklar iyi kalibre edilmişse optimum eşik
p_test = m0.predict_proba(X_test)[:, 1]              # Standart modelin pozitif olasılıkları
maliyet_raporu(f"3a) Teorik eşik t*={t_teorik:.3f}", y_test, (p_test >= t_teorik).astype(int))

# Eşiği doğrulama setinde ara, testte sadece bir kez uygula
p_val = m0.predict_proba(X_val)[:, 1]
esikler = np.linspace(0.02, 0.9, 45)
val_maliyet = []
for t in esikler:
    TN, FP, FN, TP = confusion_matrix(y_val, (p_val >= t).astype(int), labels=[0, 1]).ravel()
    val_maliyet.append(FP * C_FP + FN * C_FN)
t_best = esikler[int(np.argmin(val_maliyet))]
maliyet_raporu(f"3b) Doğrulamada seçilen eşik={t_best:.2f}", y_test, (p_test >= t_best).astype(int))
print("\n→ Doğruluk (accuracy) düşse bile toplam maliyet belirgin şekilde azalır.")
