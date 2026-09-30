# -*- coding: utf-8 -*-
"""
08 - Topluluk Öğrenmesi (Ensemble Learning): Bagging, Boosting, Stacking
=========================================================================
Ders notu bölümü: "14. Topluluk Öğrenmesi"

Bölüm A (sınıflandırma): Iris ve meme kanseri veri setlerinde tek karar ağacı ile
                         Bagging, Random Forest, AdaBoost, Gradient Boosting ve Stacking'i
                         AYNI 10 katlı çapraz doğrulama bölmeleriyle karşılaştırır.
Bölüm B (regresyon):     Diyabet veri setinde aynı karşılaştırmayı MAE/RMSE/R² ile yapar.
Bölüm C:                 Random Forest'ın OOB (torba dışı) skorunu gösterir.

Neden tek bir train/test bölmesi yerine CV?  Iris gibi küçük veri setlerinde 45 örneklik
tek bir test setinde modeller arasındaki fark çoğu zaman 1-2 örnekten ibarettir; CV daha güvenilirdir.

Çalıştırma:  python 08_topluluk_ogrenmesi.py
"""

import numpy as np                                            # Sayısal işlemler
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes   # Veri setleri
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_validate   # CV araçları
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor      # Temel (base) öğreniciler
from sklearn.ensemble import (
    BaggingClassifier,                                        # Genel bagging: istediğin modeli sarmalar
    RandomForestClassifier, RandomForestRegressor,            # Bagging + rastgele öznitelik seçimi
    AdaBoostClassifier,                                       # Uyarlamalı boosting (örnek ağırlıklarını günceller)
    GradientBoostingClassifier, GradientBoostingRegressor,    # Artıklar (residual) üzerine ardışık ağaçlar
    StackingClassifier, StackingRegressor,                    # Meta-model ile birleştirme
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge   # Meta-model / temel model adayları
from sklearn.svm import SVC                                   # Destek vektör sınıflandırıcı
from sklearn.naive_bayes import GaussianNB                    # Naive Bayes
from sklearn.pipeline import make_pipeline                    # Ölçekleme + model zinciri
from sklearn.preprocessing import StandardScaler              # Z-skoru ölçekleme

# ===========================================================================
# BÖLÜM A — SINIFLANDIRMA
# ===========================================================================
def siniflandirma_modelleri():
    """Karşılaştırılacak modelleri her veri seti için sıfırdan oluşturur."""
    return {
        "Tek Karar Ağacı": DecisionTreeClassifier(random_state=42),
        "Bagging (10 ağaç)": BaggingClassifier(
            estimator=DecisionTreeClassifier(),               # Not: eski sürümlerde parametre adı 'base_estimator' idi
            n_estimators=10, random_state=42),
        "Random Forest (100)": RandomForestClassifier(n_estimators=100, random_state=42),
        "AdaBoost (kütük)": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),    # Karar kütüğü (decision stump): tek soruluk zayıf öğrenici
            n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Stacking": StackingClassifier(
            estimators=[                                      # Seviye-0: birbirinden FARKLI temel modeller
                ("svc", make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))),
                ("nb", GaussianNB()),
                ("dt", DecisionTreeClassifier(max_depth=4, random_state=42)),
            ],
            final_estimator=LogisticRegression(max_iter=1000),  # Seviye-1: meta-model
            cv=5),                                            # Temel modellerin tahminleri 'out-of-fold' üretilir → sızıntı yok
    }


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)   # Tüm modeller için aynı bölmeler

for veri_adi, yukleyici in [("Iris", load_iris), ("Meme Kanseri", load_breast_cancer)]:
    X, y = yukleyici(return_X_y=True)                         # Veriyi yükle
    print(f"=== A) {veri_adi} — 10 katlı CV doğruluğu ===")
    for ad, model in siniflandirma_modelleri().items():
        s = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)   # n_jobs=-1: tüm çekirdekleri kullan
        print(f"{ad:<22}: {s.mean():.3f} ± {s.std():.3f}")
    print()

# ===========================================================================
# BÖLÜM B — REGRESYON
# ===========================================================================
X, y = load_diabetes(return_X_y=True)                         # 442 hasta, 10 öznitelik, hedef: 1 yıl sonraki hastalık ilerlemesi
cv_r = KFold(n_splits=10, shuffle=True, random_state=42)      # Regresyonda tabakalama yapılmaz
reg_modeller = {
    "Tek Karar Ağacı": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Lineer Regresyon": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42),
    "Stacking": StackingRegressor(
        estimators=[("ridge", Ridge()),
                    ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
                    ("dt", DecisionTreeRegressor(max_depth=5, random_state=42))],
        final_estimator=LinearRegression(), cv=5),
}
print("=== B) Diyabet (regresyon) — 10 katlı CV ===")
print(f"{'Model':<20} {'MAE':>7} {'RMSE':>7} {'R²':>6}")
for ad, model in reg_modeller.items():
    sonuc = cross_validate(model, X, y, cv=cv_r, n_jobs=-1,
                           scoring=("neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"))
    # sklearn "büyük daha iyi" kuralı için hata metriklerini negatif döndürür → başına eksi koyuyoruz
    print(f"{ad:<20} {-sonuc['test_neg_mean_absolute_error'].mean():>7.2f} "
          f"{-sonuc['test_neg_root_mean_squared_error'].mean():>7.2f} {sonuc['test_r2'].mean():>6.3f}")
print()

# ===========================================================================
# BÖLÜM C — OOB skoru: ayrı test seti olmadan tahmini genelleme başarısı
# ===========================================================================
X, y = load_breast_cancer(return_X_y=True)
rf = RandomForestClassifier(n_estimators=300, oob_score=True, random_state=42).fit(X, y)   # oob_score=True
print("=== C) Random Forest OOB ===")
print(f"OOB doğruluğu: {rf.oob_score_:.3f}  (her ağaç, eğitiminde görmediği ~%37'lik kısımda test edildi)")
