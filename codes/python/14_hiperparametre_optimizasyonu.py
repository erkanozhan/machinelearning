# -*- coding: utf-8 -*-
"""
14 - Hiperparametre Optimizasyonu: Grid Search, Random Search, Nested CV
=========================================================================
Ders notu bölümü: "20. Optimizasyon ve Hiperparametre Ayarlama"

  1) Pipeline + GridSearchCV (ölçekleme her katmanda yeniden fit edilir → sızıntı yok)
  2) RandomizedSearchCV ile aynı bütçeyi rastgele dağıtmak
  3) Nested (iç içe) çapraz doğrulama ile tarafsız performans tahmini
  4) "GridSearch'ün best_score_ değerini raporlamak" ile nested skor arasındaki fark
  5) Early stopping (erken durdurma) örneği

Çalıştırma:  python 14_hiperparametre_optimizasyonu.py   (1-2 dakika sürebilir)
"""

import numpy as np                                            # Sayısal işlemler
from scipy.stats import loguniform                            # Log-ölçekli sürekli dağılım (C gibi parametreler için)
from sklearn.datasets import load_breast_cancer               # 569 örnek, 30 öznitelik
from sklearn.pipeline import Pipeline                         # Adımları isimli zincir
from sklearn.preprocessing import StandardScaler              # Ölçekleme
from sklearn.svm import SVC                                   # Ayarlanacak model
from sklearn.neural_network import MLPClassifier              # Early stopping örneği için
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score, train_test_split)

X, y = load_breast_cancer(return_X_y=True)

pipe = Pipeline([                                             # Adım adları parametre adlarında ön ek olur
    ("olcek", StandardScaler()),
    ("svm", SVC(kernel="rbf")),
])

ic_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)    # İç döngü (parametre seçimi)
dis_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)  # Dış döngü (performans ölçümü)

# ---------------------------------------------------------------------------
# 1) GRID SEARCH: 4 x 4 = 16 kombinasyon x 5 katman = 80 eğitim
# ---------------------------------------------------------------------------
izgara = {"svm__C": [0.1, 1, 10, 100],                        # 'adımadı__parametre'
          "svm__gamma": [0.001, 0.01, 0.1, 1]}
grid = GridSearchCV(pipe, izgara, cv=ic_cv, scoring="accuracy", n_jobs=-1).fit(X, y)
print("=== 1) Grid Search ===")
print(f"En iyi parametreler : {grid.best_params_}")
print(f"best_score_ (iç CV) : {grid.best_score_:.4f}  ← seçim için kullanıldı, raporlamak için İYİMSER")
print()

# ---------------------------------------------------------------------------
# 2) RANDOM SEARCH: aynı bütçe (16 deneme) ama sürekli dağılımlardan
# ---------------------------------------------------------------------------
dagilim = {"svm__C": loguniform(1e-2, 1e3),                   # 0.01 ile 1000 arası, log-ölçekte eşit olasılık
           "svm__gamma": loguniform(1e-4, 1e1)}
rand = RandomizedSearchCV(pipe, dagilim, n_iter=16, cv=ic_cv, scoring="accuracy",
                          random_state=0, n_jobs=-1).fit(X, y)
print("=== 2) Random Search ===")
print(f"En iyi parametreler : { {k: round(float(v), 4) for k, v in rand.best_params_.items()} }")
print(f"best_score_         : {rand.best_score_:.4f}")
print()

# ---------------------------------------------------------------------------
# 3) NESTED CV: GridSearchCV nesnesini bir "model" gibi dış CV'ye veriyoruz
# ---------------------------------------------------------------------------
nested = cross_val_score(GridSearchCV(pipe, izgara, cv=ic_cv, scoring="accuracy"),
                         X, y, cv=dis_cv, scoring="accuracy", n_jobs=-1)
print("=== 3) Nested CV (10 dış x 5 iç katman) ===")
print(f"Dış katman skorları : {np.round(nested, 3)}")
print(f"Tarafsız tahmin     : {nested.mean():.4f} ± {nested.std():.4f}")
print()

# ---------------------------------------------------------------------------
# 4) Early stopping: doğrulama skoru iyileşmeyi bırakınca eğitimi durdur
#    Beklenen: çok daha az epoch, eğitim skoru %100 değil (ezber azalır), test skoru aynı ya da daha iyi
# ---------------------------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
sc = StandardScaler().fit(X_tr)
for es in [False, True]:
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=3000, early_stopping=es,   # es=True → eğitimin %15'i doğrulamaya ayrılır
                        validation_fraction=0.15, n_iter_no_change=10,               # 10 epoch iyileşme yoksa dur
                        alpha=1e-6, learning_rate_init=0.01, random_state=0)
    mlp.fit(sc.transform(X_tr), y_tr)
    print(f"early_stopping={es!s:<5}: {mlp.n_iter_:>4} epoch, "
          f"eğitim={mlp.score(sc.transform(X_tr), y_tr):.3f}, test={mlp.score(sc.transform(X_te), y_te):.3f}")
