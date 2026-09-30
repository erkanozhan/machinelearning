# -*- coding: utf-8 -*-
"""
09 - Destek Vektör Makineleri (SVM)
===================================
Ders notu bölümü: "10. Destek Vektör Makineleri"

  1) Doğrusal ayrılabilir veride destek vektörlerini ve marjı bulur
  2) C parametresinin (yumuşak marj) etkisini gösterir
  3) İç içe halkalar (circles) verisinde lineer çekirdek başarısız, RBF başarılı olur
  4) Iris üzerinde ölçekleme + SVM Pipeline'ı ve küçük bir C/gamma araması yapar
  5) Karar sınırlarını çizer

Çalıştırma:  python 09_svm.py
"""

import numpy as np                                            # Sayısal işlemler
import matplotlib.pyplot as plt                               # Grafikler
from sklearn.datasets import make_blobs, make_circles, load_iris   # Veri üreticiler / veri seti
from sklearn.svm import SVC                                   # Destek vektör sınıflandırıcı
from sklearn.pipeline import make_pipeline                    # Ölçekleme + model zinciri
from sklearn.preprocessing import StandardScaler              # SVM mesafeye duyarlıdır → ölçekleme şart
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold

# ---------------------------------------------------------------------------
# 1) Doğrusal ayrılabilir veri: destek vektörleri ve marj
# ---------------------------------------------------------------------------
X, y = make_blobs(n_samples=40, centers=2, cluster_std=1.0, random_state=6)   # İki ayrık küme
svm = SVC(kernel="linear", C=1000).fit(X, y)                  # Çok büyük C ≈ sert marj (hiç hata kabul etme)
w, b = svm.coef_[0], svm.intercept_[0]                        # Hiperdüzlem: w·x + b = 0
print("=== 1) Doğrusal SVM ===")
print(f"w = {np.round(w, 3)}, b = {b:.3f}")
print(f"Destek vektörü sayısı: {len(svm.support_vectors_)} / {len(X)}  ← sınırı sadece bunlar belirler")
print(f"Marj genişliği 2/||w|| = {2 / np.linalg.norm(w):.3f}")
print()

# ---------------------------------------------------------------------------
# 2) C'nin etkisi (üst üste binen veri)
# ---------------------------------------------------------------------------
X2, y2 = make_blobs(n_samples=200, centers=2, cluster_std=2.2, random_state=3)   # Sınıflar iç içe geçiyor
print("=== 2) C parametresi (yumuşak marj) ===")
for C in [0.01, 0.1, 1, 10, 100]:
    m = SVC(kernel="linear", C=C).fit(X2, y2)
    cv_skor = cross_val_score(SVC(kernel="linear", C=C), X2, y2, cv=5).mean()
    print(f"C={C:<6}: destek vektörü={len(m.support_):>3}, marj={2/np.linalg.norm(m.coef_):.2f}, CV doğruluğu={cv_skor:.3f}")
print("→ Küçük C: geniş marj, çok destek vektörü (hataya toleranslı). Büyük C: dar marj, eğitim verisine sıkı uyum.")
print()

# ---------------------------------------------------------------------------
# 3) Çekirdek hilesi: iç içe halkalar
# ---------------------------------------------------------------------------
X3, y3 = make_circles(n_samples=300, factor=0.4, noise=0.08, random_state=0)   # İç halka ve dış halka
print("=== 3) Doğrusal olmayan veri (iç içe halkalar) ===")
for kernel in ["linear", "poly", "rbf"]:
    s = cross_val_score(SVC(kernel=kernel, degree=2, gamma="scale"), X3, y3, cv=5).mean()
    print(f"kernel={kernel:<7}: CV doğruluğu = {s:.3f}")
print()

# ---------------------------------------------------------------------------
# 4) Iris: Pipeline + küçük bir hiperparametre araması
# ---------------------------------------------------------------------------
Xi, yi = load_iris(return_X_y=True)
pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))     # Adımların adları: 'standardscaler', 'svc'
izgara = {"svc__C": [0.1, 1, 10, 100],                        # 'adımadı__parametre' söz dizimi
          "svc__gamma": [0.01, 0.1, 1, "scale"]}
arama = GridSearchCV(pipe, izgara, cv=StratifiedKFold(5, shuffle=True, random_state=0)).fit(Xi, yi)
print("=== 4) Iris + RBF SVM ===")
print(f"En iyi parametreler: {arama.best_params_}")
print(f"En iyi CV doğruluğu: {arama.best_score_:.3f}")
print()

# ---------------------------------------------------------------------------
# 5) Karar sınırlarını çiz
# ---------------------------------------------------------------------------
def sinir_ciz(ax, model, X, y, baslik):
    """2 boyutlu veride modelin karar sınırını ve (varsa) marj çizgilerini çizer."""
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300))   # Izgara noktaları
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)   # w·x+b değeri
    ax.contourf(xx, yy, Z > 0, alpha=0.15, cmap="coolwarm")                          # Bölgeleri boya
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=["--", "-", "--"], colors="k")  # Marj ve hiperdüzlem
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=20, edgecolors="k")
    if hasattr(model, "support_vectors_"):
        ax.scatter(*model.support_vectors_.T, s=120, facecolors="none", edgecolors="k", label="Destek vektörleri")
    ax.set_title(baslik)


fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
sinir_ciz(ax[0], svm, X, y, "Doğrusal SVM (sert marj)")
sinir_ciz(ax[1], SVC(kernel="linear").fit(X3, y3), X3, y3, "Halkalar – lineer çekirdek")
sinir_ciz(ax[2], SVC(kernel="rbf").fit(X3, y3), X3, y3, "Halkalar – RBF çekirdek")
ax[0].legend(loc="lower left")
plt.tight_layout()
plt.show()
