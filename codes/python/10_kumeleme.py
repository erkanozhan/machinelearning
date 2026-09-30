# -*- coding: utf-8 -*-
"""
10 - Kümeleme: K-Means, Hiyerarşik Kümeleme ve DBSCAN
=====================================================
Ders notu bölümü: "15. Kümeleme"

  1) K-Means'i sıfırdan (NumPy ile) yazar ve adımlarını yazdırır
  2) Iris üzerinde dirsek (elbow) yöntemi ve siluet skoru ile K seçimi
  3) Bulunan kümeleri gerçek türlerle karşılaştırır (Weka "Classes to clusters" benzeri)
  4) Hiyerarşik kümelemede dendrogram çizer
  5) Ay (moons) şeklindeki veride K-Means ile DBSCAN'i karşılaştırır

Çalıştırma:  python 10_kumeleme.py
"""

import numpy as np                                            # Sayısal işlemler
import matplotlib.pyplot as plt                               # Grafikler
from sklearn.datasets import load_iris, make_moons            # Veri setleri
from sklearn.preprocessing import StandardScaler, MinMaxScaler   # Ölçekleme (kümeleme mesafeye dayanır!)
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN   # Kümeleme algoritmaları
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix   # Değerlendirme
from scipy.cluster.hierarchy import linkage, dendrogram       # Dendrogram çizimi için

# ---------------------------------------------------------------------------
# 1) K-MEANS'İ SIFIRDAN YAZMAK
# ---------------------------------------------------------------------------
def kmeans_elle(X, K, max_iter=100, tohum=0):
    """Basit K-Means: rastgele merkez → atama → güncelleme → tekrar."""
    rng = np.random.default_rng(tohum)                                        # Tekrarlanabilir rastgelelik
    merkezler = X[rng.choice(len(X), size=K, replace=False)]                  # 1. K rastgele noktayı merkez seç
    for adim in range(max_iter):
        mesafeler = np.linalg.norm(X[:, None, :] - merkezler[None, :, :], axis=2)   # Her nokta-merkez Öklid mesafesi (n x K)
        etiketler = mesafeler.argmin(axis=1)                                  # 2. Atama: en yakın merkez
        yeni = np.array([X[etiketler == k].mean(axis=0) for k in range(K)])   # 3. Güncelleme: küme ortalaması
        sse = ((X - yeni[etiketler]) ** 2).sum()                              # Küme içi kareler toplamı (WCSS)
        print(f"  adım {adim + 1}: SSE = {sse:.3f}")
        if np.allclose(yeni, merkezler):                                      # 4. Merkezler değişmediyse dur
            break
        merkezler = yeni
    return etiketler, merkezler, sse


iris = load_iris()
X = MinMaxScaler().fit_transform(iris.data)                   # Weka SimpleKMeans de öznitelikleri 0-1'e normalize eder
print("=== 1) Elle yazılmış K-Means (K=3) ===")
etiket_elle, merkez_elle, sse_elle = kmeans_elle(X, K=3)
print()

# ---------------------------------------------------------------------------
# 2) K SEÇİMİ: Dirsek yöntemi ve siluet skoru
# ---------------------------------------------------------------------------
print("=== 2) K seçimi ===")
print(f"{'K':>2} {'SSE (inertia)':>14} {'Siluet':>8}")
sse_listesi = []
for K in range(1, 9):
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X)             # n_init=10: 10 farklı başlangıç, en iyisi
    sse_listesi.append(km.inertia_)                                         # inertia_ = SSE
    sil = silhouette_score(X, km.labels_) if K > 1 else float("nan")        # Siluet K=1 için tanımsız
    print(f"{K:>2} {km.inertia_:>14.3f} {sil:>8.3f}")
print("→ SSE her zaman azalır; 'dirsek' K=2-3 civarında. Siluet K=2'de en yüksek (setosa çok ayrık).")
print()

# ---------------------------------------------------------------------------
# 3) Kümeler gerçek sınıflarla ne kadar örtüşüyor?
# ---------------------------------------------------------------------------
km3 = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
print("=== 3) K=3 kümeler vs gerçek türler ===")
print("Satır: gerçek tür, sütun: küme")
print(confusion_matrix(iris.target, km3.labels_))
print(f"Adjusted Rand Index = {adjusted_rand_score(iris.target, km3.labels_):.3f}  (1 = mükemmel örtüşme, 0 = rastgele)")
print()

# ---------------------------------------------------------------------------
# 4) HİYERARŞİK KÜMELEME
# ---------------------------------------------------------------------------
Z = linkage(X, method="ward")                                 # Ward: birleşince SSE'yi en az artıran iki kümeyi birleştir
agg = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(X)
print("=== 4) Hiyerarşik (Ward) K=3 ===")
print(confusion_matrix(iris.target, agg.labels_))
print()

# ---------------------------------------------------------------------------
# 5) DBSCAN vs K-Means (ay şeklindeki kümeler)
# ---------------------------------------------------------------------------
Xm, ym = make_moons(n_samples=300, noise=0.06, random_state=0)   # İki hilal: küresel olmayan kümeler
Xm = StandardScaler().fit_transform(Xm)
km_m = KMeans(n_clusters=2, n_init=10, random_state=0).fit(Xm)
db_m = DBSCAN(eps=0.3, min_samples=5).fit(Xm)                 # eps: komşuluk yarıçapı, min_samples: minPts
print("=== 5) Ay şeklindeki veri ===")
print(f"K-Means ARI = {adjusted_rand_score(ym, km_m.labels_):.3f}")
print(f"DBSCAN  ARI = {adjusted_rand_score(ym, db_m.labels_):.3f}, bulunan küme = {len(set(db_m.labels_) - {-1})}, "
      f"gürültü noktası = {(db_m.labels_ == -1).sum()}")

# --- Grafikler ---
fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
ax[0].plot(range(1, 9), sse_listesi, "o-")
ax[0].set_xlabel("K")
ax[0].set_ylabel("SSE")
ax[0].set_title("Dirsek yöntemi (Iris)")
dendrogram(Z, ax=ax[1], truncate_mode="lastp", p=20, no_labels=True)   # Son 20 birleşmeyi göster
ax[1].set_title("Dendrogram (Ward)")
ax[2].scatter(Xm[:, 0], Xm[:, 1], c=km_m.labels_, cmap="coolwarm", s=12)
ax[2].set_title("K-Means (K=2) – başarısız")
ax[3].scatter(Xm[:, 0], Xm[:, 1], c=db_m.labels_, cmap="coolwarm", s=12)
ax[3].set_title("DBSCAN – başarılı")
plt.tight_layout()
plt.show()
