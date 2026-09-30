# -*- coding: utf-8 -*-
"""
02 - Lineer Regresyon: En Küçük Kareler ve Gradyan İnişi
========================================================
Ders notu bölümü: "5. Lineer Regresyon"

Bu betik aynı problemi üç farklı yolla çözer ve sonuçların aynı çıktığını gösterir:
  1) Kapalı form (en küçük kareler) formülü ile elle çözüm
  2) Gradyan inişi (gradient descent) ile adım adım öğrenme
  3) scikit-learn LinearRegression ile tek satırda çözüm

Örnek veri: Ders çalışma süresi (saat) → Sınav notu

Çalıştırma:  python 02_lineer_regresyon.py
"""

import numpy as np                                   # Sayısal hesaplamalar
import matplotlib.pyplot as plt                      # Grafik çizimi
from sklearn.linear_model import LinearRegression    # Hazır lineer regresyon modeli

# ---------------------------------------------------------------------------
# 0) VERİ
# ---------------------------------------------------------------------------
x = np.array([1, 2, 3, 4, 5], dtype=float)           # Bağımsız değişken: çalışma süresi (saat)
y = np.array([52, 58, 65, 70, 80], dtype=float)      # Bağımlı değişken: sınav notu
n = len(x)                                           # Örnek sayısı (m veya n ile gösterilir)

# ---------------------------------------------------------------------------
# 1) KAPALI FORM ÇÖZÜM (En Küçük Kareler)
#    θ1 = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²
#    θ0 = ȳ - θ1·x̄
# ---------------------------------------------------------------------------
x_ort, y_ort = x.mean(), y.mean()                    # x̄ ve ȳ (ortalamalar)
theta1 = ((x - x_ort) * (y - y_ort)).sum() / ((x - x_ort) ** 2).sum()   # Eğim
theta0 = y_ort - theta1 * x_ort                      # Kesişim (sabit terim)

print("=== 1) Kapalı form (en küçük kareler) ===")
print(f"θ0 (kesişim) = {theta0:.3f}")
print(f"θ1 (eğim)    = {theta1:.3f}")
print(f"Model: ŷ = {theta0:.2f} + {theta1:.2f}·x")
print(f"6 saat çalışan öğrencinin tahmini notu: {theta0 + theta1 * 6:.1f}")
print()

# ---------------------------------------------------------------------------
# 2) GRADYAN İNİŞİ
#    Maliyet: J(θ) = (1/2m) Σ (ŷ - y)²
#    Güncelleme: θj := θj - α · ∂J/∂θj
# ---------------------------------------------------------------------------
def maliyet(t0, t1):
    """Verilen θ0, θ1 için ortalama karesel hatanın yarısını (J) döndürür."""
    tahmin = t0 + t1 * x                             # Modelin tahminleri (ŷ)
    return ((tahmin - y) ** 2).sum() / (2 * n)       # J(θ) = 1/(2m) · Σ(ŷ - y)²

t0, t1 = 0.0, 0.0                                    # Parametrelere başlangıç değeri ver (genelde 0 veya rastgele)
alfa = 0.05                                          # Öğrenme oranı (learning rate, α veya η)
adim_sayisi = 5000                                   # Kaç kez güncelleme yapılacağı (iterasyon/epoch)
gecmis = []                                          # Her adımdaki maliyeti saklayacağımız liste

for adim in range(adim_sayisi):                      # Belirlenen sayıda tekrar et
    tahmin = t0 + t1 * x                             # 1. İleri adım: mevcut θ ile tahmin yap
    hata = tahmin - y                                # 2. Hata: ŷ - y
    grad_t0 = hata.sum() / n                         # 3. ∂J/∂θ0 = (1/m) Σ (ŷ - y)
    grad_t1 = (hata * x).sum() / n                   #    ∂J/∂θ1 = (1/m) Σ (ŷ - y)·x
    t0 -= alfa * grad_t0                             # 4. Eğimin tersi yönünde adım at
    t1 -= alfa * grad_t1
    gecmis.append(maliyet(t0, t1))                   # 5. Maliyeti kaydet (grafik için)

print("=== 2) Gradyan inişi ===")
print(f"α = {alfa}, adım = {adim_sayisi}")
print(f"θ0 = {t0:.3f}, θ1 = {t1:.3f}   (kapalı formla aynı olmalı)")
print(f"Son maliyet J = {gecmis[-1]:.4f}")
print()

# ---------------------------------------------------------------------------
# 3) SCIKIT-LEARN
# ---------------------------------------------------------------------------
model = LinearRegression()                           # Modeli oluştur
model.fit(x.reshape(-1, 1), y)                       # sklearn 2 boyutlu X ister → reshape(-1, 1)
print("=== 3) scikit-learn ===")
print(f"θ0 = {model.intercept_:.3f}, θ1 = {model.coef_[0]:.3f}")
print()

# ---------------------------------------------------------------------------
# 4) ÖĞRENME ORANININ ETKİSİ (deney)
# ---------------------------------------------------------------------------
print("=== 4) Farklı öğrenme oranları (100 adım sonra maliyet) ===")
for a in [0.001, 0.01, 0.05, 0.15, 0.2]:                  # Küçükten büyüğe öğrenme oranları
    a0, a1 = 0.0, 0.0                                # Her deneyde sıfırdan başla
    for _ in range(100):                             # 100 adım
        h = a0 + a1 * x - y                          # Hata
        a0, a1 = a0 - a * h.mean(), a1 - a * (h * x).mean()   # Aynı güncelleme kuralı
    j = maliyet(a0, a1)
    durum = "ıraksadı (çok büyük α!)" if not np.isfinite(j) or j > 1e6 else ""
    print(f"α = {a:<6} → J = {j:,.3f} {durum}")

# ---------------------------------------------------------------------------
# 5) GRAFİKLER
# ---------------------------------------------------------------------------
fig, eksen = plt.subplots(1, 2, figsize=(11, 4))     # Yan yana iki grafik

eksen[0].scatter(x, y, color="tab:blue", label="Gerçek veri")          # Veri noktaları
xx = np.linspace(0, 6, 50)                                             # Doğru için x değerleri
eksen[0].plot(xx, theta0 + theta1 * xx, color="tab:red", label="ŷ = θ0 + θ1·x")
for xi, yi in zip(x, y):                                               # Artıkları (residual) dikey çizgiyle göster
    eksen[0].plot([xi, xi], [yi, theta0 + theta1 * xi], "k--", lw=0.8)
eksen[0].set_xlabel("Çalışma süresi (saat)")
eksen[0].set_ylabel("Sınav notu")
eksen[0].set_title("En iyi uyan doğru ve artıklar")
eksen[0].legend()

eksen[1].plot(gecmis[:300])                                            # İlk 300 adımın maliyeti
eksen[1].set_xlabel("Adım (iterasyon)")
eksen[1].set_ylabel("Maliyet J(θ)")
eksen[1].set_title("Gradyan inişinde maliyetin düşüşü")

plt.tight_layout()                                   # Grafikler üst üste binmesin
plt.show()                                           # Pencereyi aç
