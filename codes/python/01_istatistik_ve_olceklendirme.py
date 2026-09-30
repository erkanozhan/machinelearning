# -*- coding: utf-8 -*-
"""
01 - Temel İstatistik ve Öznitelik Ölçeklendirme
================================================
Ders notu bölümü: "4. Veri ve Öznitelikler" → "Özniteliklerin Ölçeklendirilmesi"

Bu betik şunları gösterir:
  1) Ortalama, medyan, varyans ve standart sapmanın elle ve NumPy ile hesaplanması
  2) Min-Max normalizasyonu, Z-skoru standardizasyonu, onluk ölçekleme ve RobustScaler
  3) Aykırı bir değerin (outlier) her yönteme etkisi
  4) En önemli kural: Ölçekleyici SADECE eğitim verisiyle "fit" edilir!

Çalıştırma:  python 01_istatistik_ve_olceklendirme.py
"""

import numpy as np                                   # Sayısal işlemler için temel kütüphane
from sklearn.preprocessing import (                  # scikit-learn'ün ön işleme araçları
    MinMaxScaler,                                    # 0-1 aralığına sıkıştırır
    StandardScaler,                                  # Ortalama=0, std=1 yapar
    RobustScaler,                                    # Medyan ve çeyrekler açıklığı (IQR) kullanır
)

# ---------------------------------------------------------------------------
# 1) TEMEL İSTATİSTİKLER
# ---------------------------------------------------------------------------
notlar = np.array([60, 70, 80, 100], dtype=float)    # Ders notundaki örnek sınav notları

n = len(notlar)                                      # Örnek sayısı (n = 4)
ortalama = notlar.sum() / n                          # Aritmetik ortalama: toplam / adet
medyan = np.median(notlar)                           # Sıralanınca ortadaki değer (çift sayıda ise ortadaki ikisinin ortalaması)

# Varyans: Her değerin ortalamadan farkının karesinin ortalaması
pop_varyans = ((notlar - ortalama) ** 2).sum() / n          # Kitle (population) varyansı → n'e böl
orn_varyans = ((notlar - ortalama) ** 2).sum() / (n - 1)    # Örneklem (sample) varyansı → n-1'e böl

pop_std = np.sqrt(pop_varyans)                       # Kitle standart sapması (sklearn bunu kullanır)
orn_std = np.sqrt(orn_varyans)                       # Örneklem standart sapması (Excel STDEV, Weka bunu kullanır)

print("=== 1) Temel İstatistikler ===")
print(f"Veri              : {notlar}")
print(f"Ortalama (μ)      : {ortalama:.2f}")
print(f"Medyan            : {medyan:.2f}")
print(f"Kitle std  (÷n)   : {pop_std:.4f}   ← np.std(x)")
print(f"Örneklem std (÷n-1): {orn_std:.4f}   ← np.std(x, ddof=1)")
print()

# ---------------------------------------------------------------------------
# 2) ÖLÇEKLENDİRME YÖNTEMLERİ (elle ve kütüphane ile)
# ---------------------------------------------------------------------------
X = notlar.reshape(-1, 1)                            # sklearn 2 boyutlu dizi ister: (örnek sayısı, öznitelik sayısı)

# 2a) Min-Max: (x - min) / (max - min)
minmax_elle = (notlar - notlar.min()) / (notlar.max() - notlar.min())
minmax_skl = MinMaxScaler().fit_transform(X).ravel()        # fit: min/max'ı öğren, transform: dönüştür

# 2b) Z-skoru: (x - μ) / σ   (sklearn kitle std'sini, yani ÷n olanı kullanır)
z_elle = (notlar - ortalama) / pop_std
z_skl = StandardScaler().fit_transform(X).ravel()

# 2c) Onluk ölçekleme: x / 10^j,  j = max(|x|)'i 1'in altına indiren en küçük tam sayı
j = int(np.ceil(np.log10(np.abs(notlar).max() + 1e-12)))    # 100 için log10=2 → ama 100/10^2 = 1 olur (1'den küçük değil)
if np.abs(notlar).max() / 10 ** j >= 1:                      # Sınır durumu: tam 10'un kuvveti ise bir basamak daha kaydır
    j += 1
onluk = notlar / 10 ** j

# 2d) Robust: (x - medyan) / IQR    (IQR = 3. çeyrek - 1. çeyrek)
robust_skl = RobustScaler().fit_transform(X).ravel()

print("=== 2) Ölçeklendirme Sonuçları ===")
print(f"Min-Max (elle)    : {np.round(minmax_elle, 3)}")
print(f"Min-Max (sklearn) : {np.round(minmax_skl, 3)}")
print(f"Z-skoru (elle)    : {np.round(z_elle, 3)}")
print(f"Z-skoru (sklearn) : {np.round(z_skl, 3)}")
print(f"Onluk (j={j})       : {np.round(onluk, 3)}")
print(f"Robust (sklearn)  : {np.round(robust_skl, 3)}")
print()

# ---------------------------------------------------------------------------
# 3) AYKIRI DEĞERİN ETKİSİ
# ---------------------------------------------------------------------------
aykiri = np.array([60, 70, 80, 100, 300], dtype=float).reshape(-1, 1)   # 300: hatalı giriş (aykırı değer)

print("=== 3) Aykırı değer (300) eklenince ===")
print(f"Min-Max : {np.round(MinMaxScaler().fit_transform(aykiri).ravel(), 3)}  ← normal notlar 0-0.17 arasına sıkıştı")
print(f"Z-skoru : {np.round(StandardScaler().fit_transform(aykiri).ravel(), 3)}  ← ortalama ve std de bozuldu")
print(f"Robust  : {np.round(RobustScaler().fit_transform(aykiri).ravel(), 3)}  ← normal notlar makul aralıkta kaldı")
print()

# ---------------------------------------------------------------------------
# 4) DOĞRU KULLANIM: fit SADECE eğitim verisinde
# ---------------------------------------------------------------------------
egitim = np.array([[60], [70], [80], [100]], dtype=float)   # Modelin öğrendiği veri
test = np.array([[90], [110]], dtype=float)                 # Modelin daha önce görmediği veri

olcekleyici = MinMaxScaler()                         # Ölçekleyiciyi oluştur
olcekleyici.fit(egitim)                              # min=60, max=100 değerlerini SADECE eğitimden öğren
egitim_olcekli = olcekleyici.transform(egitim)       # Eğitimi dönüştür
test_olcekli = olcekleyici.transform(test)           # Testi AYNI kurallarla dönüştür (yeniden fit ETME!)

print("=== 4) Eğitim/Test ayrımıyla doğru ölçekleme ===")
print(f"Eğitim → {egitim_olcekli.ravel()}")
print(f"Test   → {test_olcekli.ravel()}   (110 → 1.25: aralık dışı olabilir, bu normaldir)")
