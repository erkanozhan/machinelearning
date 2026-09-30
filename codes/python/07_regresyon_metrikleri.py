# -*- coding: utf-8 -*-
"""
07 - Regresyon Performans Ölçütleri
===================================
Ders notu bölümü: "9. Performans Ölçütleri" → Regresyon

Ders notundaki 5 evlik örnek üzerinde MAE, MSE, RMSE, R², Düzeltilmiş R²
ve korelasyon katsayısını hem elle hem scikit-learn ile hesaplar.
Ayrıca R²'nin NEGATİF olabileceğini gösterir.

Çalıştırma:  python 07_regresyon_metrikleri.py
"""

import numpy as np                                            # Sayısal işlemler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score   # Hazır metrik fonksiyonları

y = np.array([250, 300, 200, 500, 420], dtype=float)          # Gerçek fiyatlar (bin TL)
y_hat = np.array([260, 290, 215, 480, 450], dtype=float)      # Modelin tahminleri
n = len(y)                                                    # Örnek sayısı

hata = y - y_hat                                              # Artıklar (residuals): e_i = y_i - ŷ_i
mae = np.abs(hata).mean()                                     # Ortalama mutlak hata
mse = (hata ** 2).mean()                                      # Ortalama karesel hata
rmse = np.sqrt(mse)                                           # Kök ortalama karesel hata

ss_res = (hata ** 2).sum()                                    # Artık kareler toplamı (açıklanamayan)
ss_tot = ((y - y.mean()) ** 2).sum()                          # Toplam kareler (toplam değişkenlik)
r2 = 1 - ss_res / ss_tot                                      # Belirlilik katsayısı

k = 1                                                         # Modeldeki öznitelik sayısı (örnek olarak 1 alıyoruz)
r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)                 # Düzeltilmiş R²

r = np.corrcoef(y, y_hat)[0, 1]                               # Pearson korelasyon katsayısı (gerçek vs tahmin)

print("=== Elle hesap ===")
print(f"Artıklar       : {hata}")
print(f"MAE            : {mae:.2f}")
print(f"MSE            : {mse:.2f}")
print(f"RMSE           : {rmse:.2f}")
print(f"SS_res, SS_tot : {ss_res:.0f}, {ss_tot:.0f}")
print(f"R²             : {r2:.4f}")
print(f"Düzeltilmiş R² : {r2_adj:.4f}  (k={k})")
print(f"r (korelasyon) : {r:.4f}   r² = {r**2:.4f}  ← R² ile AYNI DEĞİL (tahminler en küçük kareler doğrusu değil)")
print()

print("=== scikit-learn kontrol ===")
print(f"MAE={mean_absolute_error(y, y_hat):.2f}  MSE={mean_squared_error(y, y_hat):.2f}  "
      f"RMSE={np.sqrt(mean_squared_error(y, y_hat)):.2f}  R²={r2_score(y, y_hat):.4f}")
# Not: sklearn >= 1.4'te root_mean_squared_error() fonksiyonu da vardır.
# Eski "mean_squared_error(..., squared=False)" kullanımı yeni sürümlerde kaldırılmıştır.
print()

print("=== R² negatif olabilir mi? ===")
kotu_tahmin = np.array([500, 200, 450, 250, 300], dtype=float)   # Ortalamadan bile kötü tahminler
print(f"Kötü model R² = {r2_score(y, kotu_tahmin):.3f}  ← Her eve ortalamayı ({y.mean():.0f}) söyleyen modelden daha kötü")
print(f"Ortalama model R² = {r2_score(y, np.full(n, y.mean())):.3f}")
