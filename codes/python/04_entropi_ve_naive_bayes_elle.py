# -*- coding: utf-8 -*-
"""
04 - Entropi, Bilgi Kazancı ve Naive Bayes'in Elle Hesaplanması
===============================================================
Ders notu bölümleri: "8. Naive Bayes" ve "9. Karar Ağaçları"

Weka ile birlikte gelen meşhur "weather.nominal" (hava durumu / tenis oynama)
veri seti üzerinde hiçbir ML kütüphanesi kullanmadan:
  1) Entropi H(S)
  2) Her öznitelik için Bilgi Kazancı (Information Gain) ve Kazanç Oranı (Gain Ratio)
  3) Gini safsızlığı
  4) Naive Bayes ile yeni bir gün için "oynanır mı?" tahmini
hesaplanır. Sonuçlar ders notundaki elle yapılan hesaplarla birebir aynıdır.

Çalıştırma:  python 04_entropi_ve_naive_bayes_elle.py
"""

from collections import Counter                      # Değerleri saymak için
from math import log2                                # 2 tabanında logaritma (bit cinsinden entropi)

# ---------------------------------------------------------------------------
# 0) VERİ: weather.nominal.arff (14 gün)
# ---------------------------------------------------------------------------
sutunlar = ["outlook", "temperature", "humidity", "windy", "play"]   # Son sütun: hedef (sınıf)
veri = [
    ["sunny", "hot", "high", "false", "no"],
    ["sunny", "hot", "high", "true", "no"],
    ["overcast", "hot", "high", "false", "yes"],
    ["rainy", "mild", "high", "false", "yes"],
    ["rainy", "cool", "normal", "false", "yes"],
    ["rainy", "cool", "normal", "true", "no"],
    ["overcast", "cool", "normal", "true", "yes"],
    ["sunny", "mild", "high", "false", "no"],
    ["sunny", "cool", "normal", "false", "yes"],
    ["rainy", "mild", "normal", "false", "yes"],
    ["sunny", "mild", "normal", "true", "yes"],
    ["overcast", "mild", "high", "true", "yes"],
    ["overcast", "hot", "normal", "false", "yes"],
    ["rainy", "mild", "high", "true", "no"],
]
hedef = [satir[-1] for satir in veri]                # Sınıf sütunu: yes/no listesi


# ---------------------------------------------------------------------------
# 1) ENTROPİ:  H(S) = - Σ p_i · log2(p_i)
# ---------------------------------------------------------------------------
def entropi(etiketler):
    """Bir etiket listesinin entropisini (bit) hesaplar."""
    n = len(etiketler)                               # Toplam örnek sayısı
    sayim = Counter(etiketler)                       # Her sınıftan kaç tane var? ör. {'yes': 9, 'no': 5}
    return -sum((c / n) * log2(c / n) for c in sayim.values())   # Tanımı birebir uyguluyoruz


def gini(etiketler):
    """Gini safsızlığı: 1 - Σ p_i²  (CART / sklearn varsayılanı)"""
    n = len(etiketler)
    return 1 - sum((c / n) ** 2 for c in Counter(etiketler).values())


H_S = entropi(hedef)                                 # Kök düğümün entropisi
print("=== 1) Kök düğüm ===")
print(f"Sınıf dağılımı: {dict(Counter(hedef))}")
print(f"Entropi H(S) = {H_S:.3f} bit")
print(f"Gini(S)      = {gini(hedef):.3f}")
print()


# ---------------------------------------------------------------------------
# 2) BİLGİ KAZANCI ve KAZANÇ ORANI
#    IG(S, A)   = H(S) - Σ_v (|S_v|/|S|) · H(S_v)
#    SplitInfo  = - Σ_v (|S_v|/|S|) · log2(|S_v|/|S|)
#    GainRatio  = IG / SplitInfo          (C4.5 / Weka J48 bunu kullanır)
# ---------------------------------------------------------------------------
def bilgi_kazanci(sutun_indeksi):
    """Verilen öznitelikle bölmenin bilgi kazancını ve kazanç oranını döndürür."""
    n = len(veri)
    degerler = Counter(satir[sutun_indeksi] for satir in veri)       # Özniteliğin değerleri ve sıklıkları
    agirlikli_entropi = 0.0                                          # Bölme sonrası kalan belirsizlik
    split_info = 0.0                                                 # Bölmenin kendi "dağınıklığı"
    for deger, adet in degerler.items():
        alt_kume = [satir[-1] for satir in veri if satir[sutun_indeksi] == deger]   # S_v: bu değere sahip satırların sınıfları
        oran = adet / n                                              # |S_v| / |S|
        agirlikli_entropi += oran * entropi(alt_kume)                # Ağırlıklı ortalama entropi
        split_info -= oran * log2(oran)
    ig = H_S - agirlikli_entropi                                     # Belirsizlikteki azalma
    return ig, ig / split_info


print("=== 2) Her öznitelik için bilgi kazancı ===")
print(f"{'Öznitelik':<12} {'IG':>7} {'GainRatio':>10}")
for i, ad in enumerate(sutunlar[:-1]):               # Hedef hariç tüm sütunlar
    ig, gr = bilgi_kazanci(i)
    print(f"{ad:<12} {ig:>7.3f} {gr:>10.3f}")
print("→ En yüksek kazanç 'outlook' → ağacın kökü outlook olur.")
print()

# ---------------------------------------------------------------------------
# 3) NAIVE BAYES:  P(sınıf | x) ∝ P(sınıf) · Π P(x_i | sınıf)
# ---------------------------------------------------------------------------
yeni_gun = {"outlook": "sunny", "temperature": "cool", "humidity": "high", "windy": "true"}
print("=== 3) Naive Bayes ile tahmin ===")
print(f"Yeni gün: {yeni_gun}")

skorlar = {}                                          # Her sınıf için normalize edilmemiş skor
for sinif in ["yes", "no"]:
    sinif_satirlari = [s for s in veri if s[-1] == sinif]            # Bu sınıfa ait günler
    oncul = len(sinif_satirlari) / len(veri)                         # Önsel olasılık P(sınıf)
    skor = oncul
    aciklama = [f"P({sinif})={len(sinif_satirlari)}/{len(veri)}"]
    for ad, deger in yeni_gun.items():
        j = sutunlar.index(ad)                                       # Özniteliğin sütun numarası
        adet = sum(1 for s in sinif_satirlari if s[j] == deger)      # Bu sınıfta bu değer kaç kez geçmiş
        olabilirlik = adet / len(sinif_satirlari)                    # P(x_i | sınıf)
        skor *= olabilirlik                                          # "Naive" varsayım: bağımsız → çarp
        aciklama.append(f"P({deger}|{sinif})={adet}/{len(sinif_satirlari)}")
    skorlar[sinif] = skor
    print(f"{sinif:>3}: " + " · ".join(aciklama) + f" = {skor:.4f}")

toplam = sum(skorlar.values())                       # Normalizasyon için toplam
for sinif, skor in skorlar.items():
    print(f"P({sinif} | yeni gün) = {skor:.4f} / {toplam:.4f} = {skor / toplam:.3f}")
print(f"Tahmin: play = {max(skorlar, key=skorlar.get)}")
