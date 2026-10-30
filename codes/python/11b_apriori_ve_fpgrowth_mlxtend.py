# -*- coding: utf-8 -*-
"""
11b - Birliktelik Kuralları: mlxtend ile Apriori ve FP-Growth
=============================================================
Ders notu bölümü: "18. Birliktelik Kuralları" (18.4 FP-Growth, 18.6 Uygulama)

  1) Ders notundaki 8 sepette Apriori ve FP-Growth'un aynı sık kümeleri bulduğunu gösterir
  2) data/market_sepeti.csv (391 sepet) üzerinde iki algoritmayı çalıştırır ve karşılaştırır
  3) Üç ürünlü kurallar dahil tüm kuralları üretir (WEKA Associate sekmesinin karşılığı)
  4) Güveni yüksek ama lift'i 1'e yakın (değersiz) kuralları ayırır

Kurulum:     pip install pandas "mlxtend>=0.23"
Çalıştırma:  python 11b_apriori_ve_fpgrowth_mlxtend.py
"""

import os                                                     # Dosya yolları
import sys                                                    # Konsol kodlaması
import time                                                   # Süre ölçümü
import pandas as pd                                           # Tablo işlemleri
from mlxtend.preprocessing import TransactionEncoder          # Sepet listesi → 0/1 tablo
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules   # Algoritmalar ve kural üretimi

sys.stdout.reconfigure(encoding="utf-8")                      # Windows konsolunda Türkçe karakter ve "→" için
pd.set_option("display.width", 140)                           # Geniş tablolar tek satırda görünsün


def kume(s):
    """frozenset({'b', 'a'}) → '{a, b}' (okunaklı yazım)"""
    return "{" + ", ".join(sorted(s)) + "}"


def kurallari_yazdir(kurallar):
    """mlxtend kural tablosunu 'X → Y, destek, güven, lift' biçiminde gösterir."""
    tablo = pd.DataFrame({
        "kural": kurallar["antecedents"].apply(kume) + " → " + kurallar["consequents"].apply(kume),
        "destek": kurallar["support"],                        # supp(X ∪ Y)
        "guven": kurallar["confidence"],                      # conf(X → Y)
        "lift": kurallar["lift"],                             # lift(X → Y)
    })
    print(tablo.round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# 1) DERS NOTUNDAKİ 8 SEPET: Apriori ve FP-Growth aynı sonucu verir
# ---------------------------------------------------------------------------
print("=" * 70)
print("1) Ders notundaki 8 sepet (min. destek = 0.25, min. güven = 0.60)")
print("=" * 70)
sepetler = [
    ["ekmek", "süt"],
    ["ekmek", "tereyağı", "yumurta"],
    ["süt", "tereyağı", "çay"],
    ["ekmek", "süt", "tereyağı"],
    ["ekmek", "süt", "tereyağı", "çay"],
    ["süt", "çay"],
    ["ekmek", "tereyağı"],
    ["ekmek", "süt", "yumurta"],
]
te = TransactionEncoder()                                     # Sepet listesini tabloya çeviren nesne
X8 = pd.DataFrame(te.fit(sepetler).transform(sepetler), columns=te.columns_)   # Her ürün bir sütun (True/False)
print(X8.astype(int).to_string())                             # 0/1 tabloyu göster

sik_a = apriori(X8, min_support=0.25, use_colnames=True)      # Apriori: seviye seviye aday üretir
sik_f = fpgrowth(X8, min_support=0.25, use_colnames=True)     # FP-Growth: aday üretmez, FP-ağacı kurar
print(f"\nApriori {len(sik_a)}, FP-Growth {len(sik_f)} sık küme buldu. "
      f"Aynı mı? {set(sik_a.itemsets) == set(sik_f.itemsets)}")
for _, r in sik_f.sort_values("support", ascending=False).iterrows():   # Sık kümeleri desteğe göre yaz
    print(f"  {kume(r['itemsets']):<28} destek = {r['support']:.3f}")

kurallar8 = association_rules(sik_f, num_itemsets=len(X8),   # Sık kümelerden kural üret
                              metric="confidence", min_threshold=0.6)
print("\nKurallar (lift'e göre):")
kurallari_yazdir(kurallar8.sort_values("lift", ascending=False))

# ---------------------------------------------------------------------------
# 2) MARKET SEPETİ VERİSİ (391 sepet, 12 ürün)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2) data/market_sepeti.csv (min. destek = 0.10, min. güven = 0.60)")
print("=" * 70)
try:                                                          # Betik olarak çalışıyorsa depo kökünü bul
    KOK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:                                             # Colab/Jupyter: __file__ tanımlı değil
    KOK = ""
YOL = os.path.join(KOK, "data", "market_sepeti.csv")           # Yerel veri dosyası
if not os.path.exists(YOL):                                   # Colab vb. ortamda dosya yoksa GitHub'dan oku
    YOL = "https://raw.githubusercontent.com/erkanozhan/machinelearning/main/data/market_sepeti.csv"
S = pd.read_csv(YOL).astype(bool)                             # Dosya zaten 0/1 tablo; mlxtend True/False ister
print(f"{S.shape[0]} sepet, {S.shape[1]} ürün")
print("Ürün destekleri:", S.mean().sort_values(ascending=False).round(2).to_dict())

t0 = time.perf_counter()
sik_a = apriori(S, min_support=0.10, use_colnames=True)       # Apriori
t_a = time.perf_counter() - t0
t0 = time.perf_counter()
sik_f = fpgrowth(S, min_support=0.10, use_colnames=True)      # FP-Growth
t_f = time.perf_counter() - t0

boyut = sik_f["itemsets"].apply(len)                          # Her sık kümenin eleman sayısı
print(f"\nSık küme sayısı: Apriori {len(sik_a)}, FP-Growth {len(sik_f)} "
      f"(aynı mı? {set(sik_a.itemsets) == set(sik_f.itemsets)})")
print("Boyuta göre (WEKA'daki L(1), L(2), L(3)):", boyut.value_counts().sort_index().to_dict())
print(f"Süre: Apriori {t_a * 1000:.1f} ms, FP-Growth {t_f * 1000:.1f} ms "
      "(bu küçük veride fark önemsizdir; FP-Growth büyük veride öne geçer)")

print("\nÜç ürünlü sık kümeler:")
for _, r in sik_f[boyut == 3].iterrows():
    print(f"  {kume(r['itemsets']):<28} destek = {r['support']:.3f} ({round(r['support'] * len(S))} sepet)")

# ---------------------------------------------------------------------------
# 3) KURALLAR: Üç ürünlü kurallar dahil (WEKA'daki "Best rules found" listesi)
# ---------------------------------------------------------------------------
kurallar = association_rules(sik_f, num_itemsets=len(S), metric="confidence", min_threshold=0.6)
kurallar = kurallar.sort_values("confidence", ascending=False)   # WEKA gibi güvene göre sırala
print(f"\n{len(kurallar)} kural bulundu (güven ≥ 0.60):")
kurallari_yazdir(kurallar)

# ---------------------------------------------------------------------------
# 4) LIFT İLE SÜZME: Güven yüksek ama ilişki yok
# ---------------------------------------------------------------------------
print("\nGerçekten ilginç kurallar (lift ≥ 1.5):")
kurallari_yazdir(kurallar[kurallar["lift"] >= 1.5].sort_values("lift", ascending=False))

print(f"\nDeğersiz kurallar (lift < 1.2). Ekmek zaten sepetlerin %{S['ekmek'].mean() * 100:.0f}'ünde var:")
kurallari_yazdir(kurallar[kurallar["lift"] < 1.2])
print("\nlift > 1: birlikte alınma şanstan yüksek | lift = 1: bağımsız | lift < 1: birbirini dışlıyor")
