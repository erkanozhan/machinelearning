# -*- coding: utf-8 -*-
"""
11 - Birliktelik Kuralları ve Apriori Algoritması
=================================================
Ders notu bölümü: "16. Birliktelik Kuralları"

Ek kütüphane gerektirmeden (sadece Python) Apriori algoritmasını uygular:
  1) Sık öğe kümelerini (frequent itemsets) seviye seviye bulur
  2) Bu kümelerden kural üretip destek (support), güven (confidence) ve kaldıraç (lift) hesaplar

Veri: Ders notundaki 8 alışveriş sepeti.
(Hazır bir kütüphane isterseniz: pip install mlxtend → mlxtend.frequent_patterns.apriori)

Çalıştırma:  python 11_birliktelik_kurallari_apriori.py
"""

from itertools import combinations                   # Alt küme / kombinasyon üretmek için

# ---------------------------------------------------------------------------
# 0) VERİ: Her satır bir müşterinin sepeti (işlem / transaction)
# ---------------------------------------------------------------------------
sepetler = [
    {"ekmek", "süt"},
    {"ekmek", "tereyağı", "yumurta"},
    {"süt", "tereyağı", "çay"},
    {"ekmek", "süt", "tereyağı"},
    {"ekmek", "süt", "tereyağı", "çay"},
    {"süt", "çay"},
    {"ekmek", "tereyağı"},
    {"ekmek", "süt", "yumurta"},
]
N = len(sepetler)                                    # Toplam işlem sayısı
MIN_DESTEK = 0.25                                    # Bir öğe kümesinin "sık" sayılması için en az %25 sepette geçmeli
MIN_GUVEN = 0.6                                      # Bir kuralın kabul edilmesi için en az %60 güven


def destek(ogeler):
    """support(X) = X'i içeren sepet sayısı / toplam sepet sayısı"""
    return sum(1 for s in sepetler if ogeler <= s) / N          # ogeler <= s : ogeler kümesi s'nin alt kümesi mi?


# ---------------------------------------------------------------------------
# 1) APRIORI: sık öğe kümelerini seviye seviye bul
#    Apriori ilkesi: Bir küme sık değilse, onu içeren hiçbir büyük küme de sık olamaz.
# ---------------------------------------------------------------------------
tum_ogeler = sorted(set().union(*sepetler))          # Tüm farklı ürünler
sik_kumeler = {}                                     # {frozenset: destek}
aday = [frozenset([o]) for o in tum_ogeler]          # Seviye 1: tek elemanlı adaylar
seviye = 1
while aday:
    print(f"--- Seviye {seviye}: {len(aday)} aday ---")
    bu_seviye = {}
    for kume in aday:
        d = destek(kume)
        durum = "✓ sık" if d >= MIN_DESTEK else "✗ elendi"
        print(f"  {set(kume)}: destek = {d:.3f} {durum}")
        if d >= MIN_DESTEK:
            bu_seviye[kume] = d
    sik_kumeler.update(bu_seviye)
    # Bir sonraki seviyenin adaylarını üret: sık kümeleri ikişer ikişer birleştir
    anahtarlar = list(bu_seviye)
    yeni_aday = set()
    for a, b in combinations(anahtarlar, 2):
        birlesim = a | b
        if len(birlesim) == seviye + 1:
            # Budama (pruning): birleşimin TÜM alt kümeleri sık olmalı (Apriori ilkesi)
            if all(frozenset(alt) in bu_seviye for alt in combinations(birlesim, seviye)):
                yeni_aday.add(birlesim)
    aday = list(yeni_aday)
    seviye += 1
print()

# ---------------------------------------------------------------------------
# 2) KURAL ÜRETİMİ:  X → Y
#    confidence(X→Y) = support(X ∪ Y) / support(X)
#    lift(X→Y)       = confidence(X→Y) / support(Y)
# ---------------------------------------------------------------------------
print("=== Kurallar (güven ≥ %.0f%%) ===" % (MIN_GUVEN * 100))
print(f"{'Kural':<32} {'Destek':>7} {'Güven':>7} {'Lift':>6}")
kurallar = []
for kume, d_xy in sik_kumeler.items():
    if len(kume) < 2:
        continue                                     # Tek elemanlı kümeden kural çıkmaz
    for r in range(1, len(kume)):
        for sol in combinations(kume, r):            # Kuralın sol tarafı (öncül / antecedent)
            X = frozenset(sol)
            Y = kume - X                             # Sağ taraf (sonuç / consequent)
            guven = d_xy / sik_kumeler[X]
            lift = guven / destek(Y)
            if guven >= MIN_GUVEN:
                kurallar.append((X, Y, d_xy, guven, lift))
for X, Y, d, g, l in sorted(kurallar, key=lambda k: -k[4]):   # Lift'e göre büyükten küçüğe
    print(f"{str(set(X)) + ' → ' + str(set(Y)):<32} {d:>7.3f} {g:>7.3f} {l:>6.2f}")
print("\nlift > 1: birlikte alınma şanstan yüksek | lift = 1: bağımsız | lift < 1: birbirini dışlıyor")
