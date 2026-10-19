# -*- coding: utf-8 -*-
"""
00 - Ders Veri Setlerini Üretme
===============================
Ders notundaki kısa uygulamalarda kullanılan veri setlerini `data/` klasörüne üretir.
Veriler YAPAY (sentetik) ama gerçekçidir: hangi kurala göre üretildikleri aşağıda açıkça yazılıdır.
Sabit tohum (seed) kullanıldığı için betik her çalıştırıldığında AYNI dosyalar oluşur.

Üretilen dosyalar (her biri CSV + ARFF):
  notlar.csv                 : data/notlar.arff ile aynı 200 öğrenci (Python/R için CSV sürümü)
  ev_fiyatlari               : 240 ev, regresyon (hedef: fiyat_bin_tl)
  kredi_basvurulari          : 600 başvuru, sınıflandırma (hedef: risk), EKSİK DEĞERLİ, dengesiz (yaklaşık %20 yüksek risk)
  musteri_segmentleri        : 300 müşteri, kümeleme (etiketsiz, 5 doğal grup)
  market_sepeti              : 400 alışveriş sepeti, birliktelik kuralları
  weather.nominal.arff       : Quinlan'ın klasik 14 günlük hava durumu verisi (WEKA ile de gelir)
  meme_kanseri               : 569 hasta, GERÇEK veri (Wisconsin Diagnostic Breast Cancer; scikit-learn'deki
                               load_breast_cancer ile aynı). WEKA ve R kullananlar için CSV + ARFF olarak kaydedilir.

Çalıştırma (depo kök klasöründen):  python codes/python/00_veri_setlerini_uret.py
"""

import os  # Dosya yolları
import numpy as np  # Sayısal diziler, rastgele sayılar
import pandas as pd  # Tablolar, CSV yazma

KOK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Depo kök klasörü
DATA = os.path.join(KOK, "data")                                                      # Çıktı klasörü


def arff_yaz(df, dosya, relation, aciklama, nominal=None):
    """DataFrame'i WEKA'nın okuyabileceği ARFF dosyasına yazar. Eksik değerler '?' olur."""
    nominal = nominal or {}
    satirlar = [f"% {s}" for s in aciklama] + [f"@relation {relation}", ""]
    # Her sütun için bir @attribute satırı
    for kol in df.columns:
        if kol in nominal:                                        # Kategorik (nominal) öznitelik
            # Kategorik: olası değerler { } içinde
            satirlar.append(f"@attribute {kol} {{{', '.join(nominal[kol])}}}")
        else:                                                     # Sayısal öznitelik
            satirlar.append(f"@attribute {kol} numeric")
    satirlar += ["", "@data"]
    for _, r in df.iterrows():                                    # Her satır @data bölümüne
        degerler = ["?" if pd.isna(v) else (f"{v:g}" if isinstance(v, float) else str(v)) for v in r]
        satirlar.append(",".join(degerler))                       # Değerleri virgülle birleştir
    # UTF-8 olarak yazmak için aç
    with open(os.path.join(DATA, dosya), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(satirlar) + "\n")                       # Tüm satırları dosyaya yaz


# ---------------------------------------------------------------------------
# 1) notlar.csv  (data/notlar.arff dosyasının CSV sürümü)
# ---------------------------------------------------------------------------
def notlar_csv():
    satirlar = []
    with open(os.path.join(DATA, "notlar.arff"), encoding="utf-8") as f:  # ARFF dosyasını oku
        veri_bolumu = False
        for s in f:                                               # Satır satır ilerle
            s = s.strip()
            if s.lower() == "@data":                              # Veri bölümü buradan başlar
                veri_bolumu = True
                continue
            if veri_bolumu and s and not s.startswith("%"):       # Boş ve yorum satırlarını atla
                satirlar.append(s.split(","))                     # Değerleri sütunlara ayır
    df = pd.DataFrame(satirlar, columns=["vize", "final", "durum"]).astype({"vize": int, "final": int})
    df.to_csv(os.path.join(DATA, "notlar.csv"), index=False)      # CSV olarak kaydet
    return df


# ---------------------------------------------------------------------------
# 2) ev_fiyatlari  (regresyon)
#    fiyat = 400 + 28·m² + 60·oda − 25·yaş + 15·kat + semt farkı + otopark farkı + gürültü   (bin TL)
# ---------------------------------------------------------------------------
def ev_fiyatlari(rng):
    n = 240
    semt = rng.choice(["merkez", "cevre", "banliyo"], size=n, p=[0.30, 0.45, 0.25])
    m2 = np.clip(np.rint(rng.normal(120, 35, n)), 50, 260).astype(int)
    oda = np.clip(np.rint(m2 / 35 + rng.normal(0, 0.6, n)), 1, 6).astype(int)
    yas = rng.integers(0, 41, n)
    kat = rng.integers(0, 16, n)
    otopark = rng.choice(["evet", "hayir"], size=n, p=[0.55, 0.45])
    semt_farki = np.select([semt == "merkez", semt == "cevre"], [1500, 600], 0)
    fiyat = (400 + 28 * m2 + 60 * oda - 25 * yas + 15 * kat + semt_farki
             + np.where(otopark == "evet", 250, 0) + rng.normal(0, 350, n))
    df = pd.DataFrame({"metrekare": m2, "oda_sayisi": oda, "bina_yasi": yas, "kat": kat,
                       "semt": semt, "otopark": otopark, "fiyat_bin_tl": np.rint(fiyat).astype(int)})
    df.to_csv(os.path.join(DATA, "ev_fiyatlari.csv"), index=False)  # Python/R için CSV
    arff_yaz(df, "ev_fiyatlari.arff", "ev_fiyatlari",             # WEKA için ARFF
             ["240 evin özellikleri ve satış fiyatı (bin TL). Yapay veri, ders notu Bölüm 6 ve 9.4.",
              "Hedef (son öznitelik): fiyat_bin_tl -> regresyon problemi."],
             nominal={"semt": ["merkez", "cevre", "banliyo"], "otopark": ["evet", "hayir"]})
    return df


# ---------------------------------------------------------------------------
# 3) kredi_basvurulari  (dengesiz sınıflandırma + eksik değerler)
# ---------------------------------------------------------------------------
def kredi_basvurulari(rng):
    n = 600
    yas = rng.integers(21, 66, n)
    gelir = np.rint(np.exp(rng.normal(np.log(35000), 0.45, n)) / 100) * 100          # Aylık gelir (TL), log-normal
    borc = np.round(np.clip(rng.beta(2.2, 4.5, n) * 1.1, 0.02, 0.95), 2)            # Aylık taksit / gelir oranı
    calisma = np.minimum(rng.integers(0, 36, n), yas - 18)                           # Çalışma yılı
    sehir = rng.choice(["istanbul", "ankara", "izmir", "diger"], size=n, p=[0.35, 0.20, 0.15, 0.30])
    gecikme = rng.choice(["evet", "hayir"], size=n, p=[0.22, 0.78])                  # Geçmişte ödeme gecikmesi
    z = (-3.4 + 4.2 * borc + 1.7 * (gecikme == "evet") - 1.0 * np.log(gelir / 35000)
         - 0.04 * calisma + 0.25 * (sehir == "istanbul") + rng.normal(0, 0.5, n))
    risk = np.where(rng.random(n) < 1 / (1 + np.exp(-z)), "yuksek", "dusuk")
    df = pd.DataFrame({"yas": yas, "aylik_gelir": gelir, "borc_gelir_orani": borc, "calisma_yili": calisma,
                       "sehir": sehir, "gecmis_gecikme": gecikme, "risk": risk})
    df["aylik_gelir"] = df["aylik_gelir"].astype(float)
    df.loc[rng.choice(n, 30, replace=False), "aylik_gelir"] = np.nan                # %5 eksik gelir
    df.loc[rng.choice(n, 18, replace=False), "sehir"] = np.nan                      # %3 eksik şehir
    df.to_csv(os.path.join(DATA, "kredi_basvurulari.csv"), index=False)             # Python/R için CSV
    arff_yaz(df, "kredi_basvurulari.arff", "kredi_basvurulari",                     # WEKA için ARFF
             ["600 kredi başvurusu. Yapay veri, ders notu Bölüm 5.5-5.7, 9.3 ve 22.",
              "Hedef: risk {dusuk, yuksek}; yaklaşık %20 yüksek risk (dengesiz). Eksik değerler ? ile gösterilir."],
             nominal={"sehir": ["istanbul", "ankara", "izmir", "diger"], "gecmis_gecikme": ["evet", "hayir"],
                      "risk": ["dusuk", "yuksek"]})
    return df


# ---------------------------------------------------------------------------
# 4) musteri_segmentleri  (kümeleme; 5 doğal grup, etiket dosyada YOK)
# ---------------------------------------------------------------------------
def musteri_segmentleri(rng):
    # (yaş, yıllık gelir bin TL, harcama puanı 1-100, kişi sayısı)
    gruplar = [(24, 320, 78, 60),    # Genç, düşük gelir, çok harcıyor
               (37, 1100, 82, 55),   # Yüksek gelir, çok harcıyor
               (46, 1050, 20, 60),   # Yüksek gelir, az harcıyor (tutumlu)
               (40, 650, 50, 70),    # Ortalama müşteri
               (63, 360, 22, 55)]    # Emekli, düşük gelir, az harcıyor
    parca = []
    for y, g, h, k in gruplar:       # Her gruptan k müşteri üret
        parca.append(pd.DataFrame({  # Grubun müşterilerini listeye ekle
            "yas": np.clip(np.rint(rng.normal(y, 4.5, k)), 18, 80).astype(int),
            "yillik_gelir_bin_tl": np.clip(np.rint(rng.normal(g, 90, k)), 150, 1600).astype(int),
            "harcama_puani": np.clip(np.rint(rng.normal(h, 8, k)), 1, 100).astype(int)}))
    df = pd.concat(parca).sample(frac=1, random_state=7).reset_index(drop=True)      # Satırları karıştır
    df.to_csv(os.path.join(DATA, "musteri_segmentleri.csv"), index=False)            # Python/R için CSV
    arff_yaz(df, "musteri_segmentleri.arff", "musteri_segmentleri",                  # WEKA için ARFF
             ["300 müşteri: yaş, yıllık gelir (bin TL), harcama puanı (1-100). Yapay veri, ders notu Bölüm 17.",
              "Sınıf etiketi YOKTUR: kaç grup olduğunu kümeleme ile bulmak gerekir."])
    return df


# ---------------------------------------------------------------------------
# 5) market_sepeti  (birliktelik kuralları)
# ---------------------------------------------------------------------------
def market_sepeti(rng):
    n = 400
    urunler = ["ekmek", "sut", "peynir", "zeytin", "yumurta", "cay", "seker",
               "makarna", "salca", "yogurt", "domates", "deterjan"]
    temel = {"ekmek": 0.60, "sut": 0.40, "peynir": 0.30, "zeytin": 0.12, "yumurta": 0.30, "cay": 0.25,
             "seker": 0.10, "makarna": 0.25, "salca": 0.08, "yogurt": 0.30, "domates": 0.35, "deterjan": 0.15}
    S = {u: rng.random(n) < p for u, p in temel.items()}
    # Gömülü ilişkiler (öğrencilerin Apriori ile "keşfetmesi" beklenen kurallar)
    S["zeytin"] |= S["peynir"] & (rng.random(n) < 0.65)              # Kahvaltı: peynir → zeytin
    S["seker"] |= S["cay"] & (rng.random(n) < 0.60)                  # cay → seker
    S["salca"] |= S["makarna"] & (rng.random(n) < 0.70)              # makarna → salca
    S["yumurta"] |= (S["peynir"] & S["zeytin"]) & (rng.random(n) < 0.5)   # {peynir, zeytin} → yumurta
    df = pd.DataFrame({u: S[u].astype(int) for u in urunler})
    df = df[df.sum(axis=1) > 0].reset_index(drop=True)                 # Boş sepet olmasın
    df.to_csv(os.path.join(DATA, "market_sepeti.csv"), index=False)  # 0/1 tablo (Python/R)
    # WEKA sepet biçimi: ürün alındıysa t, alınmadıysa ? (supermarket.arff ile aynı mantık)
    tf = df.apply(lambda s: s.map({1: "t", 0: np.nan}))
    arff_yaz(tf, "market_sepeti.arff", "market_sepeti",              # WEKA sepet biçiminde ARFF
             [f"{len(df)} alışveriş sepeti. Yapay veri, ders notu Bölüm 18.",
              "Her öznitelik bir ürün: t = sepette var, ? = sepette yok."],
             nominal={u: ["t"] for u in urunler})
    return df


# ---------------------------------------------------------------------------
# 6) weather.nominal.arff  (Quinlan, 1986; WEKA'nın data klasöründeki dosyanın aynısı)
# ---------------------------------------------------------------------------
def weather_nominal():
    icerik = """% Klasik "tenis oynanır mı?" verisi (Quinlan, 1986). WEKA ile birlikte gelen weather.nominal.arff ile aynıdır.
% Ders notu Bölüm 11 (Naive Bayes) ve 12 (Karar Ağaçları) elle hesaplanan örnekler.
@relation weather.symbolic

@attribute outlook {sunny, overcast, rainy}
@attribute temperature {hot, mild, cool}
@attribute humidity {high, normal}
@attribute windy {TRUE, FALSE}
@attribute play {yes, no}

@data
sunny,hot,high,FALSE,no
sunny,hot,high,TRUE,no
overcast,hot,high,FALSE,yes
rainy,mild,high,FALSE,yes
rainy,cool,normal,FALSE,yes
rainy,cool,normal,TRUE,no
overcast,cool,normal,TRUE,yes
sunny,mild,high,FALSE,no
sunny,cool,normal,FALSE,yes
rainy,mild,normal,FALSE,yes
sunny,mild,normal,TRUE,yes
overcast,mild,high,TRUE,yes
overcast,hot,normal,FALSE,yes
rainy,mild,high,TRUE,no
"""
    # Dosyayı UTF-8 olarak yaz
    with open(os.path.join(DATA, "weather.nominal.arff"), "w", encoding="utf-8", newline="\n") as f:
        f.write(icerik)


# ---------------------------------------------------------------------------
# 7) Meme kanseri teşhis verisi (Wolberg vd., 1995; UCI, CC BY 4.0). YAPAY DEĞİL, gerçek veri.
#    Python kodları bunu doğrudan load_breast_cancer() ile alır; bu dosyalar WEKA ve R içindir.
# ---------------------------------------------------------------------------
def meme_kanseri():
    from sklearn.datasets import load_breast_cancer              # Veri scikit-learn ile birlikte gelir
    v = load_breast_cancer(as_frame=True)
    df = v.data.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]        # "mean radius" -> "mean_radius" (ARFF boşluk sevmez)
    df["tani"] = v.target.map({0: "kotu_huylu", 1: "iyi_huylu"})  # 0 = malignant, 1 = benign
    df.to_csv(os.path.join(DATA, "meme_kanseri.csv"), index=False)  # Python/R için CSV
    arff_yaz(df, "meme_kanseri.arff", "meme_kanseri",              # WEKA için ARFF
             ["Wisconsin Diagnostic Breast Cancer (WDBC) verisi: 569 hasta, 30 sayısal öznitelik.",
              "Kaynak: Wolberg, Street ve Mangasarian (1995), UCI Machine Learning Repository (CC BY 4.0).",
              "scikit-learn'deki load_breast_cancer() ile aynı veridir. Ders notu Bölüm 7, 8, 9, 12, 16, 19, 20.",
              "Not: WEKA ile gelen breast-cancer.arff FARKLI bir veridir (286 hasta, nominal öznitelikler)."],
             nominal={"tani": ["kotu_huylu", "iyi_huylu"]})
    return df


if __name__ == "__main__":                                           # Betik doğrudan çalıştırılırsa
    rng = np.random.default_rng(2026)                 # Tüm veri setleri için tek, sabit tohum
    n = notlar_csv()
    e = ev_fiyatlari(rng)
    k = kredi_basvurulari(rng)
    m = musteri_segmentleri(rng)
    s = market_sepeti(rng)
    weather_nominal()                                 # weather.nominal.arff dosyasını yaz
    b = meme_kanseri()
    print(f"notlar.csv           : {len(n)} satır, gecti={int((n.durum == 'gecti').sum())}")
    print(f"ev_fiyatlari         : {len(e)} satır, fiyat {e.fiyat_bin_tl.min()}–{e.fiyat_bin_tl.max()} bin TL")
    print(f"kredi_basvurulari    : {len(k)} satır, yuksek risk oranı={(k.risk == 'yuksek').mean():.3f}, "
          f"eksik gelir={k.aylik_gelir.isna().sum()}, eksik şehir={k.sehir.isna().sum()}")
    print(f"musteri_segmentleri  : {len(m)} satır")
    print(f"market_sepeti        : {len(s)} sepet, ürün başına destek: " +
          ", ".join(f"{u}={v:.2f}" for u, v in s.mean().items()))
    print(f"meme_kanseri         : {len(b)} hasta, " + ", ".join(f"{t}={c}" for t, c in b.tani.value_counts().items()))
