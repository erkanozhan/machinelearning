# 4. Araçlar: WEKA ve Python

Bu derste iki araç birlikte kullanılır:

| | **WEKA** | **Python (scikit-learn)** |
| :--- | :--- | :--- |
| Kullanım | Görsel arayüz, kod yazmadan | Kod yazarak |
| Avantaj | Hızlı deneme, kavramları görsel öğrenme | Esneklik, otomasyon, endüstri standardı |
| Dosya formatı | `.arff` (ayrıca `.csv`) | `.csv`, NumPy dizileri, pandas tabloları |

## 4.1 WEKA'ya Hızlı Başlangıç

**WEKA (Waikato Environment for Knowledge Analysis)**, Yeni Zelanda'daki Waikato Üniversitesi tarafından geliştirilmiş, **Java tabanlı**, **açık kaynak** bir makine öğrenmesi yazılımıdır. İçinde sınıflandırma, regresyon, kümeleme, birliktelik kuralları, öznitelik seçimi ve görselleştirme için yüzlerce algoritma vardır. [weka.io](https://ml.cms.waikato.ac.nz/weka) adresinden indirilebilir.

**Başlatma:** Windows'ta "**Weka (with Console)**" seçeneğiyle açmak iyi bir alışkanlıktır. Konsol penceresi arka planda olan işlemleri ve hata mesajlarını gösterir.

**Bellek ayarı:** Büyük veri setlerinde `OutOfMemoryError` (bellek yetersiz) hatası alabilirsiniz. WEKA'nın kurulu olduğu klasördeki `RunWeka.ini` dosyasında `maxheap=` değerini artırın (ör. `maxheap=4g`) ve WEKA'yı yeniden başlatın.

**GUI Chooser (ana pencere) seçenekleri:**

| Arayüz | Ne için kullanılır? |
| :--- | :--- |
| **Explorer** | Veri yükleme, ön işleme (İng. *preprocessing*; Lat. *processus*: "ilerleyiş, süreç"), model kurma ve değerlendirme. Derste en çok kullanacağımız ekran. |
| **Experimenter** | Birden çok algoritmayı birden çok veri setinde, tekrarlı deneylerle ve istatistiksel testlerle karşılaştırma ([Bölüm 24](24-weka-experimenter.md)). |
| **KnowledgeFlow** | Sürükle-bırak ile veri akışı (İng. *flow*) tasarlama ([Bölüm 23](23-weka-knowledgeflow.md)). |
| **Workbench** | Yukarıdakilerin tümünü tek pencerede toplayan arayüz. |
| **Simple CLI** | Komut satırından WEKA komutları çalıştırma. |

**Explorer sekmeleri:**

| Sekme | İşlevi |
| :--- | :--- |
| **Preprocess** | Veri yükleme, istatistikleri görme, filtre (ön işleme) uygulama. |
| **Classify** | Sınıflandırma **ve regresyon** modelleri kurma ve test etme. |
| **Cluster** | Kümeleme. |
| **Associate** | Birliktelik kuralları (sepet analizi). |
| **Select attributes** | Öznitelik seçimi. |
| **Visualize** | Öznitelik çiftlerinin dağılım grafikleri. |

## 4.2 ARFF Dosya Formatı

WEKA'nın kendi dosya biçimi **ARFF**'dir (*Attribute-Relation File Format*). Düz bir metin dosyasıdır ve üç bölümden oluşur:

```text
% Yüzde (%) işaretiyle başlayan satırlar yorumdur. Yorumları her zaman ayrı satıra yazın.
% 1) Veri setinin adı
@relation ogrenci_notlari

% 2) Öznitelik adı ve türü (numeric = sayısal, { } = kategorik/nominal değerler)
@attribute vize numeric
@attribute final numeric
@attribute durum {gecti, kaldi}

% 3) Veriler: her satır bir örnek, değerler virgülle ayrılır
@data
60,75,gecti
45,50,kaldi
80,90,gecti
95,92,gecti
30,40,kaldi
```

- Denetimli öğrenmede tahmin edilecek **hedef öznitelik (sınıf)** genellikle **son** `@attribute` olarak yazılır.
- Bilinmeyen (eksik) değerler `?` ile gösterilir.
- Diğer türler: `string` (metin), `date` (tarih).
- `.csv` dosyalarını WEKA'da açıp **Save** ile `.arff` olarak kaydedebilirsiniz. Veritabanından veri çekmek için ilgili **JDBC** sürücüsünün WEKA'ya tanıtılması gerekir.

Bu dosyanın **200 öğrencilik** genişletilmiş hâli depoda hazırdır: [`data/notlar.arff`](https://github.com/erkanozhan/machinelearning/blob/main/data/notlar.arff). Öğrencinin durumu, dönem ortalamasına göre belirlenmiştir: $\text{ortalama} = 0.4 \cdot \text{vize} + 0.6 \cdot \text{final}$; ortalaması **60 ve üstü** olanlar `gecti`, diğerleri `kaldi` (117 geçen, 83 kalan).

Derse özel, Türkçe veri setleri depodaki [`data/`](https://github.com/erkanozhan/machinelearning/tree/main/data) klasöründedir; çoğu hem CSV (Python/R) hem ARFF (WEKA) biçimindedir (liste: [Ek C](ek-c-kodlar-ve-animasyonlar.md)). Ayrıca WEKA'nın kurulum klasöründeki `data/` dizininde derste kullanacağımız hazır veri setleri bulunur: `iris.arff`, `weather.nominal.arff`, `cpu.arff`, `diabetes.arff`, `credit-g.arff`, `supermarket.arff` vb.

## 4.3 Python ve scikit-learn

**scikit-learn**, Python'daki en yaygın makine öğrenmesi kütüphanesidir. Neredeyse tüm modeller aynı arayüzü kullanır; bunu bir kez öğrenmek yeterlidir:

```python
from sklearn.tree import DecisionTreeClassifier   # 1) Modeli içe aktar

model = DecisionTreeClassifier()                   # 2) Modeli oluştur (hiperparametreler burada verilir)
model.fit(X_egitim, y_egitim)                      # 3) Eğit: parametreleri veriden öğren
y_tahmin = model.predict(X_test)                   # 4) Görülmemiş veride tahmin yap
olasilik = model.predict_proba(X_test)             # 5) (Varsa) sınıf olasılıklarını al
```

Ön işleme araçları da benzer biçimde çalışır: `fit` (istatistikleri öğren), `transform` (dönüştür), `fit_transform` (ikisi birden).

> **Altın kural:** `fit` her zaman **yalnızca eğitim verisiyle** yapılır. Test (İng. *test*; Lat. *testum*: madenleri denemek için kullanılan toprak kap) verisine sadece `transform` / `predict` uygulanır. Bu kuralın neden bu kadar önemli olduğu [Bölüm 21](21-veri-sizintisi.md)'de anlatılmaktadır.

---

[← 3. Öğrenme Türleri](03-ogrenme-turleri.md) · [İçindekiler](../Ders_Notu.md) · [5. Veri ve Öznitelikler →](05-veri-ve-oznitelikler.md)
