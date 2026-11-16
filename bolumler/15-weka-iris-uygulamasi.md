# 15. WEKA ile Uçtan Uca Uygulama: Iris

Bu uygulama, bir makine öğrenmesi projesinin **eğitim → değerlendirme → kaydetme → yeni veride kullanma** adımlarının tamamını içerir.

**Iris veri seti:** 3 süsen türünden (*Iris-setosa*, *Iris-versicolor*, *Iris-virginica*) 50'şer, toplam 150 çiçek. 4 sayısal öznitelik: çanak yaprak (sepal) uzunluğu ve genişliği, taç yaprak (petal) uzunluğu ve genişliği (cm).

## Adım 1 – Veriyi Yükleme ve İnceleme

1. GUI Chooser → **Explorer** → Preprocess → **Open file…** → WEKA klasöründeki `data/iris.arff`.
2. Sağdaki **Attributes** panelinde öznitelikleri tek tek seçip istatistiklerine (min, max, mean, stddev) ve histogramlarına bakın. Histogramın üstündeki açılır menüde `Class: class (Nom)` seçiliyken histogramlar sınıflara göre renklenir (tümünü birlikte görmek için **Visualize All**). `petallength` histogramında setosa'nın tamamen ayrık durduğunu fark edin.
3. **Visualize** sekmesinde öznitelik çiftlerinin dağılım grafiklerini inceleyin.

## Adım 2 – Model Eğitme ve Değerlendirme

1. **Classify** → Choose → `trees → J48`.
2. Test options: **Cross-validation, Folds = 10**. Altındaki açılır menüde hedef olarak `(Nom) class` seçili olsun.
3. **Start**.

Tipik çıktı (10 katlı CV, seed = 1):

```text
Correctly Classified Instances         144               96      %    ← Doğruluk (accuracy)
Incorrectly Classified Instances         6                4      %
Kappa statistic                          0.94                         ← Şansa göre düzeltilmiş uyum
Mean absolute error                      0.035
Root mean squared error                  0.1586

=== Detailed Accuracy By Class ===
                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC    ROC Area  PRC Area  Class
                 0.980    0.000    1.000      0.980    0.990      0.985  0.990     0.987     Iris-setosa
                 0.940    0.030    0.940      0.940    0.940      0.910  0.952     0.880     Iris-versicolor
                 0.960    0.030    0.941      0.960    0.950      0.925  0.961     0.905     Iris-virginica
Weighted Avg.    0.960    0.020    0.960      0.960    0.960      0.940  0.968     0.924

=== Confusion Matrix ===
  a  b  c   <-- classified as
 49  1  0 |  a = Iris-setosa
  0 47  3 |  b = Iris-versicolor
  0  2 48 |  c = Iris-virginica
```

**Yorum:**
- `TP Rate` = recall, `FP Rate` = sahte pozitif oranı, `F-Measure` = F1, `ROC Area` = AUC, `PRC Area` = PR eğrisi altındaki alan.
- WEKA'da karışıklık matrisinin **satırları gerçek sınıfı, sütunları tahmini** gösterir. `b` satırı, `c` sütunundaki 3: *Gerçekte versicolor olan 3 çiçek virginica sanılmış.*
- Hataların tamamına yakını **versicolor ile virginica** arasındadır; bu iki türün ölçüleri birbirine çok yakındır.
- `Mean absolute error` gibi değerler sınıflandırmada **olasılık tahminlerinin** hatasıdır; yorumlarken öncelik doğruluk, Kappa ve sınıf bazlı ölçütlerdedir.
- Ağacı görmek için: **Result list**'te modele sağ tık → **Visualize tree**.

> **İpucu:** Classify ekranında "Classifier output"taki `=== Classifier model (full training set) ===` bölümü, **tüm veriyle** eğitilmiş son modeldir. Performans değerleri ise çapraz doğrulamadan gelir.

## Adım 3 – Modeli Kaydetme

Result list'te modele sağ tık → **Save model** → ör. `iris_j48.model`.

## Adım 4 – Yeni (Etiketsiz) Veri Hazırlama

Yeni çiçekler için bir `.arff` dosyası hazırlanır. **Başlık bölümü (`@relation` ve `@attribute` satırları) eğitim verisiyle birebir aynı olmalıdır**. Sınıf bilinmediği için `?` yazılır. Hazır dosya: [`data/iris_yeni_test.arff`](https://github.com/erkanozhan/machinelearning/blob/main/data/iris_yeni_test.arff)

```text
% Gerçek türler (kontrol amaçlı): setosa, virginica, versicolor, setosa, virginica
@relation iris-yeni-test

@attribute sepallength numeric
@attribute sepalwidth numeric
@attribute petallength numeric
@attribute petalwidth numeric
@attribute class {Iris-setosa,Iris-versicolor,Iris-virginica}

@data
5.1,3.5,1.4,0.2,?
6.0,2.2,5.0,1.5,?
7.0,3.2,4.7,1.4,?
4.9,3.0,1.4,0.2,?
6.3,3.3,6.0,2.5,?
```

> **Dikkat:** ARFF'de yorum satırı `%` ile **satır başında** başlar. Veri satırlarının sonuna yorum eklemeyin; WEKA dosyayı okuyamayabilir.

## Adım 5 – Kaydedilmiş Modelle Tahmin

**Senaryo A – WEKA hâlâ açık:**
1. Classify → Test options → **Supplied test set** → **Set…** → `iris_yeni_test.arff`.
2. **More options…** → **Output predictions** → `PlainText`.
3. Result list'te modele sağ tık → **Re-evaluate model on current test set**.
4. Çıktıda her örnek için tahmin edilen sınıf ve olasılığı (`prediction`) görünür.

**Senaryo B – WEKA kapatılıp yeniden açıldı:**
1. Explorer → Preprocess'te herhangi bir `.arff` dosyası yükleyin (Classify sekmesi veri yüklenmeden aktif olmaz).
2. Classify → Result list'te boş alana sağ tık → **Load model** → `iris_j48.model`.
3. Senaryo A'daki 1–4. adımlar.

> **Gerçek bir uygulamada** model, programın içine gömülür. Bu depodaki [`application/`](https://github.com/erkanozhan/machinelearning/tree/main/application) klasöründe WEKA modelini Java masaüstü (Swing) ve Spring Boot web uygulamasında kullanan örnek projeler bulunur.

## Ek: WEKA ile R Bağlantısı (İleri Seviye)

WEKA'nın kolay arayüzü R'ın istatistik ve görselleştirme gücüyle birleştirilebilir. (Geliştiricilerin anlatımı: [YouTube](https://www.youtube.com/watch?v=EGwHXC3baWU))
1. R ve WEKA'yı kurun. R konsolunda: `install.packages("rJava")`.
2. Ortam değişkenleri (Windows: *Sistem → Gelişmiş sistem ayarları → Ortam Değişkenleri*):
   - `R_HOME` = R'ın kurulu olduğu klasör (ör. `C:\Program Files\R\R-4.x.x`)
   - `R_LIBS_USER` = kullanıcı paket klasörü (ör. `C:\Users\<Kullanıcı>\Documents\R\win-library\4.x`)
   - `PATH`'e `R.exe`'nin bulunduğu `bin\x64` klasörünü ekleyin.
3. GUI Chooser → **Tools → Package manager** → `RPlugin` paketini kurun.
4. WEKA'yı yeniden başlatın. Artık **R Console** kullanılabilir ve `MLRClassifier` ile R'daki modeller WEKA içinden çalıştırılabilir.

---

[← 14. Yapay Sinir Ağlarına Giriş](14-yapay-sinir-aglari.md) · [İçindekiler](../Ders_Notu.md) · [16. Topluluk Öğrenmesi →](16-topluluk-ogrenmesi.md)
