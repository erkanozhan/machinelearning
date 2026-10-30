# Iris Tahmin – Masaüstü Uygulaması

WEKA ile eğitilmiş bir çok katmanlı algılayıcı (MLP) modelini kullanan Java Swing uygulamasıdır. Kullanıcı bir iris çiçeğinin dört ölçümünü (sepal uzunluğu, sepal genişliği, petal uzunluğu, petal genişliği) girer; uygulama çiçeğin türünü (setosa, versicolor, virginica) tahmin eder.

## Çalıştırma

`IrisPredictor.jar` dosyası, WEKA kütüphanesini ve eğitilmiş modeli içinde barındırır; ayrıca bir kurulum gerekmez. Bilgisayarda Java 17 veya üzeri yüklü olmalıdır (WEKA 3.8.6 kurulumuyla gelen Java da kullanılabilir).

```
java -jar IrisPredictor.jar
```

Windows'ta dosyaya çift tıklamak da yeterlidir.

## Kaynak kod

| Dosya | İçerik |
| :--- | :--- |
| `IrisPredictor/src/IrisPredictorForm.java` | Formun davranışı: modelin yüklenmesi, girilen değerlerden bir WEKA örneği (`Instance`) oluşturulması ve tahmin |
| `IrisPredictor/src/IrisPredictorForm.form` | IntelliJ IDEA GUI Designer ile hazırlanmış form tasarımı |
| `IrisPredictor/src/resources/MLP_iris_model.model` | WEKA Explorer'da eğitilip kaydedilmiş MLP modeli |

Projeyi derlemek için IntelliJ IDEA kullanılır: `IrisPredictor` klasörü proje olarak açılır, WEKA kurulum klasöründeki `weka.jar` **File → Project Structure → Libraries** bölümünden kütüphane olarak eklenir. Form dosyası IntelliJ'e özgü olduğundan başka bir geliştirme ortamında doğrudan derlenmez.

Aynı modelin web üzerinden kullanıldığı sürüm için bkz. [`../../IrisMaven`](../../IrisMaven).
