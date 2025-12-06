# Makine Öğrenmesi (Machine Learning) ve Yapay Zeka (AI)

### Weka ile Cost-Sensitive Learning Uygulaması

Gençler, şimdiye kadar modellerimizi eğitirken genellikle tek bir hedefe odaklandık: Mümkün olduğunca çok doğru tahminde bulunmak, yani doğruluk oranını (accuracy) maksimize etmek. Ancak gerçek hayatta her hata aynı ağırlıkta değildir.

Bir doktorun, hasta bir kişiye yanlışlıkla "sağlıklısın" demesi (yanlış negatif) ile sağlıklı bir kişiye "emin olamadım, tekrar test yapalım" demesi (yanlış pozitif) arasındaki farkı düşünün. İlk hata, bir hayatı tehlikeye atabilirken, ikincisi sadece biraz zaman ve para kaybına yol açar. İşte bu "hata maliyeti" fikrini, makine öğrenmesi modelimize öğretebiliriz. Amacımız artık sadece hata sayısını değil, toplam hata maliyetini minimize etmektir.

Bu iş için Weka'da `CostSensitiveClassifier` adında bir meta-sınıflandırıcı bulunur. "Meta" dememizin sebebi, tek başına bir algoritma olmasından ziyade, J48 gibi başka bir temel sınıflandırıcıyı sarmalayarak ona maliyet bilinci kazandıran bir üst katman olmasıdır. Şimdi, bu aracı nasıl kullanacağımıza bakalım.

#### Adımlar:

1.  **Veri Setini Yükleme**
    *   Weka Explorer arayüzünü açın.
    *   `Preprocess` sekmesinden, `credit-g.arff` gibi dengesiz dağılıma sahip veya hata maliyetlerinin farklı olduğu bir veri setini yükleyin. Bu veri seti, kredi başvurularının "iyi" veya "kötü" olarak sınıflandırılmasını içerir. Burada, "kötü" bir müşteriye "iyi" demek (yanlış negatif), banka için çok daha maliyetlidir.

2.  **Cost-Sensitive Classifier Seçimi**
    *   `Classify` sekmesine geçin.
    *   `Choose` butonuna tıklayarak açılan menüden `meta` → `CostSensitiveClassifier` yolunu izleyin.
    *   `CostSensitiveClassifier` ayarlarını açtığınızda, bir temel sınıflandırıcı (`classifier`) seçmeniz istenir. Buradan `trees` → `J48` gibi bildiğiniz bir algoritmayı seçebilirsiniz.

3.  **Maliyet Matrisini Tanımlama**
    *   Bu sürecin kalbi, **Maliyet Matrisi (Cost Matrix)**'dir. Bu matris, modelimize hangi hatanın ne kadar "pahalıya" patlayacağını söylediğimiz bir tarifnamedir.
    *   `CostSensitiveClassifier` ayarlarında `costMatrix` özelliğini bulun ve düzenlemek için üzerine tıklayın. Karşınıza genellikle şöyle bir 2x2'lik matris çıkar:
        ```
        [[0.0, 1.0], [1.0, 0.0]]
        ```
    *   Bu matris şöyle okunur: Satırlar gerçek sınıfı, sütunlar ise tahmin edilen sınıfı temsil eder. Sınıflarımız `(good, bad)` olsun.
        *   `[0,0]` (sol üst): Gerçek `good`, tahmin `good` (Doğru Pozitif). Maliyet 0.
        *   `[0,1]` (sağ üst): Gerçek `good`, tahmin `bad` (Yanlış Negatif). Maliyet 1.0.
        *   `[1,0]` (sol alt): Gerçek `bad`, tahmin `good` (Yanlış Pozitif). Maliyet 1.0.
        *   `[1,1]` (sağ alt): Gerçek `bad`, tahmin `bad` (Doğru Negatif). Maliyet 0.
    *   Şimdi bu matrisi problemimize göre düzenleyelim. Gerçekte "kötü" olan bir müşteriyi "iyi" olarak etiketlemek (yanlış pozitif) bizim için çok daha maliyetli. Bu hatanın maliyetini 10 yapalım. Matrisimiz şöyle görünür:
        ```
        [[0.0, 1.0], [10.0, 0.0]]
        ```
    *   Bu matrisle Weka'ya diyoruz ki: "Birinci sınıftaki bir hatanın maliyeti 1 birimken, ikinci sınıftaki bir hatanın maliyeti 10 birimdir. Kararlarını buna göre ver."

4.  **Modeli Eğitme ve Değerlendirme**
    *   `Start` butonuna basarak modeli eğitin.
    *   Sonuçları incelerken sadece doğruluk oranına bakmayın. Asıl odaklanmanız gereken yer **Karışıklık Matrisi (Confusion Matrix)**'dir.
    *   Modelin, maliyeti 10 birim olan hatayı (bizim örneğimizde `bad` müşteriyi `good` olarak tahmin etme) yapmaktan kaçınmak için daha temkinli davrandığını göreceksiniz. Bu hatanın sayısı azalırken, daha ucuz olan diğer hata türünün sayısı artabilir.
    *   Genel doğruluk oranı belki bir miktar düşebilir, ancak bizim için kritik olan pahalı hataları önlemiş oluruz. Unutmayın, hedefimiz artık en çok doğruyu bulmak değil, en düşük maliyetle işi bitirmektir.

Weka, bu maliyet bilincini modele entegre etmek için iki temel strateji sunar:
*   **Reweight Training Instances:** Eğitim setindeki örnekleri maliyetlerine göre yeniden ağırlıklandırır. Yani, hatası pahalı olan sınıfa ait örnekleri eğitim sırasında daha "önemli" hale getirir.
*   **Minimize Expected Cost:** Sınıflandırma sırasında her bir tahminin beklenen maliyetini hesaplar ve en düşük maliyetli olanı seçer.

Bu yöntemler, özellikle tıp, finans, güvenlik gibi alanlarda, bir hatanın sonuçlarının diğerinden çok daha ağır olduğu durumlarda modellerimizi daha akıllı ve amaca yönelik hale getirmemizi sağlar.
