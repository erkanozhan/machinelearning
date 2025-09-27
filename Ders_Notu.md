# Makine Öğrenmesi (Machine Learning) ve Yapay Zeka (AI)

---

### 1. Temel Kavramlar ve Avantaj

**Yapay Zeka (AI)**, İnsanın sahip olduğu **deneyim** ve **tecrübeyi** bilgisayarlara aktarılmasının yollarını inceleyen bilim dalının en genel adıdır. **Makine Öğrenmesi** ise bu amaç için kendine özgü teknikleri barındıran yapay zekanın bir **alt koludur**.

Bilgisayarlar **dijital** kullanan makinelerdir. İnsanların sahip olduğu tecrübe, deneyim, uzmanlık, yetenekler ve kabiliyetlerin bu dil yardımıyla onlara aktarılması, bu alanın en büyük **avantajıdır**.

### 2. Veri ve Bilgi Hiyerarşisi
**Datum**, Latince kökenli bir kelime olup, "verilen şey" veya "gerçek" anlamına gelir. İngilizcede "data" kelimesinin tekil hali olarak kullanılır. Genellikle tek bir gözlemi, ölçümü veya gerçeği ifade ederken, "data" ise bu tekil gerçeklerin çoğulunu, yani bir koleksiyonunu temsil eder. Bu ayrım, özellikle bilimsel ve teknik metinlerde verinin temel birimini vurgulamak için önemlidir.

**Veri**, bir nesne, varlık, durum, olay hakkında nitel ya da nicel bulguların (gerçeklerin) belirli bir sistematiğe göre kayıt altına alınmış haline denilir.

Veri, bilgiye dönüşürken bir hiyerarşiden geçer:

$$\text{Raw Data (İşlenmemiş Veri)} \rightarrow \text{Data (Veri)} \rightarrow \text{Knowledge (Bilgi)}$$

* **İşlenmemiş Veri Örn.:** Yeni doğan bir bebeğin ağırlığı (Henüz bilinmiyor ancak bir ağırlığı var).

* **Veri Örn.:** Bir metriğe (örn. m, kg, inç, galon vb.) göre bu bebeğin ağırlığının ölçülüp bir yere kayıt edilmesi.

* **Bilgi (information) Örn.:** Bebeğin kilosu vb. özelliklerini kullanarak elde edilen gerçeklerdir. Örn: Çorlu'da Mart ayında doğan bebeklerin kilo ortalaması 2.8 kg'dır.

* **Knowledge (Anlamlı-İşe yarar bilgi):** Büyük ve karmaşık verilerden, ilk bakışta fark edilmeyen, daha önce elde edilmemiş, işe yarar ve anlamlı gerçeklerdir.
    * **Yapay Zeka (AI) ile Elde Edilen Bilgi Örn.:** Bir yapay zeka sistemi, yeni doğan bebeklerin ağlama seslerini analiz ederek, ağlamanın açlık, uykusuzluk veya rahatsızlık gibi farklı nedenlerini ayırt edebilir. Bu sayede ebeveynlere, bebeğin neden ağladığına dair daha hızlı ve doğru bir tahmin sunarak, bebeğin ihtiyaçlarına daha etkili yanıt vermelerine yardımcı olabilir.

### 3. Süreç ve İlişkili Disiplinler

Makine öğrenmesi **İstatistik** ve **matematiksel teorileri** kullanır.

#### Süreç Akışı (Model)
```mermaid
graph LR
    A[Girdi] --> B["ML Model"]
    B --> C[Çıktı]
```

#### İlişkili Alanların Bazıları

* **Veri Madenciliği (Mining)**
* **Temel Bilimler:** İstatistik, Matematik
* **Mühendislik Dalları:** Bilg. Müh., Elek. Müh.
* **Uygulamalar ve Teknikler:** Finans, Neural Network, Google Translate, Genetik Algoritmalar, High Performance Computing.