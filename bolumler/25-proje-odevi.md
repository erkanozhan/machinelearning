# 25. Kapsamlı Uygulama (Proje Ödevi)

Bu uygulama dersin neredeyse tüm konularını kapsar. Bunu baştan sona yapabiliyorsanız önemli bir yol kat etmişsiniz demektir.

**Veri:** [`data/insanlar.csv`](https://github.com/erkanozhan/machinelearning/blob/main/data/insanlar.csv) (öznitelikler: `ad, yas, boy, kilo, ayak_no`)

**Görevler:**
1. **Veriyi tanıyın (WEKA Preprocess):** Dosyayı açın; her özniteliğin türünü, min/max/ortalama/std değerlerini ve eksik değer olup olmadığını raporlayın.
2. **Kimlik sütununu çıkarın:** `ad` bir **kimlik** sütunudur ve hiçbir genellenebilir bilgi taşımaz. `unsupervised → attribute → Remove` ile çıkarın. *(Çıkarmazsanız ne olur? Tartışın.)*
3. **Kümeleme ile etiket üretin:** `unsupervised → attribute → AddCluster` filtresi, `clusterer = SimpleKMeans`, `numClusters = 3`. Her kişi için yeni bir `cluster` sütunu oluşur. Küme merkezlerini yorumlayın: Kümeler hangi tür insan gruplarını temsil ediyor?
4. **Sınıflandırma:** `cluster` sütununu hedef seçin. **Classify** sekmesinde 10 katlı çapraz doğrulamayla en az 5 algoritma deneyin (ZeroR, J48, NaiveBayes, IBk, SMO, RandomForest, MultilayerPerceptron…). Doğruluk, **Kappa**, F1 ve ROC Area değerlerini bir tabloda karşılaştırın. En yüksek Kappa değerini veren algoritmayı belirleyin.
5. **İstatistiksel karşılaştırma:** Aynı algoritmaları **Experimenter**'da (10 tekrar × 10 kat) çalıştırıp farkların anlamlı olup olmadığını test edin.
6. **Öznitelik seçimi:** **Select attributes** sekmesinde `InfoGainAttributeEval + Ranker` ve `CfsSubsetEval + BestFirst` ile en önemli öznitelikleri bulun. Seçilen özniteliklerle en iyi modeli yeniden çalıştırın; başarı değişti mi?
7. **Eleştirel düşünme:** Kümeleri K-Means ile ürettiğimiz için etiketler, en yakın merkeze göre belirlendiği için öznitelik uzayında **doğrusal sınırlarla ayrılmış bölgelere** (Voronoi hücreleri) karşılık gelir. Hangi algoritmaların bu yapıyı öğrenmeye daha yatkın olduğunu tartışın. Bu yöntemin (kümeleme → sınıflandırma) gerçek bir tahmin probleminden farkı nedir?
8. **Raporlama:** [Bölüm 24.3](24-weka-experimenter.md)'teki yapıya uygun kısa bir rapor yazın ve sonuçlarınızı paylaşın.

**Ek alıştırma (Python):** Aynı adımları `pandas` + `scikit-learn` ile tekrarlayın (`KMeans` → etiket, `cross_val_score` ile model karşılaştırma, `SelectKBest` ile seçim). Ön işlemenin Pipeline içinde olmasına dikkat edin. İkinci bir veri seti olarak [`data/musteri_ticaret.csv`](https://github.com/erkanozhan/machinelearning/blob/main/data/musteri_ticaret.csv) (kategorik sütunlar içerir: eğitim durumu, ödeme yöntemi) ile one-hot kodlamayı da uygulayın.

---

[← 24. WEKA Experimenter ve Sonuçların Raporlanması](24-weka-experimenter.md) · [İçindekiler](../Ders_Notu.md) · [Ek A: Sembol Sözlüğü →](ek-a-sembol-sozlugu.md)
