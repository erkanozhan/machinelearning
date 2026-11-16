# Makine Öğrenmesine Giriş (Introduction to Machine Learning)

> **Ders Notu** · Bu not hem konuyla **ilk kez karşılaşan** öğrenciler hem de temel bilgisi olup **derinleşmek isteyenler** için hazırlanmıştır.

**Bu not iki yerden okunabilir; içerik ikisinde de aynıdır:**

| | Adres | Ne zaman tercih edilir? |
| :--- | :--- | :--- |
| **Web sitesi** | [erkanozhan.github.io/machinelearning](https://erkanozhan.github.io/machinelearning/) | Notu okumak ve çalışmak için: Formüller hızlı yüklenir, soldaki listeden bölümler arasında geçilir, tüm notta **arama** yapılabilir, animasyonlar doğrudan çalışır. |
| **GitHub deposu** | [github.com/erkanozhan/machinelearning](https://github.com/erkanozhan/machinelearning) | Kod dosyalarını, veri setlerini ve görselleri indirmek, notun son değişikliklerini takip etmek için. |

> **Dikkat:** GitHub deposunda bir `.html` dosyasına (ör. `animation/` klasöründeki animasyonlar) tıklandığında sayfa çalışmaz, yalnızca kaynak kodu görünür. Animasyonlar notun içindeki bağlantılardan ya da web sitesinden açılmalıdır.

**Notun tamamını yazdırmak veya PDF olarak kaydetmek için:** [Notun tamamı tek sayfada](https://erkanozhan.github.io/machinelearning/yazdir.html) bağlantısı açılır, sayfa yüklendikten sonra tarayıcının yazdırma menüsünden (Ctrl+P) "PDF olarak kaydet" seçilir. PDF, depodaki [`araclar/pdf_olustur.py`](https://github.com/erkanozhan/machinelearning/blob/main/araclar/pdf_olustur.py) betiğiyle de üretilebilir.

---

## Bu Notu Nasıl Okumalısınız?

Notta bazı kısımlar şu etiketlerle işaretlenmiştir:

| Etiket | Anlamı |
| :--- | :--- |
| **(İleri Seviye)** / **Derinleşme** | İlk okumada atlanabilir; konuyu pekiştirmek isteyenler için. Çoğu, tıklayınca açılan kutucukların içindedir. |
| **İpucu** | Sezgi veya günlük hayattan benzetme. |
| **Dikkat** | Sık yapılan hata veya dikkat edilmesi gereken nokta. |
| **Uygulama** | WEKA veya Python ile yapılacak çalışma. |
| **Animasyon** | Tarayıcıda açılan etkileşimli animasyon. Bağlantılar, animasyonun web sitesindeki çalışan sürümüne gider. |
| **Kod** | `codes/` klasöründeki ilgili kod dosyası. |

**Terimler hakkında:** Türkçe kavramların ilk geçtiği yerde parantez içinde İngilizce karşılığı ve varsa kelimenin kökeni verilmiştir. Örnek: **Model** (İng. *model*; Lat. *modulus*: "küçük ölçü"). Kısaltmalar: *İng.* İngilizce, *Lat.* Latince, *Yun.* Yunanca, *Alm.* Almanca, *Fr.* Fransızca. Tüm terimlerin listesi [Ek B](bolumler/ek-b-terimler-sozlugu.md)'dedir.

**Formüller hakkında:** Her önemli formülün altında, formülün **ne anlattığını** açıklayan kısa bir paragraf ve sembollerin **nasıl okunduğunu** ve **ne anlama geldiğini** gösteren bir tablo bulunur. Tüm semboller ayrıca [Ek A: Sembol Sözlüğü](bolumler/ek-a-sembol-sozlugu.md) bölümünde toplanmıştır.

**Kodlar hakkında:** Kısa kodlar anlatımın içinde verilmiştir. Uzun ve uygulama dersinde çalıştırılacak kodlar [`codes/`](https://github.com/erkanozhan/machinelearning/tree/main/codes) klasöründedir. Her kod satırında Türkçe açıklama vardır. Kurulum:

```bash
pip install numpy scipy pandas scikit-learn matplotlib   # Gerekli Python kütüphaneleri
```

> **İpucu:** Bilgisayarınıza hiçbir şey kurmadan [Google Colab](https://colab.research.google.com) üzerinde de tüm Python kodlarını çalıştırabilirsiniz.

---

## İçindekiler

Not, her bölüm ayrı bir sayfa olacak şekilde bölümlere ayrılmıştır. Her bölümün sonunda önceki ve sonraki bölüme geçiş bağlantıları bulunur.

1. [Temel Kavramlar](bolumler/01-temel-kavramlar.md)
2. [Bir Makine Öğrenmesi Projesinin Adımları](bolumler/02-proje-adimlari.md)
3. [Öğrenme Türleri](bolumler/03-ogrenme-turleri.md)
4. [Araçlar: WEKA ve Python](bolumler/04-araclar-weka-python.md)
5. [Veri ve Öznitelikler](bolumler/05-veri-ve-oznitelikler.md)
6. [Lineer Regresyon](bolumler/06-lineer-regresyon.md)
7. [Lojistik Regresyon](bolumler/07-lojistik-regresyon.md)
8. [Model Değerlendirme Yöntemleri](bolumler/08-model-degerlendirme.md)
9. [Performans Ölçütleri](bolumler/09-performans-olcutleri.md)
10. [k-En Yakın Komşu (k-NN)](bolumler/10-k-en-yakin-komsu.md)
11. [Naive Bayes](bolumler/11-naive-bayes.md)
12. [Karar Ağaçları](bolumler/12-karar-agaclari.md)
13. [Destek Vektör Makineleri (SVM)](bolumler/13-destek-vektor-makineleri.md)
14. [Yapay Sinir Ağlarına Giriş](bolumler/14-yapay-sinir-aglari.md)
15. [WEKA ile Uçtan Uca Uygulama: Iris](bolumler/15-weka-iris-uygulamasi.md)
16. [Topluluk Öğrenmesi](bolumler/16-topluluk-ogrenmesi.md)
17. [Kümeleme](bolumler/17-kumeleme.md)
18. [Birliktelik Kuralları](bolumler/18-birliktelik-kurallari.md)
19. [Boyut Azaltma: PCA ve Öznitelik Seçimi](bolumler/19-boyut-azaltma.md)
20. [Optimizasyon ve Hiperparametre Ayarlama](bolumler/20-optimizasyon-hiperparametre.md)
21. [Veri Sızıntısı](bolumler/21-veri-sizintisi.md)
22. [Dengesiz Veri ve Maliyete Duyarlı Öğrenme](bolumler/22-dengesiz-veri.md)
23. [WEKA KnowledgeFlow](bolumler/23-weka-knowledgeflow.md)
24. [WEKA Experimenter ve Sonuçların Raporlanması](bolumler/24-weka-experimenter.md)
25. [Kapsamlı Uygulama (Proje Ödevi)](bolumler/25-proje-odevi.md)

[Ek A: Sembol Sözlüğü](bolumler/ek-a-sembol-sozlugu.md) · [Ek B: Türkçe–İngilizce Terimler](bolumler/ek-b-terimler-sozlugu.md) · [Ek C: Kodlar ve Animasyonlar](bolumler/ek-c-kodlar-ve-animasyonlar.md) · [Kaynaklar](bolumler/kaynaklar.md)
