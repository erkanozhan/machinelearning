# Makine Öğrenmesi Ders Materyalleri

Tekirdağ Namık Kemal Üniversitesi Makine Öğrenmesi dersinin ders notu, uygulama kodları, veri setleri ve animasyonları.

- **Ders notu:** [Ders_Notu.md](Ders_Notu.md) (giriş ve içindekiler); her bölüm [`bolumler/`](bolumler) klasöründe ayrı bir dosyadır.
- **Web sürümü:** [erkanozhan.github.io/machinelearning](https://erkanozhan.github.io/machinelearning/). Aynı not; bölüm listesi, tüm notta arama, formüller ve doğrudan çalışan animasyonlarla birlikte. Notu okumak için önerilen adres budur.
- **PDF / yazdırma:** [Notun tamamı tek sayfada](https://erkanozhan.github.io/machinelearning/yazdir.html) açılıp Ctrl+P ile "PDF olarak kaydet" seçilebilir. Aynı PDF `python araclar/pdf_olustur.py` komutuyla da üretilir (Chrome veya Edge gerekir).
- **Animasyonlar:** Çalışan sürümlerin listesi [`animation/`](animation) klasörünün açıklamasındadır. Depoda bir `.html` dosyasına tıklandığında GitHub sayfayı çalıştırmaz, yalnızca kaynak kodunu gösterir.

## Klasörler

| Klasör / dosya | İçerik |
| :--- | :--- |
| [`Ders_Notu.md`](Ders_Notu.md) | Ders notunun giriş sayfası ve içindekiler |
| [`bolumler/`](bolumler) | Notun bölümleri: temel kavramlardan proje ödevine kadar 25 bölüm ve ekler, her biri ayrı dosya |
| [`codes/python/`](codes/python) | Bölümlere göre numaralandırılmış Python kodları ([açıklama](codes/README.md)) |
| [`codes/R/`](codes/R) | Hiperparametre arama ve iç içe çapraz doğrulama için R kodları |
| [`data/`](data) | Derse özel veri setleri, CSV ve ARFF biçiminde ([açıklama](data/README.md)) |
| [`animation/`](animation), [`roc_animation.html`](roc_animation.html) | Gradyan inişi, k-NN, K-Means, iç içe çapraz doğrulama ve ROC eğrisi animasyonları (web sitesinde çalışır) |
| [`application/`](application) | WEKA ile eğitilmiş Iris modelini kullanan Java masaüstü ve web uygulamaları |
| [`images/`](images) | Ders notundaki şekiller, açıklamalı formüller ve diyagramlar |
| [`index.html`](index.html), [`_sidebar.md`](_sidebar.md), [`docsify-ayar.js`](docsify-ayar.js), [`docsify-stil.css`](docsify-stil.css) | Web sürümünü oluşturan docsify sayfası, bölüm listesi, ayarları ve stilleri |
| [`yazdir.html`](yazdir.html), [`araclar/pdf_olustur.py`](araclar/pdf_olustur.py) | Notun tamamını tek sayfada gösteren yazdırma sayfası ve PDF üreten betik |

## Kurulum

Python kodları için:

```bash
pip install -r codes/requirements.txt
```

Kurulum yapmadan çalıştırmak için kodlar [Google Colab](https://colab.research.google.com)'a yapıştırılabilir. WEKA uygulamaları için [WEKA 3.8](https://ml.cms.waikato.ac.nz/weka) yeterlidir; derste kullanılan `iris.arff`, `diabetes.arff` gibi standart veri setleri WEKA kurulum klasöründeki `data/` dizinindedir.

## Lisans

Bu depodaki içerik [CC0 1.0](LICENSE) kapsamında kamu malıdır.
