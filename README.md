# Makine Öğrenmesi Ders Materyalleri

Tekirdağ Namık Kemal Üniversitesi Makine Öğrenmesi dersinin ders notu, uygulama kodları, veri setleri ve animasyonları.

- **Ders notu:** [Ders_Notu.md](Ders_Notu.md)
- **Web sürümü:** [erkanozhan.github.io/machinelearning](https://erkanozhan.github.io/machinelearning/). Aynı not; formüller ve arama ile birlikte.

## Klasörler

| Klasör / dosya | İçerik |
| :--- | :--- |
| [`Ders_Notu.md`](Ders_Notu.md) | Ders notunun tamamı (temel kavramlardan proje ödevine kadar 25 bölüm ve ekler) |
| [`codes/python/`](codes/python) | Bölümlere göre numaralandırılmış Python kodları ([açıklama](codes/README.md)) |
| [`codes/R/`](codes/R) | Hiperparametre arama ve iç içe çapraz doğrulama için R kodları |
| [`data/`](data) | Derse özel veri setleri, CSV ve ARFF biçiminde ([açıklama](data/README.md)) |
| [`animation/`](animation), [`roc_animation.html`](roc_animation.html) | Gradyan inişi, K-Means, iç içe çapraz doğrulama ve ROC eğrisi animasyonları (tarayıcıda açılır) |
| [`application/`](application) | WEKA ile eğitilmiş Iris modelini kullanan Java masaüstü ve web uygulamaları |
| [`images/`](images) | Ders notundaki şekiller, açıklamalı formüller ve diyagramlar |
| [`index.html`](index.html) | Web sürümünü oluşturan docsify sayfası |

## Kurulum

Python kodları için:

```bash
pip install -r codes/requirements.txt
```

Kurulum yapmadan çalıştırmak için kodlar [Google Colab](https://colab.research.google.com)'a yapıştırılabilir. WEKA uygulamaları için [WEKA 3.8](https://ml.cms.waikato.ac.nz/weka) yeterlidir; derste kullanılan `iris.arff`, `diabetes.arff` gibi standart veri setleri WEKA kurulum klasöründeki `data/` dizinindedir.

## Lisans

Bu depodaki içerik [CC0 1.0](LICENSE) kapsamında kamu malıdır.
