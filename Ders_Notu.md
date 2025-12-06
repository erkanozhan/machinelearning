# Makine Öğrenmesi (Machine Learning) ve Yapay Zeka (AI)
### Maliyete Duyarlı Öğrenme (Cost-Sensitive Learning)

Gençler, şimdiye kadar modellerimizi eğitirken genellikle tek bir hedefe odaklandık: Mümkün olduğunca çok doğru tahminde bulunmak, yani doğruluk oranını (accuracy) maksimize etmek. Ancak gerçek dünyadaki problemlerde her hata aynı ağırlıkta değildir.

Bir tıp doktorunun, hasta bir kişiye yanlışlıkla "sağlıklısın" demesi (yanlış negatif - false negative) ile sağlıklı bir kişiye "emin olmak için bir test daha yapalım" demesi (yanlış pozitif - false positive) arasındaki farkı düşünelim. İlk hata, bir hastanın tedavi edilememesine yol açabilirken, ikincisi en fazla zaman ve kaynak israfına neden olur. İşte bu "hata maliyeti" fikrini, makine öğrenmesi modelimize öğretebiliriz. Bu durumda amacımız artık sadece hata sayısını değil, toplam hata maliyetini minimize etmektir.

#### Weka ile Cost-Sensitive Learning Uygulaması

Bu yaklaşıma Weka'da `CostSensitiveClassifier` adlı bir meta-sınıflandırıcı ile olanak tanınır. "Meta" olarak adlandırılmasının sebebi, kendi başına bir öğrenme algoritması olmasından ziyade, J48 gibi başka bir temel sınıflandırıcıyı sarmalayarak ona maliyet bilinci kazandıran bir üst katman görevi görmesidir. Şimdi, bu aracı nasıl kullanacağımıza bakalım.

**Adımlar:**

1.  **Veri Setini Yükleme**
    *   Weka Explorer arayüzünü açın.
    *   `Preprocess` sekmesinden, `credit-g.arff` gibi hata maliyetlerinin farklı olduğu bir veri setini yükleyin. Bu veri seti, kredi başvurularının "iyi" (good) veya "kötü" (bad) olarak sınıflandırılmasını içerir. Burada, "kötü" bir müşteriye "iyi" diyerek kredi vermek (yanlış pozitif), banka için çok daha maliyetlidir.

2.  **Cost-Sensitive Classifier Seçimi**
    *   `Classify` sekmesine geçin.
    *   `Choose` butonuna tıklayarak açılan menüden `meta` → `CostSensitiveClassifier` yolunu izleyin.
    *   `CostSensitiveClassifier` yapılandırmasını açtığınızda, bir temel sınıflandırıcı (`classifier`) seçmeniz istenir. Buradan `trees` → `J48` gibi bildiğiniz bir algoritmayı seçebilirsiniz.

3.  **Maliyet Matrisini (Cost Matrix) Tanımlama**
    *   Bu sürecin en kritik kısmı **Maliyet Matrisi**'dir. Bu matris, modelimize hangi hatanın ne kadar "pahalıya" mal olacağını bildiren bir tablodur.
    *   `CostSensitiveClassifier` ayarlarında `costMatrix` özelliğini düzenlemek için üzerine tıklayın. Karşınıza genellikle şöyle 2x2'lik bir matris çıkar:
        ```
        [[0.0, 1.0], [1.0, 0.0]]
        ```
    *   Bu matris şöyle yorumlanır: Satırlar **gerçek sınıfı**, sütunlar ise **tahmin edilen sınıfı** temsil eder. Sınıflarımız sırasıyla `(good, bad)` olsun.
        *   `[0,0]` (sol üst): Gerçek `good`, tahmin `good` (Doğru Pozitif). Maliyet 0'dır, çünkü hata yoktur.
        *   `[0,1]` (sağ üst): Gerçek `good`, tahmin `bad` (Yanlış Negatif). Maliyeti 1.0'dir.
        *   `[1,0]` (sol alt): Gerçek `bad`, tahmin `good` (Yanlış Pozitif). Maliyeti 1.0'dir.
        *   `[1,1]` (sağ alt): Gerçek `bad`, tahmin `bad` (Doğru Negatif). Maliyet 0'dır, çünkü hata yoktur.
    *   Şimdi bu matrisi kredi problemimize göre düzenleyelim. Gerçekte "kötü" olan bir müşteriyi "iyi" olarak etiketlemek (yanlış pozitif) bizim için çok daha maliyetli olsun. Bu hatanın maliyetini 10 yapalım. Diğer hatanın maliyeti 1 olarak kalsın. Matrisimiz şöyle görünür:
        ```
        [[0.0, 1.0], [10.0, 0.0]]
        ```
    *   Bu matrisle Weka'ya şunu söylemiş oluruz: "Birinci sınıfta (`good`) hata yapmanın maliyeti 1 birimken, ikinci sınıftaki (`bad`) bir hatanın maliyeti 10 birimdir. Sınıflandırma kararlarını bu maliyetleri göz önünde bulundurarak ver."

4.  **Modeli Eğitme ve Değerlendirme**
    *   `Start` butonuna basarak modeli eğitin.
    *   Sonuçları incelerken sadece doğruluk oranına değil, **Karışıklık Matrisi (Confusion Matrix)**'ne odaklanın.
    *   Modelin, maliyeti 10 birim olan hatayı (bizim örneğimizde `bad` müşteriyi `good` olarak tahmin etme) yapmaktan kaçınmak için daha temkinli davrandığını göreceksiniz. Bu kritik hatanın sayısı azalırken, daha ucuz olan diğer hata türünün sayısı artabilir.
    *   Genel doğruluk oranı belki bir miktar düşebilir, ancak bizim için önemli olan pahalı hataları önlemektir. Hedefimiz en yüksek doğruluğa ulaşmak değil, en düşük toplam maliyetle süreci tamamlamaktır.

Weka, bu maliyet bilincini modele entegre etmek için iki temel strateji sunar:
*   **Reweighting:** Eğitim setindeki örnekleri maliyetlerine göre yeniden ağırlıklandırır. Hatası pahalı olan sınıfa ait örnekleri eğitim sırasında daha "önemli" hale getirir.
*   **Minimize Expected Cost:** Sınıflandırma sırasında her bir tahminin beklenen maliyetini hesaplar ve en düşük maliyetli olanı seçer.

Bu yöntemler, özellikle tıp, finans ve güvenlik gibi alanlarda, bir hatanın sonuçlarının diğerinden çok daha ağır olduğu durumlarda modellerimizi daha akıllı ve amaca yönelik hale getirmemizi sağlar.

## Python ile Maliyete Duyarlı Sınıflandırma Örneği

Aşağıdaki Python kodu, `scikit-learn` kütüphanesini kullanarak maliyete duyarlı sınıflandırmanın adımlarını göstermektedir. Önce standart bir modelin performansını ve maliyetini hesaplayacak, ardından maliyet bilincine sahip iki farklı yaklaşımla (sınıf ağırlıklandırma ve karar eşiği optimizasyonu) karşılaştıracağız.

```python
# -*- coding: utf-8 -*-
# Maliyet duyarlı sınıflandırmanın uçtan uca bir örneği.

import numpy as np
# Sayısal işlemler için temel kütüphane.

from sklearn.datasets import make_classification
# Dengesiz bir sınıflandırma veri seti oluşturmak için.

from sklearn.model_selection import train_test_split
# Veriyi eğitim ve test setlerine ayırmak için.

from sklearn.linear_model import LogisticRegression
# Sınıflandırma için kullanılacak Lojistik Regresyon modeli.

from sklearn.metrics import confusion_matrix, classification_report
# Model performansını ölçmek için karışıklık matrisi ve raporlama.

# ===== 0) Tekrarlanabilirlik için rastgelelik kontrolü =====
rng = np.random.RandomState(42)

# ===== 1) Dengesiz Veri Seti Oluşturma =====
X, y = make_classification(
    n_samples=4000,         # Toplam örnek sayısı.
    n_features=20,          # Öznitelik sayısı.
    n_informative=4,        # Bilgi taşıyan (anlamlı) öznitelik sayısı.
    n_redundant=2,          # Başka özniteliklerden türetilmiş öznitelik sayısı.
    n_clusters_per_class=1, # Her sınıf için küme sayısı.
    weights=[0.95, 0.05],   # Sınıf 0 (%95) ve Sınıf 1 (%5) oranı, belirgin bir dengesizlik oluşturur.
    flip_y=0.01,            # Etiketlerdeki gürültü oranı.
    class_sep=1.0,          # Sınıflar arası ayrımın zorluk derecesi.
    random_state=rng
)

# ===== 2) Veriyi Eğitim ve Test Setlerine Ayırma =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,  # Verinin %30'u test için ayrılır.
    stratify=y,      # Sınıf oranları hem eğitim hem de test setinde korunur.
    random_state=rng
)

# ===== 3) Hata Maliyetlerinin Tanımlanması =====
# Gerçekte negatif olan bir örneği pozitif tahmin etmenin maliyeti.
C_FP = 1.0
# Gerçekte pozitif olan bir örneği negatif tahmin etmenin maliyeti (daha pahalı).
C_FN = 10.0

# ===== 4) Baseline Model: Standart Lojistik Regresyon =====
baseline_clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=rng)
baseline_clf.fit(X_train, y_train)
y_pred_baseline = baseline_clf.predict(X_test)

cm_base = confusion_matrix(y_test, y_pred_baseline, labels=[0, 1])
TN_b, FP_b, FN_b, TP_b = cm_base.ravel()

total_cost_b = FP_b * C_FP + FN_b * C_FN

print("=== Baseline (Maliyetsiz) Lojistik Regresyon ===")
print("Karışıklık Matrisi:\n", cm_base)
print(f"Gerçek Negatif (TN): {TN_b}, Sahte Pozitif (FP): {FP_b}")
print(f"Sahte Negatif (FN): {FN_b}, Gerçek Pozitif (TP): {TP_b}")
print(f"Toplam Maliyet (FP*{C_FP} + FN*{C_FN}): {total_cost_b:.1f}\n")
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_baseline, target_names=["negatif(0)", "pozitif(1)"]))

# ===== 5) Yaklaşım 1: Sınıf Ağırlıklandırma (Class Weighting) =====
# 'balanced' modu, sınıf ağırlıklarını örnek sayısıyla ters orantılı olarak ayarlar.
# Alternatif olarak, class_weight={0: 1, 1: 10} gibi manuel bir atama da yapılabilir.
cost_clf = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced",
    random_state=rng
)
cost_clf.fit(X_train, y_train)
y_pred_cost = cost_clf.predict(X_test)

cm_cost = confusion_matrix(y_test, y_pred_cost, labels=[0, 1])
TN_c, FP_c, FN_c, TP_c = cm_cost.ravel()

total_cost_c = FP_c * C_FP + FN_c * C_FN

print("\n=== Maliyete Duyarlı (class_weight='balanced') Lojistik Regresyon ===")
print("Karışıklık Matrisi:\n", cm_cost)
print(f"Gerçek Negatif (TN): {TN_c}, Sahte Pozitif (FP): {FP_c}")
print(f"Sahte Negatif (FN): {FN_c}, Gerçek Pozitif (TP): {TP_c}")
print(f"Toplam Maliyet (FP*{C_FP} + FN*{C_FN}): {total_cost_c:.1f}\n")
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_cost, target_names=["negatif(0)", "pozitif(1)"]))

# ===== 6) Yaklaşım 2: Karar Eşiği Optimizasyonu (Threshold Tuning) =====
# Modelin tahmin olasılıklarını al (pozitif sınıf için).
y_proba = cost_clf.predict_proba(X_test)[:, 1]

def calculate_cost(y_true, y_pred):
    """Verilen tahminler için toplam maliyeti hesaplar."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    return FP * C_FP + FN * C_FN

# Farklı eşik değerleri için maliyetleri hesapla.
thresholds = np.linspace(0.05, 0.95, 19)
costs = [calculate_cost(y_test, (y_proba >= t).astype(int)) for t in thresholds]

# En düşük maliyeti veren eşiği bul.
best_t_idx = np.argmin(costs)
best_t, best_cost = thresholds[best_t_idx], costs[best_t_idx]

print("\n=== Karar Eşiği Optimizasyonu (Maliyete Göre) ===")
for t, c in zip(thresholds, costs):
    print(f"Eşik={t:.2f} → Toplam Maliyet={c:.1f}")
print(f"\nEn iyi eşik: {best_t:.2f} (Toplam Maliyet={best_cost:.1f})")

# ===== 7) Optimal Eşiğe Göre Nihai Performans =====
y_best_pred = (y_proba >= best_t).astype(int)
cm_best = confusion_matrix(y_test, y_best_pred, labels=[0, 1])
TN_o, FP_o, FN_o, TP_o = cm_best.ravel()
total_cost_o = FP_o * C_FP + FN_o * C_FN

print("\n=== Optimal Eşik ile Performans ===")
print("Karışıklık Matrisi:\n", cm_best)
print(f"Gerçek Negatif (TN): {TN_o}, Sahte Pozitif (FP): {FP_o}")
print(f"Sahte Negatif (FN): {FN_o}, Gerçek Pozitif (TP): {TP_o}")
print(f"Toplam Maliyet (FP*{C_FP} + FN*{C_FN}): {total_cost_o:.1f}\n")
print("Sınıflandırma Raporu (Optimal Eşik):")
print(classification_report(y_test, y_best_pred, target_names=["negatif(0)", "pozitif(1)"]))

# ===== 8) Sonuçların Karşılaştırılması =====
print("\n=== Özet Karşılaştırma ===")
print(f"Baseline Toplam Maliyet:       {total_cost_b:.1f}")
print(f"Sınıf Ağırlıklandırma Maliyeti: {total_cost_c:.1f}")
print(f"Optimal Eşik Maliyeti:         {total_cost_o:.1f}")
```
