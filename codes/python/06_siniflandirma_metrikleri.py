# -*- coding: utf-8 -*-
"""
06 - Sınıflandırma Performans Ölçütleri
=======================================
Ders notu bölümü: "9. Performans Ölçütleri" → Sınıflandırma

Bölüm A: Ders notundaki 1000 kişilik örneğin (TP=350, FN=150, FP=250, TN=250)
         tüm metriklerini elle ve scikit-learn ile hesaplar.
Bölüm B: Dengesiz bir veri setinde Lojistik Regresyon eğitir;
         sınıflandırma raporu, karışıklık matrisi, ROC ve Precision-Recall eğrilerini çizer.

Çalıştırma:  python 06_siniflandirma_metrikleri.py
"""

import numpy as np                                            # Sayısal işlemler
import matplotlib.pyplot as plt                               # Grafikler
from sklearn.datasets import make_classification              # Yapay veri üretimi
from sklearn.model_selection import train_test_split          # Holdout bölme
from sklearn.linear_model import LogisticRegression           # Sınıflandırıcı
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,                 # Karışıklık matrisi ve çizimi
    accuracy_score, precision_score, recall_score, f1_score,  # Temel metrikler
    cohen_kappa_score, matthews_corrcoef,                     # Şansa göre düzeltilmiş metrikler
    classification_report,                                    # Hepsini tablo halinde verir
    roc_curve, roc_auc_score,                                 # ROC eğrisi ve AUC
    precision_recall_curve, average_precision_score,          # PR eğrisi ve ortalama kesinlik
)

# ===========================================================================
# BÖLÜM A — Ders notundaki örnek
# ===========================================================================
TP, FN, FP, TN = 350, 150, 250, 250                           # Karışıklık matrisinin dört hücresi
N = TP + FN + FP + TN                                         # Toplam = 1000

accuracy = (TP + TN) / N                                      # Tüm tahminlerin doğru oranı
precision = TP / (TP + FP)                                    # "Pozitif dediklerimin ne kadarı doğru?"
recall = TP / (TP + FN)                                       # "Gerçek pozitiflerin ne kadarını yakaladım?" (TPR)
specificity = TN / (TN + FP)                                  # "Gerçek negatiflerin ne kadarını doğru bildim?" (TNR)
fpr = FP / (FP + TN)                                          # Yanlış alarm oranı = 1 - specificity
f1 = 2 * precision * recall / (precision + recall)            # Precision ve recall'un harmonik ortalaması

p_o = accuracy                                                # Gözlenen uyum
p_e = ((TP + FN) / N) * ((TP + FP) / N) + ((FP + TN) / N) * ((FN + TN) / N)   # Şans eseri beklenen uyum
kappa = (p_o - p_e) / (1 - p_e)                               # Cohen's Kappa

mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))  # Matthews korelasyon katsayısı

print("=== A) Ders notundaki örnek (elle) ===")
for ad, deger in [("Accuracy", accuracy), ("Precision", precision), ("Recall (TPR)", recall),
                  ("Specificity (TNR)", specificity), ("FPR", fpr), ("F1", f1),
                  ("Pe (şans uyumu)", p_e), ("Kappa", kappa), ("MCC", mcc)]:
    print(f"{ad:<18}: {deger:.4f}")

# Aynı sonuçları sklearn ile doğrulayalım: matrise karşılık gelen etiket dizilerini üretiyoruz
y_gercek = np.array([1] * (TP + FN) + [0] * (FP + TN))        # 500 pozitif, 500 negatif
y_tahmin = np.array([1] * TP + [0] * FN + [1] * FP + [0] * TN)
print("sklearn kontrol → acc={:.3f} prec={:.3f} rec={:.3f} f1={:.3f} kappa={:.3f} mcc={:.3f}".format(
    accuracy_score(y_gercek, y_tahmin), precision_score(y_gercek, y_tahmin), recall_score(y_gercek, y_tahmin),
    f1_score(y_gercek, y_tahmin), cohen_kappa_score(y_gercek, y_tahmin), matthews_corrcoef(y_gercek, y_tahmin)))
print()

# ===========================================================================
# BÖLÜM B — Dengesiz veri setinde gerçek bir model
# ===========================================================================
X, y = make_classification(
    n_samples=1000,          # 1000 örnek
    n_features=2,            # 2 öznitelik (görselleştirilebilir olsun)
    n_informative=2,         # İkisi de bilgi taşıyor
    n_redundant=0,           # Türetilmiş (gereksiz) öznitelik yok
    weights=[0.9, 0.1],      # %90 negatif, %10 pozitif → dengesiz
    flip_y=0,                # Etiket gürültüsü yok
    random_state=42,         # Tekrarlanabilirlik
)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)  # stratify: oranları koru

model = LogisticRegression().fit(X_tr, y_tr)                  # Modeli eğit
y_pred = model.predict(X_te)                                  # Kesin sınıf kararları (eşik = 0.5)
y_skor = model.predict_proba(X_te)[:, 1]                      # Pozitif sınıf olasılıkları (ROC/PR için gerekli)

print("=== B) Dengesiz veri: sınıflandırma raporu ===")
print(classification_report(y_te, y_pred, target_names=["Negatif (0)", "Pozitif (1)"], digits=3))
print(f"Hiç düşünmeden herkese 'negatif' diyen modelin doğruluğu: {1 - y_te.mean():.3f}  ← accuracy tek başına yanıltıcıdır")
print(f"ROC-AUC = {roc_auc_score(y_te, y_skor):.3f}")
print(f"PR-AUC (Average Precision) = {average_precision_score(y_te, y_skor):.3f}  (rastgele model ≈ {y_te.mean():.2f})")

# Eşiği değiştirince precision/recall nasıl değişir?
print("\nEşik   Precision  Recall")
for esik in [0.2, 0.3, 0.5, 0.7]:
    t = (y_skor >= esik).astype(int)                          # Olasılığı eşikle karşılaştırıp sınıfa çevir
    print(f"{esik:<6} {precision_score(y_te, t, zero_division=0):>9.3f} {recall_score(y_te, t):>7.3f}")

# --- Grafikler ---
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred), display_labels=["Negatif", "Pozitif"]).plot(ax=ax[0], cmap="Blues", colorbar=False)
ax[0].set_title("Karışıklık Matrisi")
ax[0].set_xlabel("Tahmin edilen")
ax[0].set_ylabel("Gerçek")

fpr_e, tpr_e, _ = roc_curve(y_te, y_skor)                     # Tüm eşikler için FPR ve TPR
ax[1].plot(fpr_e, tpr_e, lw=2, label=f"Model (AUC = {roc_auc_score(y_te, y_skor):.2f})")
ax[1].plot([0, 1], [0, 1], "--", color="gray", label="Rastgele (AUC = 0.5)")
ax[1].set_xlabel("Sahte Pozitif Oranı (FPR)")
ax[1].set_ylabel("Gerçek Pozitif Oranı (TPR)")
ax[1].set_title("ROC Eğrisi")
ax[1].legend(loc="lower right")

prec_e, rec_e, _ = precision_recall_curve(y_te, y_skor)       # Tüm eşikler için precision ve recall
ax[2].plot(rec_e, prec_e, lw=2, label=f"Model (AP = {average_precision_score(y_te, y_skor):.2f})")
ax[2].axhline(y_te.mean(), ls="--", color="gray", label="Rastgele")
ax[2].set_xlabel("Recall")
ax[2].set_ylabel("Precision")
ax[2].set_title("Precision-Recall Eğrisi")
ax[2].legend()

plt.tight_layout()
plt.show()
