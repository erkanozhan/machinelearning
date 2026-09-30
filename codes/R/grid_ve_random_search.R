# =============================================================================
# R ile Grid Search ve Random Search (caret paketi)
# Ders notu bölümü: "19. Optimizasyon ve Hiperparametre Ayarlama"
#
# Gerekli paketler:  install.packages(c("caret", "glmnet", "mlbench"))
# =============================================================================

library(caret)                                   # Model eğitimi ve hiperparametre araması için çatı paket
data(PimaIndiansDiabetes, package = "mlbench")   # Pima yerlileri diyabet veri seti (768 kişi, 8 öznitelik)
veri <- PimaIndiansDiabetes                      # Kısa bir isimle kullanalım

set.seed(42)                                     # Tekrarlanabilirlik: CV bölmeleri her çalıştırmada aynı olsun

# ---- Çapraz doğrulama ayarı (5 katlı) ----
kontrol <- trainControl(
  method = "cv",                                 # K-katlı çapraz doğrulama
  number = 5,                                    # K = 5
  verboseIter = FALSE                            # TRUE yaparsanız her adımı ekrana yazar
)

# ---- GRID SEARCH: tüm kombinasyonlar (3 x 3 = 9) ----
parametre_grid <- expand.grid(
  alpha  = c(0, 0.5, 1),                         # 0: Ridge (L2), 1: Lasso (L1), 0.5: Elastic Net
  lambda = c(0.001, 0.01, 0.1)                   # Ceza (regularization) şiddeti
)

model_grid <- train(
  diabetes ~ .,                                  # Hedef: diabetes, girdiler: diğer tüm sütunlar
  data = veri,
  method = "glmnet",                             # Cezalı lojistik regresyon
  trControl = kontrol,
  tuneGrid = parametre_grid,                     # Denenecek ızgara
  preProcess = c("center", "scale")              # Standartlaştırma (her CV katmanında ayrı yapılır → sızıntı yok)
)
print(model_grid)                                # Tüm kombinasyonların CV doğruluğu
print(model_grid$bestTune)                       # En iyi alpha ve lambda

# ---- RANDOM SEARCH: 20 rastgele kombinasyon ----
kontrol_rastgele <- trainControl(method = "cv", number = 5, search = "random")
model_random <- train(
  diabetes ~ ., data = veri, method = "glmnet",
  trControl = kontrol_rastgele,
  tuneLength = 20,                               # Random search'te "kaç deneme yapılacağı"
  preProcess = c("center", "scale")
)
print(model_random$bestTune)
