# =============================================================================
# R ile Nested (İç İçe) Çapraz Doğrulama
# Ders notu bölümü: "19. Optimizasyon ve Hiperparametre Ayarlama" → Nested CV
#
# Dış döngü (10 katman): performansı ölçer
# İç döngü  (5 katman) : her dış katmanın EĞİTİM kısmında Grid Search yapar
# Gerekli paketler:  install.packages(c("caret", "glmnet", "mlbench"))
# =============================================================================

library(caret)
library(mlbench)
data(PimaIndiansDiabetes)
veri <- PimaIndiansDiabetes

set.seed(42)                                                      # Tekrarlanabilirlik
dis_katmanlar <- createFolds(veri$diabetes, k = 10)               # Tabakalı 10 katman (test indeksleri)
dis_sonuclar <- numeric(length(dis_katmanlar))                    # Her dış katmanın doğruluğu burada tutulacak

parametre_grid <- expand.grid(alpha = c(0, 0.5, 1),               # İç döngüde aranacak ızgara
                              lambda = c(0.001, 0.01, 0.1))

for (i in seq_along(dis_katmanlar)) {
  test_indeks <- dis_katmanlar[[i]]                               # i. katman test
  egitim_veri <- veri[-test_indeks, ]                             # Kalan 9 katman eğitim
  test_veri   <- veri[test_indeks, ]

  model <- train(                                                 # İÇ DÖNGÜ: sadece egitim_veri üzerinde 5 katlı CV + Grid Search
    diabetes ~ ., data = egitim_veri, method = "glmnet",
    trControl = trainControl(method = "cv", number = 5),
    tuneGrid = parametre_grid,
    preProcess = c("center", "scale")
  )

  tahmin <- predict(model, test_veri)                             # DIŞ DÖNGÜ: hiç görülmemiş test katmanında tahmin
  dis_sonuclar[i] <- mean(tahmin == test_veri$diabetes)           # Doğruluk
  cat(sprintf("Dış katman %2d: doğruluk = %.3f  (seçilen alpha=%.1f, lambda=%.3f)\n",
              i, dis_sonuclar[i], model$bestTune$alpha, model$bestTune$lambda))
}

cat(sprintf("\nNested CV doğruluğu: %.3f ± %.3f\n", mean(dis_sonuclar), sd(dis_sonuclar)))
