# Ek A: Sembol Sözlüğü

| Sembol | Okunuşu | Anlamı | İlk geçtiği bölüm |
| :---: | :--- | :--- | :---: |
| $x$, $\mathbf{x}$ | iks, iks vektörü | Girdi / öznitelik (vektörü) | 5.1 |
| $X$ | büyük iks | Öznitelik matrisi ($n \times d$) | 5.1 |
| $x_i$, $x_{ij}$ | iks alt i, iks alt i j | $i$. örnek; $i$. örneğin $j$. özniteliği (alt indis = sıra numarası) | 5.1 |
| $y$, $\hat{y}$ | ye, ye şapka | Gerçek değer, tahmin | 5.1 |
| $n$, $m$ | en, em | Örnek sayısı ($m$: Bölüm 6–7'deki maliyet formüllerinde) | 5.1 |
| $d$ | de | Öznitelik sayısı | 5.1 |
| $\sum$ | sigma (toplam) | Toplama | 5.2 |
| $\mu$, $\bar{x}$ | mü, iks bar | Ortalama (kitle / örneklem) | 5.2 |
| $Q_1$, $Q_3$, $IQR$ | ku bir, ku üç, ay ku ar | Birinci ve üçüncü çeyrek; çeyrekler açıklığı ($Q_3 - Q_1$) | 5.2 |
| $\sigma$, $\sigma^2$ | sigma, sigma kare | Standart sapma, varyans | 5.2 |
| $\lvert a \rvert$ | mutlak değer | İşaretsiz büyüklük | 5.8 |
| $\lvert S \rvert$ | es'in eleman sayısı | Bir kümenin eleman sayısı (mutlak değer değil) | 12.2 |
| $\theta$, $\boldsymbol{\theta}$ | teta | Model parametreleri | 1.5 |
| $h_\theta(x)$ | h teta iks | Hipotez (model) fonksiyonu | 6.1 |
| $\beta_0$, $\beta_1$ | beta sıfır, beta bir | İstatistik kaynaklarında $\theta_0$, $\theta_1$ yerine kullanılan gösterim | 6.1 |
| $J(\theta)$ | ce teta | Maliyet (kayıp) fonksiyonu | 6.2 |
| $\alpha$, $\eta$ | alfa, eta | Öğrenme oranı | 6.4 |
| $\frac{\partial J}{\partial \theta}$, $\nabla J$ | kısmi türev, nabla ce | Eğim, gradyan | 6.4 |
| $:=$, $\leftarrow$ | atanır | Değer atama | 6.4 |
| $X^{T}$ | iks transpoz | Satırları ile sütunları yer değiştirmiş matris | 6.5 |
| $\sigma(z)$ | sigma zet | Sigmoid fonksiyonu | 7.2 |
| $e$ | e | Euler sayısı ≈ 2.718 | 7.2 |
| $e^{-z}$ | e üzeri eksi zet | $1/e^{z}$; $z$ büyüdükçe 0'a yaklaşır | 7.2 |
| $\hat{p}$ | pe şapka | Modelin tahmin ettiği olasılık | 7.2 |
| $\log$, $\log_2$ | logaritma, iki tabanında logaritma | $\log$ bu notta doğal logaritmadır (taban $e$); $\log_2$ sonucu bit cinsinden verir | 7.3 / 12.2 |
| $\xrightarrow{\;\cdot\;}$ | dönüşür; limit | Okun üstündeki işlemle dönüşüm (6.6); $\xrightarrow{n\to\infty}$: $n$ büyüdükçe yaklaşılan değer (8.9) | 6.6 / 8.9 |
| $P(A \mid B)$ | B verildiğinde A'nın olasılığı | Koşullu olasılık | 7.2 / 11.1 |
| $f$, $\hat{f}$ | ef, ef şapka | Gerçek (bilinmeyen) ilişki; modelin öğrendiği tahmin | 8.2 |
| $E[\cdot]$ | beklenen değer | Çok sayıda tekrarda alınacak ortalama | 8.2 |
| $k$, $K$ | ka | Bağlama göre: k-NN'deki komşu sayısı (10), çapraz doğrulamadaki kat sayısı (8.6), K-Means'teki küme sayısı (17.1) | 8.6 |
| $\kappa$ | kappa | Cohen'in Kappa katsayısı | 9.2 |
| $p_o$, $p_e$ | pe o, pe e | Gözlenen uyum (doğruluk); şansla beklenen uyum | 9.2 |
| $R^2$, $r$ | ar kare, küçük ar | Belirlilik katsayısı, korelasyon | 6.8 / 9.4 |
| $\prod$ | pi (çarpım) | Çarpma | 11.2 |
| $\propto$ | orantılıdır | Sabit çarpan dışında eşit | 11.2 |
| $\pi$ | pi | Pi sayısı ≈ 3.14159 (Gauss yoğunluğunda); $\prod$ çarpım işaretiyle karıştırılmamalıdır | 11.4 |
| $H(S)$ | ha es | Entropi | 12.2 |
| $IG$ | bilgi kazancı | Information gain | 12.2 |
| $\lVert \mathbf{w} \rVert$ | norm | Vektör uzunluğu | 13.2 |
| $\mathbf{w}$, $b$ | dabılyu, be | Ağırlık vektörü, sapma (bias) | 13.2 |
| $\mathbf{w}\cdot\mathbf{x}$ | dabılyu nokta iks | İç çarpım: $\sum_j w_j x_j$ | 13.2 |
| $\min_{\mathbf{w}}$ | dabılyu üzerinden minimum | Sağdaki ifadeyi en küçük yapan $\mathbf{w}$ aranır | 13.2 |
| $\xi$ | ksi | Gevşek değişken (SVM) | 13.3 |
| $C$ | ce | SVM ceza katsayısı / maliyet matrisi | 13.3 / 22.2 |
| $\gamma$ | gama | RBF çekirdek parametresi | 8.4 / 13.4 |
| $K(\mathbf{x},\mathbf{z})$ | ka | Çekirdek fonksiyonu | 13.4 |
| $\varphi$ | fi | Özellik dönüşümü | 13.4 |
| $\hat{y}^{(b)}$ | ye şapka üst be | $b$. modelin tahmini (parantezli üst indis kuvvet değildir) | 16.2 |
| $\mathrm{mod}$ | mod | En sık görülen değer (çoğunluk oyu) | 16.2 |
| $\text{argmin}_ h$ | arg min ha | Sağdaki ifadeyi en küçük yapan $h$'nin kendisi (en küçük değer değil) | 16.3 |
| $F_t$, $h_t$, $\nu$ | ef te, ha te, nü | Boosting'de $t$. adımdaki toplam model; o adımda eklenen zayıf model; küçültme (öğrenme) oranı | 16.3 |
| $r_i^{(t)}$ | ar i üst te | $t$. adımda $i$. örneğin artığı (gerçek − tahmin) | 16.3 |
| $C_k$, $\boldsymbol{\mu}_k$ | ce ka, mü ka | $k$. küme ve merkezi | 17.1 |
| $a(i)$, $b(i)$, $s(i)$ | a i, be i, es i | Siluet: kendi kümesine ortalama uzaklık; en yakın diğer kümeye ortalama uzaklık; siluet skoru | 17.1 |
| $\max\lbrace a, b \rbrace$ | a ile be'nin büyüğü | İki değerden büyük olanı | 17.1 |
| $\varepsilon$ | epsilon | Gürültü (hata) terimi (8.2); DBSCAN komşuluk yarıçapı (17.3) | 8.2 / 17.3 |
| $\cup$ | birleşim | Küme birleşimi | 18.2 |
| $\Rightarrow$ | ise | Genel olarak "ise" (7.2); birliktelik kuralında $X \Rightarrow Y$, $X$ varsa $Y$'nin de bulunduğunu anlatır (18.2) | 7.2 / 18.2 |
| supp, conf, lift | destek, güven, kaldıraç | Birliktelik kuralı ölçütleri | 18.2 |
| $\lambda$ | lamda | Düzenlileştirme katsayısı / özdeğer (PCA) | 8.4 / 19.2 |
| $\Sigma$ | büyük sigma | Kovaryans matrisi | 19.2 |
| $\mathbf{v}_k$, $Z$ | ve ka, zet | PCA: $k$. özvektör; yeni (bileşen) koordinatlardaki veri | 19.2 |
| $\mathbb{I}(\cdot)$ | gösterge fonksiyonu | Koşul doğruysa 1, değilse 0 | 22.1 |
| $\Longrightarrow$ | ise, gerektirir | Soldaki doğruysa sağdaki de doğrudur | 22.3 |
| $w_{y_i}$ | dabılyu ye i | $i$. örneğin gerçek sınıfının ağırlığı | 22.3 |
| $t^{*}$ | te yıldız | Maliyeti en küçük yapan karar eşiği | 22.3 |
| $\bar{d}$ | de bar | İki modelin skor farklarının ortalaması | 24.2 |
| $\in$ | elemanıdır | Kümeye aitlik | 7.3 |

---

[← 25. Kapsamlı Uygulama (Proje Ödevi)](25-proje-odevi.md) · [İçindekiler](../Ders_Notu.md) · [Ek B: Türkçe–İngilizce Terimler →](ek-b-terimler-sozlugu.md)
