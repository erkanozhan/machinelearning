# Açıklamalı formül resimleri

Bu klasördeki PNG dosyaları, `Ders_Notu.md` içindeki açıklamalı formüllerin (parçaları üst ve alt parantezlerle işaretlenmiş formüllerin) resimleridir. GitHub bir sayfada yalnızca sınırlı miktarda formül çizebildiği için bu formüller notta resim olarak yer alır. Aşağıda her resmin LaTeX kaynağı vardır; bir etiketi değiştirmek için kaynağı düzenleyip resmi KaTeX ile yeniden üretin (beyaz zemin, 2x çözünürlük).

## 01.png

*Büyük iks, n satırlı ve d sütunlu bir matristir; i'nci satırının j'nci elemanı iks alt i j'dir. Ye ise n elemanlı bir sütun vektörüdür; i'nci elemanı ye alt i'dir.*

```math
\underbrace{X}_{\substack{\text{öznitelik matrisi}\\ n \times d}} = \Big[\; \underbrace{x_{ij}}_{\substack{\text{i. satır (örnek),}\\[2pt]\text{j. sütun (öznitelik)}}} \;\Big], \qquad \underbrace{y}_{\text{hedef vektörü}} = \Big[\; \overbrace{y_i}^{\text{i. örneğin hedefi}} \;\Big]
```

## 02.png

*Ye şapka eşittir teta sıfır artı teta bir iks bir artı teta iki iks iki artı … artı teta de iks de.*

```math
\underbrace{\hat{y}}_{\text{tahmin}} = \overbrace{\theta_0}^{\text{sabit terim}} + \underbrace{\theta_1 x_1}_{\substack{\text{1. öznitelik}\\[2pt]\text{çarpı katsayısı}}} + \overbrace{\theta_2 x_2}^{\substack{\text{2. öznitelik}\\[2pt]\text{çarpı katsayısı}}} + \dots + \underbrace{\theta_d x_d}_{\substack{\text{d. öznitelik}\\[2pt]\text{çarpı katsayısı}}}
```

## 03.png

*İks bar eşittir mü eşittir, bir bölü en çarpı, i eşittir 1'den n'ye kadar iks alt i'lerin toplamı.*

```math
\underbrace{\bar{x} = \mu}_{\text{ortalama}} = \overbrace{\frac{1}{n}}^{\substack{\text{toplamı}\\[2pt]\text{n'ye böl}}} \; \underbrace{\sum_{i=1}^{n}}_{\substack{\text{i = 1'den n'ye kadar}\\[2pt]\text{hepsini topla}}} \overbrace{x_i}^{\text{i. değer}}
```

## 04.png

*Sigma kare eşittir, bir bölü en çarpı, i eşittir 1'den n'ye kadar iks alt i eksi mü'nün karelerinin toplamı. Es kare eşittir, bir bölü en eksi bir çarpı, i eşittir 1'den n'ye kadar iks alt i eksi iks bar'ın karelerinin toplamı.*

```math
\underbrace{\sigma^2}_{\text{kitle varyansı}} = \underbrace{\frac{1}{n}\sum_{i=1}^{n}}_{\substack{\text{n değerin}\\[2pt]\text{ortalamasını al}}} \underbrace{\big(\overbrace{x_i-\mu}^{\text{ortalamadan sapma}}\big)^2}_{\text{sapmanın karesi}} \qquad\qquad \underbrace{s^2}_{\text{örneklem varyansı}} = \overbrace{\frac{1}{n-1}}^{\substack{\text{n yerine}\\ n-1 \text{ ile böl}}} \sum_{i=1}^{n}(x_i-\bar{x})^2
```

## 05.png

*İks üssü eşittir, iks eksi iks min, bölü iks maks eksi iks min.*

```math
\underbrace{x'}_{\text{yeni değer}} = \frac{\overbrace{x - x_{\min}}^{\substack{\text{değerin en küçük}\\[2pt]\text{değerden uzaklığı}}}}{\underbrace{x_{\max} - x_{\min}}_{\substack{\text{özniteliğin}\\[2pt]\text{değişim aralığı}}}}
```

## 06.png

*Zet eşittir, iks eksi mü, bölü sigma.*

```math
\underbrace{z}_{\text{z-skoru}} = \frac{\overbrace{x - \mu}^{\text{ortalamadan sapma}}}{\underbrace{\sigma}_{\substack{\text{standart sapma}\\[2pt]\text{(ölçü birimi)}}}}
```

## 07.png

*İks üssü eşittir, iks eksi iks'in medyanı, bölü IQR. IQR eşittir ku üç eksi ku bir.*

```math
\underbrace{x'}_{\text{yeni değer}} = \frac{\overbrace{x - \text{medyan}(x)}^{\text{ortancadan uzaklık}}}{\underbrace{IQR}_{\substack{\text{ortadaki \%50'nin}\\[2pt]\text{genişliği}}}}, \qquad IQR = \underbrace{Q_3}_{\substack{\text{\%75}\\[2pt]\text{noktası}}} - \underbrace{Q_1}_{\substack{\text{\%25}\\[2pt]\text{noktası}}}
```

## 08.png

*İks üssü eşittir, iks bölü on üzeri j. Burada j, ölçeklenmiş değerlerin mutlak değerce en büyüğünü 1'in altına indiren en küçük tam sayıdır.*

```math
\underbrace{x'}_{\text{yeni değer}} = \frac{x}{\underbrace{10^{j}}_{\substack{\text{virgülü j basamak}\\[2pt]\text{sola kaydır}}}}, \qquad \overbrace{j}^{\substack{\text{en küçük}\\[2pt]\text{tam sayı}}}:\ \underbrace{\max_i \lvert x'_i \rvert < 1}_{\substack{\text{ölçeklenmiş değerlerin en büyük}\\[2pt]\text{mutlak değeri 1'den küçük olsun}}}
```

## 09.png

*a ile be arasındaki Öklid uzaklığı eşittir, j eşittir 1'den d'ye kadar a alt j eksi be alt j'nin karelerinin toplamının karekökü. Manhattan uzaklığı eşittir, j eşittir 1'den d'ye kadar a alt j eksi be alt j'nin mutlak değerlerinin toplamı.*

```math
\underbrace{d_{\text{Öklid}}(\mathbf{a},\mathbf{b})}_{\text{kuş uçuşu uzaklık}} = \sqrt{\underbrace{\sum_{j=1}^{d}}_{\substack{\text{tüm öznitelikler}\\[2pt]\text{için topla}}} \overbrace{(a_j-b_j)^2}^{\substack{\text{j. öznitelikteki}\\[2pt]\text{farkın karesi}}}} \qquad \underbrace{d_{\text{Manhattan}}(\mathbf{a},\mathbf{b})}_{\text{blok blok yürüme uzaklığı}} = \sum_{j=1}^{d}\overbrace{\lvert a_j-b_j \rvert}^{\substack{\text{j. öznitelikteki}\\[2pt]\text{farkın mutlak değeri}}}
```

## 10.png

*Ye şapka eşittir, h alt teta iks eşittir, teta sıfır artı teta bir iks.*

```math
\underbrace{\hat{y}}_{\text{tahmin}} = \underbrace{h_\theta(x)}_{\substack{\text{teta parametreli}\\[2pt]\text{model}}} = \overbrace{\theta_0}^{\text{kesişim}} + \overbrace{\theta_1}^{\text{eğim}} \underbrace{x}_{\text{girdi}}
```

## 11.png

*Ce teta sıfır, teta bir eşittir, bir bölü iki em çarpı, i eşittir 1'den m'ye kadar ye şapka alt i eksi ye alt i'nin karelerinin toplamı.*

```math
\underbrace{J(\theta_0,\theta_1)}_{\substack{\text{maliyet: bu parametreler}\\[2pt]\text{ne kadar kötü?}}} = \overbrace{\frac{1}{2m}}^{\substack{\text{ortalama al;}\\[2pt]\text{1/2 türev için}}} \underbrace{\sum_{i=1}^{m}}_{\substack{\text{tüm eğitim}\\[2pt]\text{örnekleri için topla}}} \overbrace{\big(\underbrace{\hat{y}_i - y_i}_{\text{hata}}\big)^2}^{\text{hatanın karesi}}
```

## 12.png

*Teta bir eşittir, i eşittir 1'den m'ye kadar iks alt i eksi iks bar çarpı ye alt i eksi ye bar'ların toplamı, bölü i eşittir 1'den m'ye kadar iks alt i eksi iks bar'ın karelerinin toplamı. Teta sıfır eşittir ye bar eksi teta bir çarpı iks bar.*

```math
\underbrace{\theta_1}_{\text{eğim}} = \frac{\overbrace{\sum_{i=1}^{m}(x_i-\bar{x})(y_i-\bar{y})}^{\substack{\text{x ile y'nin ortalamadan}\\[2pt]\text{birlikte sapması}}}}{\underbrace{\sum_{i=1}^{m}(x_i-\bar{x})^2}_{\substack{\text{x'in kendi sapmalarının}\\[2pt]\text{kareleri}}}}, \qquad \underbrace{\theta_0}_{\text{kesişim}} = \overbrace{\bar{y} - \theta_1\,\bar{x}}^{\substack{\text{doğru, ortalama noktadan}\\[2pt]\text{geçsin diye}}}
```

## 13.png

*Teta j'ye, teta j eksi alfa çarpı ce'nin teta j'ye göre kısmi türevi atanır.*

```math
\underbrace{\theta_j}_{\text{yeni değer}} \overbrace{:=}^{\text{atanır}} \underbrace{\theta_j}_{\text{eski değer}} - \overbrace{\alpha}^{\substack{\text{adım}\\[2pt]\text{büyüklüğü}}} \underbrace{\frac{\partial J}{\partial \theta_j}}_{\substack{\text{eğim: teta j artınca}\\[2pt]\text{J ne kadar değişir?}}}
```

## 14.png

*Ce'nin teta sıfıra göre kısmi türevi eşittir, bir bölü em çarpı, i eşittir 1'den m'ye kadar ye şapka alt i eksi ye alt i'lerin toplamı. Ce'nin teta bire göre kısmi türevi eşittir, bir bölü em çarpı, aynı hataların iks alt i ile çarpılarak toplamı.*

```math
\underbrace{\frac{\partial J}{\partial \theta_0}}_{\substack{\text{teta sıfıra}\\[2pt]\text{göre eğim}}} = \overbrace{\frac{1}{m}\sum_{i=1}^{m}}^{\text{ortalama}}\underbrace{(\hat{y}_i - y_i)}_{\text{hata}}, \qquad \underbrace{\frac{\partial J}{\partial \theta_1}}_{\substack{\text{teta bire}\\[2pt]\text{göre eğim}}} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)\,\overbrace{x_i}^{\substack{\text{hata, girdiyle}\\[2pt]\text{ağırlıklandırılır}}}
```

## 15.png

*Ye şapka eşittir teta sıfır artı, j eşittir 1'den d'ye kadar teta alt j çarpı iks alt j'lerin toplamı.*

```math
\underbrace{\hat{y}}_{\text{tahmin}} = \underbrace{\theta_0}_{\text{sabit terim}} + \underbrace{\sum_{j=1}^{d}}_{\substack{\text{j = 1'den d'ye kadar}\\[2pt]\text{tüm öznitelikler}}} \overbrace{\theta_j x_j}^{\substack{\text{j. öznitelik çarpı}\\[2pt]\text{katsayısı}}}
```

## 16.png

*Teta vektörü eşittir, iks transpoz iks'in tersi çarpı iks transpoz çarpı ye vektörü.*

```math
\underbrace{\boldsymbol{\theta}}_ {\substack{\text{tüm parametreler}\\\text{tek seferde}}} = \underbrace{\left(X^{T}X\right)^{-1}}_ {\substack{\text{özniteliklerin kendi aralarındaki}\\\text{ilişkinin tersi}}}\overbrace{X^{T}\mathbf{y}}^{\substack{\text{öznitelikler ile hedefin}\\\text{ilişkisi}}}
```

## 17.png

*Ye şapka eşittir teta sıfır artı teta bir iks artı teta iki iks kare. İks iki'ye iks kare atanırsa: ye şapka eşittir teta sıfır artı teta bir iks bir artı teta iki iks iki.*

```math
\underbrace{\hat{y} = \theta_0 + \theta_1 x + \theta_2 x^2}_{\text{x'e göre eğri (parabol)}} \quad\underbrace{\xrightarrow{\;x_2 \,:=\, x^2\;}}_{\text{yeni sütun ekle}}\quad \underbrace{\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2}_{\substack{\text{tetalara göre doğrusal:}\\[2pt]\text{çoklu lineer regresyon}}}
```

## 18.png

*Zet eşittir teta sıfır artı teta bir iks bir artı … artı teta de iks de. Pe şapka eşittir, iks vektörü verildiğinde ye'nin bir olma olasılığı, eşittir sigma zet, eşittir bir bölü bir artı e üzeri eksi zet.*

```math
\underbrace{z}_{\substack{\text{doğrusal skor}\\[2pt]\text{(logit)}}} = \theta_0 + \theta_1 x_1 + \cdots + \theta_d x_d, \qquad \underbrace{\hat{p}}_{\substack{\text{tahmin edilen}\\[2pt]\text{olasılık}}} = \overbrace{P(y=1 \mid \mathbf{x})}^{\substack{\text{x verildiğinde sınıfın}\\[2pt]\text{1 olma olasılığı}}} = \underbrace{\sigma(z)}_{\text{sigmoid}} = \frac{1}{1+\underbrace{e^{-z}}_{\substack{\text{z büyüdükçe}\\[2pt]\text{0'a yaklaşır}}}}
```

## 19.png

*Ce teta eşittir eksi bir bölü em çarpı, i eşittir 1'den m'ye kadar, ye alt i çarpı log pe şapka alt i artı bir eksi ye alt i çarpı log bir eksi pe şapka alt i'lerin toplamı.*

```math
\underbrace{J(\boldsymbol{\theta})}_{\text{ortalama kayıp}} = \overbrace{-\frac{1}{m}\sum_{i=1}^{m}}^{\substack{\text{eksi işaretli}\\[2pt]\text{ortalama}}}\Big[\,\underbrace{y_i \log(\hat{p}_i)}_{\substack{\text{gerçek sınıf 1 ise}\\[2pt]\text{çalışan terim}}} + \underbrace{(1-y_i)\log(1-\hat{p}_i)}_{\substack{\text{gerçek sınıf 0 ise}\\[2pt]\text{çalışan terim}}}\Big]
```

## 20.png

*Seçilmeme olasılığı eşittir, bir eksi bir bölü en'in en'inci kuvveti; en sonsuza giderken e üzeri eksi bire, yani yaklaşık 0.368'e yaklaşır.*

```math
\underbrace{P(\text{seçilmeme})}_{\substack{\text{bir örneğin hiç}\\[2pt]\text{seçilmeme olasılığı}}} = \underbrace{\Big(\overbrace{1-\frac{1}{n}}^{\substack{\text{tek çekişte}\\[2pt]\text{seçilmeme}}}\Big)^{n}}_{\substack{\text{n çekişin hepsinde}\\[2pt]\text{seçilmeme}}} \xrightarrow{\;n\to\infty\;} \underbrace{e^{-1} \approx 0.368}_{\substack{\text{n büyüdükçe}\\[2pt]\text{yaklaşılan değer}}}
```

## 21.png

*Kesinlik eşittir, te pe bölü te pe artı ef pe.*

```math
\text{Precision} = \frac{\overbrace{TP}^{\text{doğru pozitifler}}}{\underbrace{TP + FP}_{\substack{\text{pozitif dediğimiz}\\[2pt]\text{tüm örnekler}}}}
```

## 22.png

*Duyarlılık eşittir te pe er eşittir, te pe bölü te pe artı ef en.*

```math
\text{Recall} = \text{TPR} = \frac{\overbrace{TP}^{\text{yakalanan pozitifler}}}{\underbrace{TP + FN}_{\substack{\text{gerçekte pozitif}\\[2pt]\text{olanların hepsi}}}}
```

## 23.png

*Özgüllük eşittir te en er eşittir, te en bölü te en artı ef pe. Ef pe er eşittir, ef pe bölü ef pe artı te en; bu da bir eksi özgüllüktür.*

```math
\text{Specificity} = \frac{\overbrace{TN}^{\substack{\text{doğru bilinen}\\[2pt]\text{negatifler}}}}{\underbrace{TN + FP}_{\substack{\text{gerçekte negatif}\\[2pt]\text{olanların hepsi}}}}, \qquad \text{FPR} = \frac{\overbrace{FP}^{\substack{\text{yanlış}\\[2pt]\text{alarmlar}}}}{\underbrace{FP + TN}_{\substack{\text{gerçekte negatif}\\[2pt]\text{olanların hepsi}}}}
```

## 24.png

*Ef bir eşittir, iki çarpı precision çarpı recall, bölü precision artı recall; bu da iki te pe bölü iki te pe artı ef pe artı ef en'e eşittir.*

```math
F_1 = \overbrace{2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision} + \text{Recall}}}^{\substack{\text{precision ile recall'un}\\[2pt]\text{harmonik ortalaması}}} = \underbrace{\frac{2\,TP}{2\,TP + FP + FN}}_{\substack{\text{aynı değerin karışıklık}\\[2pt]\text{matrisiyle yazılışı}}}
```

## 25.png

*Kappa eşittir, pe o eksi pe e, bölü bir eksi pe e.*

```math
\underbrace{\kappa}_{\text{kappa}} = \frac{\overbrace{p_o - p_e}^{\substack{\text{şansın üstündeki}\\[2pt]\text{gerçek başarı}}}}{\underbrace{1 - p_e}_{\substack{\text{şansın üstünde ulaşılabilecek}\\[2pt]\text{en büyük başarı}}}}
```

## 26.png

*Em ce ce eşittir, te pe çarpı te en eksi ef pe çarpı ef en, bölü te pe artı ef pe, te pe artı ef en, te en artı ef pe ve te en artı ef en'in çarpımının karekökü.*

```math
MCC = \frac{\overbrace{TP\cdot TN}^{\text{doğruların çarpımı}} - \overbrace{FP\cdot FN}^{\text{hataların çarpımı}}}{\underbrace{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}_{\substack{\text{iki satır ve iki sütun toplamı;}\\[2pt]\text{sonucu } [-1, +1] \text{ aralığına taşır}}}}
```

## 27.png

*Em a e eşittir, bir bölü en çarpı, i eşittir 1'den n'ye kadar ye alt i eksi ye şapka alt i'nin mutlak değerlerinin toplamı.*

```math
MAE = \underbrace{\frac{1}{n}\sum_{i=1}^{n}}_{\text{ortalaması}}\overbrace{\lvert y_i - \hat{y}_i \rvert}^{\substack{\text{i. örnekteki hatanın}\\[2pt]\text{mutlak değeri}}}
```

## 28.png

*Em es e eşittir, bir bölü en çarpı, i eşittir 1'den n'ye kadar ye alt i eksi ye şapka alt i'nin karelerinin toplamı. Ar em es e eşittir karekök em es e.*

```math
MSE = \frac{1}{n}\sum_{i=1}^{n}\overbrace{(y_i - \hat{y}_i)^2}^{\text{hatanın karesi}}, \qquad \underbrace{RMSE}_{\substack{\text{hedefle aynı}\\[2pt]\text{birimde}}} = \overbrace{\sqrt{MSE}}^{\substack{\text{karekök birimi}\\[2pt]\text{geri getirir}}}
```

## 29.png

*Ar kare eşittir, bir eksi es es res bölü es es tot; yani bir eksi, ye alt i eksi ye şapka alt i'nin karelerinin toplamı bölü ye alt i eksi ye bar'ın karelerinin toplamı.*

```math
R^2 = 1 - \frac{\overbrace{SS_{res}}^{\substack{\text{modelin}\\[2pt]\text{hatası}}}}{\underbrace{SS_{tot}}_{\substack{\text{ortalama tahmininin}\\[2pt]\text{hatası}}}} = 1 - \frac{\overbrace{\sum_i (y_i - \hat{y}_i)^2}^{\text{tahmine göre sapmalar}}}{\underbrace{\sum_i (y_i - \bar{y})^2}_{\text{ortalamaya göre sapmalar}}}
```

## 30.png

*Ar kare düzeltilmiş eşittir, bir eksi, bir eksi ar kare çarpı en eksi bir, bölü en eksi ka eksi bir.*

```math
R^2_{adj} = 1 - \frac{\overbrace{(1-R^2)}^{\substack{\text{açıklanamayan}\\[2pt]\text{pay}}}\quad\overbrace{(n-1)}^{\substack{\text{örnek sayısı}\\[2pt]\text{eksi 1}}}}{\underbrace{n-k-1}_{\substack{\text{öznitelik eklendikçe}\\[2pt]\text{küçülür}}}}
```

## 31.png

*Ar eşittir, iks alt i eksi iks bar çarpı ye alt i eksi ye bar'ların toplamı, bölü iks alt i eksi iks bar'ın karelerinin toplamının karekökü çarpı ye alt i eksi ye bar'ın karelerinin toplamının karekökü.*

```math
r = \frac{\overbrace{\sum_i (x_i - \bar{x})(y_i - \bar{y})}^{\substack{\text{x ile y ortalamadan}\\[2pt]\text{aynı yönde mi sapıyor?}}}}{\underbrace{\sqrt{\sum_i (x_i-\bar{x})^2}}_{\text{x'in yayılımı}}\,\underbrace{\sqrt{\sum_i (y_i-\bar{y})^2}}_{\text{y'nin yayılımı}}}
```

## 32.png

*İks verildiğinde ce'nin olasılığı eşittir, ce verildiğinde iks'in olasılığı çarpı ce'nin olasılığı, bölü iks'in olasılığı.*

```math
\underbrace{P(C \mid \mathbf{x})}_{\substack{\text{sonsal: x görüldükten}\\[2pt]\text{sonra sınıfın olasılığı}}} = \frac{\overbrace{P(\mathbf{x} \mid C)}^{\text{olabilirlik}}\;\overbrace{P(C)}^{\text{önsel}}}{\underbrace{P(\mathbf{x})}_{\substack{\text{kanıt: tüm sınıflar}\\[2pt]\text{için aynı}}}}
```

## 33.png

*İks bir'den iks de'ye kadar öznitelikler verildiğinde ce'nin olasılığı, ce'nin olasılığı çarpı, j eşittir 1'den d'ye kadar ce verildiğinde iks alt j'nin olasılıklarının çarpımı ile orantılıdır.*

```math
\underbrace{P(C \mid x_1,\dots,x_d)}_{\substack{\text{tüm öznitelikler verildiğinde}\\[2pt]\text{sınıfın olasılığı}}} \;\overbrace{\propto}^{\text{orantılıdır}}\; \underbrace{P(C)}_{\text{önsel}}\,\underbrace{\prod_{j=1}^{d} \overbrace{P(x_j \mid C)}^{\substack{\text{her özniteliğin}\\[2pt]\text{tek başına olasılığı}}}}_{\text{hepsinin çarpımı}}
```

## 34.png

*Ce verildiğinde iks alt j'nin ve'ye eşit olma olasılığı eşittir, sayım artı bir, bölü en alt ce artı ka.*

```math
\underbrace{P(x_j = v \mid C)}_ {\substack{\text{C sınıfında özniteliğin}\\\text{v değerini alma olasılığı}}} = \frac{\overbrace{\text{sayım} + 1}^{\substack{\text{her sayıma}\\\text{1 eklenir}}}}{\underbrace{n_C + k}_ {\substack{\text{sınıftaki örnek sayısı}\\\text{artı farklı değer sayısı}}}}
```

## 35.png

*Ef iks eşittir, bir bölü karekök iki pi çarpı sigma, çarpı e üzeri eksi, iks eksi mü'nün karesi bölü iki sigma kare.*

```math
\underbrace{f(x)}_ {\substack{\text{olasılık}\\\text{yoğunluğu}}} = \overbrace{\frac{1}{\sqrt{2\pi}\thinspace \sigma}}^{\substack{\text{eğrinin altındaki alan}\\\text{1 olsun diye}}}\thickspace  \underbrace{e^{-\frac{(x-\mu)^2}{2\sigma^2}}}_ {\substack{\text{x, ortalamadan uzaklaştıkça}\\\text{hızla küçülür}}}
```

## 36.png

*Ha es eşittir eksi, i eşittir 1'den c'ye kadar pe alt i çarpı log iki tabanında pe alt i'lerin toplamı.*

```math
\underbrace{H(S)}_{\substack{\text{S'deki belirsizlik}\\[2pt]\text{(bit)}}} = \overbrace{-}^{\substack{\text{sonucu pozitif}\\[2pt]\text{yapar}}} \underbrace{\sum_{i=1}^{c}}_{\substack{\text{c sınıfın}\\[2pt]\text{hepsi için topla}}} \overbrace{p_i}^{\substack{\text{i. sınıfın oranı}\\[2pt]\text{(ağırlık)}}} \; \underbrace{\log_2 p_i}_{\substack{\text{i. sınıfın sürpriz}\\[2pt]\text{değeri (negatif)}}}
```

## 37.png

*Ay ci es a eşittir, ha es eksi, A'nın değerlerindeki her v için es alt v'nin eleman sayısı bölü es'in eleman sayısı çarpı ha es alt v'lerin toplamı.*

```math
\underbrace{IG(S, A)}_{\substack{\text{A ile bölmenin}\\[2pt]\text{kazancı}}} = \underbrace{H(S)}_{\text{bölmeden önce}} - \underbrace{\sum_{v \,\in\, \text{Değerler}(A)} \overbrace{\frac{\lvert S_v \rvert}{\lvert S \rvert}}^{\substack{\text{v dalına düşen}\\[2pt]\text{örnek oranı}}}\, \overbrace{H(S_v)}^{\text{v dalının entropisi}}}_{\text{bölmeden sonra: dalların ağırlıklı ortalama entropisi}}
```

## 38.png

*Gini es eşittir, bir eksi pe alt i'lerin karelerinin toplamı.*

```math
\underbrace{Gini(S)}_ {\text{safsızlık}} = 1 - \underbrace{\sum_i p_i^2}_ {\substack{\text{rastgele seçilen iki örneğin}\\\text{aynı sınıftan olma olasılığı}}}
```

## 39.png

*Dabılyu ve be üzerinden, bir bölü iki çarpı dabılyu'nun normunun karesini en küçük yap; koşul: her i için ye alt i çarpı, dabılyu nokta iks alt i artı be, birden büyük veya eşit olsun.*

```math
\underbrace{\min_{\mathbf{w},\,b}}_{\substack{\text{w ve b'yi öyle seç ki}\\[2pt]\text{şu en küçük olsun}}}\; \overbrace{\frac{1}{2}\lVert\mathbf{w}\rVert^2}^{\substack{\text{küçükse}\\[2pt]\text{marj geniş}}} \quad \text{koşul:}\quad \underbrace{y_i\,(\mathbf{w}\cdot\mathbf{x}_i + b) \ge 1}_{\substack{\text{her örnek kendi tarafında}\\[2pt]\text{ve marjın dışında}}} \quad \overbrace{(i = 1,\dots,n)}^{\text{tüm örnekler için}}
```

## 40.png

*Dabılyu, be ve ksi üzerinden, bir bölü iki dabılyu'nun normunun karesi artı ce çarpı, i eşittir 1'den n'ye kadar ksi alt i'lerin toplamını en küçük yap.*

```math
\min_{\mathbf{w},\,b,\,\xi}\; \underbrace{\frac{1}{2}\lVert\mathbf{w}\rVert^2}_{\text{marjı genişlet}} + \overbrace{C}^{\substack{\text{ihlallerin}\\[2pt]\text{ceza katsayısı}}}\underbrace{\sum_{i=1}^{n}\xi_i}_{\substack{\text{toplam marj}\\[2pt]\text{ihlali}}}
```

## 41.png

*Zet eşittir, j eşittir 1'den d'ye kadar dabılyu alt j çarpı iks alt j'lerin toplamı artı be; ye şapka eşittir ef zet.*

```math
\underbrace{z}_{\text{ağırlıklı toplam}} = \underbrace{\sum_{j=1}^{d} \overbrace{w_j}^{\text{ağırlık}}\quad\overbrace{x_j}^{\text{girdi}}}_{\substack{\text{her girdiyi ağırlığıyla}\\[2pt]\text{çarpıp topla}}} + \underbrace{b}_{\text{sapma}}, \qquad \underbrace{\hat{y}}_{\text{çıktı}} = \overbrace{f}^{\substack{\text{aktivasyon}\\[2pt]\text{fonksiyonu}}}(z)
```

## 42.png

*Ye şapka sınıf eşittir, ye şapka üst bir'den ye şapka üst be'ye kadar tahminlerin modu. Ye şapka regresyon eşittir, bir bölü be çarpı, b eşittir 1'den B'ye kadar ye şapka üst b'lerin toplamı.*

```math
\underbrace{\hat{y}_{\text{sınıf}}}_{\substack{\text{sınıflandırmada}\\[2pt]\text{nihai tahmin}}} = \overbrace{\mathrm{mod}}^{\substack{\text{en sık}\\[2pt]\text{görülen}}}\big\lbrace \underbrace{\hat{y}^{(1)}, \dots, \hat{y}^{(B)}}_{\text{B modelin tahminleri}} \big\rbrace, \qquad \underbrace{\hat{y}_{\text{regresyon}}}_{\substack{\text{regresyonda}\\[2pt]\text{nihai tahmin}}} = \overbrace{\frac{1}{B}\sum_{b=1}^{B}}^{\text{ortalama}}\hat{y}^{(b)}
```

## 43.png

*Ef alt te iks eşittir, ef alt te eksi bir iks artı nü çarpı ha alt te iks.*

```math
\underbrace{F_t(\mathbf{x})}_ {\text{yeni topluluk}} = \underbrace{F_ {t-1}(\mathbf{x})}_ {\text{şimdiye kadarki topluluk}} + \overbrace{\nu}^{\substack{\text{küçük}\\\text{adım}}}\thinspace  \underbrace{h_t(\mathbf{x})}_ {\substack{\text{artıkları tahmin eden}\\\text{yeni küçük ağaç}}}
```

## 44.png

*Es es e eşittir, ka eşittir 1'den büyük ka'ya kadar, ce alt ka kümesindeki her iks için iks eksi mü alt ka'nın normunun karesinin toplamı.*

```math
\underbrace{SSE}_{\substack{\text{küme içi}\\[2pt]\text{kareler toplamı}}} = \underbrace{\sum_{k=1}^{K}}_{\substack{\text{tüm kümeler}\\[2pt]\text{için topla}}}\;\overbrace{\sum_{\mathbf{x}\in C_k}}^{\substack{\text{k. kümedeki her}\\[2pt]\text{nokta için topla}}}\underbrace{\lVert \mathbf{x} - \boldsymbol{\mu}_k \rVert^2}_{\substack{\text{noktanın kendi merkezine}\\[2pt]\text{uzaklığının karesi}}}
```

## 45.png

*Es i eşittir, be i eksi a i, bölü a i ile be i'nin büyüğü.*

```math
\underbrace{s(i)}_{\substack{\text{i. noktanın}\\[2pt]\text{siluet değeri}}} = \frac{\overbrace{b(i)}^{\substack{\text{en yakın diğer}\\[2pt]\text{kümeye uzaklık}}} - \overbrace{a(i)}^{\substack{\text{kendi kümesine}\\[2pt]\text{uzaklık}}}}{\underbrace{\max\lbrace a(i),\, b(i) \rbrace}_{\substack{\text{ikisinden büyüğü:}\\[2pt]\text{sonucu } [-1, +1] \text{ aralığına sıkıştırır}}}}
```

## 46.png

*İks'in desteği eşittir, iks'in sayımı bölü en. İks ise ye kuralının güveni eşittir, iks birleşim ye'nin desteği bölü iks'in desteği. İks ise ye kuralının lift'i eşittir, güven bölü ye'nin desteği.*

```math
\begin{gathered}
\underbrace{\text{supp}(X)}_{\text{destek}} = \frac{\overbrace{\text{count}(X)}^{\substack{\text{X'i içeren}\\[2pt]\text{sepet sayısı}}}}{\underbrace{N}_{\substack{\text{toplam}\\[2pt]\text{sepet sayısı}}}} \qquad \underbrace{\text{conf}(X \Rightarrow Y)}_{\text{güven}} = \frac{\overbrace{\text{supp}(X \cup Y)}^{\text{X ve Y birlikte}}}{\underbrace{\text{supp}(X)}_{\substack{\text{yalnızca X'li sepetlere}\\[2pt]\text{bakılır}}}} \\[1.5em]
\underbrace{\text{lift}(X \Rightarrow Y)}_{\text{kaldıraç}} = \frac{\overbrace{\text{conf}(X\Rightarrow Y)}^{\substack{\text{X alanlarda}\\[2pt]\text{Y oranı}}}}{\underbrace{\text{supp}(Y)}_{\substack{\text{tüm sepetlerde}\\[2pt]\text{Y oranı}}}}
\end{gathered}
```

## 47.png

*Büyük sigma çarpı ve alt ka eşittir lamda alt ka çarpı ve alt ka. Açıklanan varyans oranı eşittir, lamda alt ka bölü bütün lamda alt j'lerin toplamı.*

```math
\underbrace{\Sigma}_ {\substack{\text{kovaryans}\\\text{matrisi}}}\thinspace \underbrace{\mathbf{v}_k}_ {\text{yön}} = \overbrace{\lambda_k}^{\substack{\text{o yöndeki}\\\text{varyans}}}\thinspace \mathbf{v}_k, \qquad \underbrace{\frac{\lambda_k}{\sum_j \lambda_j}}_ {\substack{\text{k. bileşenin açıkladığı}\\\text{varyans oranı}}}
```

## 48.png

*Ce L2 teta eşittir, ce teta artı lamda çarpı, j eşittir 1'den d'ye kadar teta alt j'nin karelerinin toplamı. Ce L1 teta eşittir, ce teta artı lamda çarpı, j eşittir 1'den d'ye kadar teta alt j'nin mutlak değerlerinin toplamı.*

```math
\underbrace{J_{\text{L2}}(\theta)}_{\text{Ridge maliyeti}} = \overbrace{J(\theta)}^{\text{eski maliyet}} + \underbrace{\lambda}_{\substack{\text{ceza}\\[2pt]\text{şiddeti}}}\overbrace{\sum_{j=1}^{d}\theta_j^2}^{\substack{\text{katsayıların}\\[2pt]\text{kareleri}}} \qquad\qquad \underbrace{J_{\text{L1}}(\theta)}_{\text{Lasso maliyeti}} = J(\theta) + \lambda\overbrace{\sum_{j=1}^{d}\lvert\theta_j\rvert}^{\substack{\text{katsayıların}\\[2pt]\text{mutlak değerleri}}}
```

## 49.png

*Teta j'ye, teta j eksi eta çarpı, ce'nin teta j'ye göre kısmi türevi artı lamda teta j atanır.*

```math
\underbrace{\theta_j}_{\text{yeni değer}} \overbrace{\leftarrow}^{\text{atanır}} \theta_j - \underbrace{\eta}_{\substack{\text{öğrenme}\\[2pt]\text{oranı}}}\Big(\overbrace{\frac{\partial J}{\partial \theta_j}}^{\substack{\text{hatayı azaltan}\\[2pt]\text{eğim}}} + \underbrace{\lambda\,\theta_j}_{\substack{\text{katsayıyı sıfıra}\\[2pt]\text{çeken ceza}}}\Big)
```

## 50.png

*Hata eşittir, i eşittir 1'den büyük en'e kadar, ye alt i ye şapka alt i'ye eşit değilse 1, eşitse 0 veren gösterge fonksiyonunun toplamı.*

```math
\underbrace{\text{Hata}}_{\text{hata sayısı}} = \underbrace{\sum_{i=1}^{N}}_{\substack{\text{tüm örnekler}\\[2pt]\text{için topla}}} \overbrace{\mathbb{I}(\underbrace{y_i \neq \hat{y}_i}_{\text{yanlış tahmin mi?}})}^{\substack{\text{evet ise 1,}\\[2pt]\text{hayır ise 0}}}
```

## 51.png

*Ce eşittir, ce sıfır sıfır, ce sıfır bir, ce bir sıfır, ce bir bir matrisi; bu da sıfır, ce ef pe, ce ef en, sıfır matrisine eşittir.*

```math
\underbrace{C}_{\text{maliyet matrisi}} = \begin{bmatrix} C_{00} & C_{01} \\ C_{10} & C_{11} \end{bmatrix} = \begin{bmatrix} \overbrace{0}^{\substack{\text{TN: doğru,}\\[2pt]\text{bedelsiz}}} & \overbrace{C_{FP}}^{\substack{\text{gerçek 0, tahmin 1:}\\[2pt]\text{yanlış alarm}}} \\[1.2em] \underbrace{C_{FN}}_{\substack{\text{gerçek 1, tahmin 0:}\\[2pt]\text{kaçırılan pozitif}}} & \underbrace{0}_{\substack{\text{TP: doğru,}\\[2pt]\text{bedelsiz}}} \end{bmatrix}
```

## 52.png

*Toplam maliyet eşittir, i eşittir 1'den büyük en'e kadar ce ye alt i, ye şapka alt i'lerin toplamı; bu da ef pe çarpı ce ef pe artı ef en çarpı ce ef en'e eşittir.*

```math
\underbrace{\text{Maliyet}_{\text{toplam}}}_{\text{toplam bedel}} = \underbrace{\sum_{i=1}^{N} C(y_i, \hat{y}_i)}_{\substack{\text{her örneğin hücresindeki}\\[2pt]\text{bedeli topla}}} = \overbrace{FP \cdot C_{FP}}^{\substack{\text{yanlış alarm sayısı}\\[2pt]\text{çarpı birim bedeli}}} + \overbrace{FN \cdot C_{FN}}^{\substack{\text{kaçırılan sayısı}\\[2pt]\text{çarpı birim bedeli}}}
```

## 53.png

*Le eşittir eksi, tüm i'ler için dabılyu alt ye i çarpı log pe şapka alt i ye i'lerin toplamı.*

```math
\underbrace{L}_ {\text{toplam kayıp}} = -\sum_ {i} \underbrace{w_ {y_i}}_ {\substack{\text{örneğin gerçek}\\\text{sınıfının ağırlığı}}} \overbrace{\log \hat{p}_ {i,y_i}}^{\substack{\text{modelin gerçek sınıfa verdiği}\\\text{olasılığın logaritması}}}
```

## 54.png

*Pe şapka çarpı sıfır artı bir eksi pe şapka çarpı ce ef pe, pe şapka çarpı ce ef en'den küçükse; pe şapka, te yıldızdan büyüktür. Te yıldız eşittir, ce ef pe bölü ce ef pe artı ce ef en.*

```math
\underbrace{\hat{p}\cdot 0 + (1-\hat{p})\,C_{FP}}_{\substack{\text{pozitif dersek}\\[2pt]\text{beklenen maliyet}}} \;<\; \underbrace{\hat{p}\,C_{FN}}_{\substack{\text{negatif dersek}\\[2pt]\text{beklenen maliyet}}} \quad\Longrightarrow\quad \hat{p} > \overbrace{t^{*}}^{\substack{\text{en iyi}\\[2pt]\text{eşik}}} = \frac{C_{FP}}{C_{FP} + C_{FN}}
```

## 55.png

*Te eşittir, de bar bölü, bir bölü ka ar artı en test bölü en train, çarpı sigma şapka de kare'nin karekökü.*

```math
\underbrace{t}_ {\text{test istatistiği}} = \frac{\overbrace{\bar{d}}^{\substack{\text{ortalama}\\\text{skor farkı}}}}{\underbrace{\sqrt{\Big(\overbrace{\tfrac{1}{k\thinspace r}}^{\substack{\text{standart}\\\text{terim}}} + \overbrace{\tfrac{n_ {test}}{n_ {train}}}^{\substack{\text{örtüşme}\\\text{düzeltmesi}}}\Big)\hat{\sigma}_d^2}}_ {\text{farkın standart hatası}}}
```
