# KNN Algoritması Nedir?
KNN (K-Nearest Neighbors, En Yakın K Komşu) algoritması, gözetimli öğrenme (supervised learning) yöntemlerinden biridir ve hem sınıflandırma (classification) hem de regresyon (regression) problemlerinde kullanılır. 
KNN, bir veri noktasının sınıfını veya değerini belirlerken, o noktaya en yakın 𝐾 komşusunu dikkate alır.

# Kullanım Alanları:

•Hastalık tespitleri(örneğin meme kanseri, bununla ilgili script bulunuyor!)

•Müşterinin verilen krediyi ödeyip ödeyemeyeceği (Customer Default Risk)

•Müşterinin aldığı servisi bırakıp bırakmayacağı (Customer Churn)

•Müşteri segmantasyon (Customer Segmantation)

•El yazısı tanıma (Handwriting Recognition)

•Biyometrik tanımlama (Biometric Identification)


# Temel Prensipler:
Mesafe Ölçümü: KNN, genellikle veri noktaları arasındaki mesafeyi ölçmek için Öklidyen mesafeyi kullanır. Ancak, Manhattan veya Minkowski mesafeleri gibi diğer mesafe ölçümleri de kullanılabilir.
K Parametresi: 𝐾 değeri, dikkate alınacak komşu sayısını belirler. Küçük bir K değeri, modelin fazla uyum yapmasına (overfitting) neden olabilirken, büyük bir 𝐾 değeri ise modelin yeterince karmaşık olmamasına (underfitting) yol açabilir.

# KNN Algoritması Nasıl Çalışır?
**Adım Adım İşleyiş:**

•KNN (K-Nearest Neighbors) Algoritması iki temel değer üzerinden tahmin yapar;

•Distance (Uzaklık): Tahmin edilecek noktanın diğer noktalara uzaklığı hesaplanır. Bunun için Minkowski uzaklık hesaplama fonksiyonu kullanılır.

•K (komuşuluk sayısı): En yakın kaç komşu üzerinden hesaplama yapılacağını söyleriz. K değeri sonucu direkt etkileyecektir. K 1 olursa overfit etme olasılığı çok yüksek olacaktır. Çok büyük olursa da çok genel sonuçlar verecektir. Bu sebeple optimum K değerini tahmin etmek problemin asıl konusu olarak karşımızda durmaktadır. K değerinin önemini aşağıdaki grafik çok güzel bir şekilde göstermektedir. Eğer K=3 ( düz çizginin olduğu yer) seçersek sınıflandırma algoritması ? işareti ile gösterilen noktayı, kırmızı üçgen sınıfı olarak tanımlayacaktır. Fakat K=5 (kesikli çizginin olduğu alan) seçersek sınıflandırma algoritması, aynı noktayı mavi kare sınıfı olarak tanımlayacaktır.

![23-1024x576](https://github.com/SametBatman/KNN-En-Yakin-Komsu/assets/160470839/851c7e0e-fba9-4552-b5d1-9d329931a3fd)

•Veri Kümesi: Eğitim veri kümesi (training dataset) ve test veri kümesi (test dataset) belirlenir.

•Mesafe Hesaplama: Test veri noktasının, eğitim veri kümesindeki her bir veri noktasına olan mesafesi hesaplanır.

•Komşuların Seçimi: Hesaplanan mesafelere göre en yakın 𝐾 komşu seçilir.

**Karar Verme**

•Sınıflandırma: En yakın 𝐾 komşu arasındaki çoğunluk sınıfı test veri noktasının sınıfı olarak atanır.

•Regresyon: En yakın 𝐾 komşunun ortalama değeri test veri noktasının tahmini değeri olarak atanır.

# KNN'in Avantajları ve Dezavantajları

**Avantajları:** 

•Basit ve Kolay Anlaşılır: KNN, sezgisel olarak anlaşılması kolay ve basit bir algoritmadır.

•Eğitim Süreci Yok: Eğitim aşaması olmadığından dolayı hızlı bir şekilde kullanılabilir.

•Esneklik: Hem sınıflandırma hem de regresyon problemlerinde kullanılabilir.

•Doğruluk: Özellikle iyi ayrılmış veri kümelerinde yüksek doğruluk sağlar.

**Dezavantajları:** 

•Hafıza Kullanımı: Tüm eğitim verisini saklaması gerektiğinden hafıza kullanımı yüksektir.

•Hesaplama Maliyeti: Her bir tahmin için mesafe hesaplaması gerektiğinden, büyük veri kümelerinde yavaş çalışabilir.

•Özellik Ölçeklendirme Gereksinimi: Özelliklerin ölçeklendirilmesi gerekebilir, aksi takdirde mesafe ölçümleri yanıltıcı olabilir.

•Gürültüye Duyarlılık: Gürültülü verilere karşı hassastır ve bu durum doğruluğu olumsuz etkileyebilir.

# KNN Accuracy Yani Doğru Tahmin Oranı Nasıl Artılır?

•Daha Fazla Veri Toplama: Daha fazla ve çeşitli veri, modelin daha genelleştirilmiş ve doğru tahminler yapmasına yardımcı olabilir. Veri setiniz ne kadar büyük olursa, modeliniz o kadar iyi öğrenir.

•Veriyi Düzenleme ve Temizleme: Veri setindeki gürültüyü, eksik veya yanlış verileri temizleyerek modelin performansını artırabilirsiniz. Ayrıca, gereksiz veya korelasyonu yüksek özellikleri kaldırarak modelin daha iyi öğrenmesini sağlayabilirsiniz.

•Modelin Eğitim Süresini Artırma: Modelin daha uzun süre eğitilmesi, daha karmaşık ilişkileri öğrenmesine ve daha iyi performans göstermesine yardımcı olabilir. Ancak, overfitting riskini de artırabilir, bu nedenle dikkatli olunmalıdır.

•Çapraz Doğrulama ve Parametre Ayarı: Çapraz doğrulama kullanarak modelin genelleştirme yeteneğini değerlendirin ve modelin hiperparametrelerini ayarlayın.
