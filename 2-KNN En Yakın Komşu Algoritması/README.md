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
K Parametresi: 
𝐾 değeri, dikkate alınacak komşu sayısını belirler. Küçük bir K değeri, modelin fazla uyum yapmasına (overfitting) neden olabilirken, büyük bir 𝐾 değeri ise modelin yeterince karmaşık olmamasına (underfitting) yol açabilir.

# KNN Algoritması Nasıl Çalışır?
**Adım Adım İşleyiş:**

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
