# KNN-En-Yakin-Komsu
KNN En Yakin Komsu Algoritması

K-Nearest Neighbors (KNN) Algoritması
KNN Nedir?
K-Nearest Neighbors (KNN), hem sınıflandırma hem de regresyon problemlerinde kullanılan, denetimli bir öğrenme algoritmasıdır. KNN algoritması, bir veri noktasının sınıfını veya değerini tahmin etmek için en yakın komşularının etiketlerini veya değerlerini kullanır. Algoritma, eğitim aşamasında sadece veriyi depolar ve tahmin aşamasında bu veriyi kullanarak yeni veriler için tahminlerde bulunur.

KNN Algoritması Nasıl Çalışır?
KNN algoritması, aşağıdaki adımları izleyerek çalışır:

Veri Hazırlığı: Veri seti, eğitim ve test seti olarak ikiye ayrılır. Eğitim seti, modelin öğrenmesi için kullanılırken test seti, modelin performansını değerlendirmek için kullanılır.

Mesafe Hesaplama: Yeni bir veri noktası ile eğitim veri setindeki tüm veri noktaları arasındaki mesafeyi hesaplar. En yaygın kullanılan mesafe ölçütü Öklidyen mesafedir, ancak Manhattan mesafesi, Minkowski mesafesi gibi diğer ölçütler de kullanılabilir.

En Yakın Komşuların Seçimi: Hesaplanan mesafelere göre en yakın k komşuyu seçer. Burada k, önceden belirlenmiş bir sabit sayıdır.

Sınıflandırma veya Tahmin:

Sınıflandırma: En yakın k komşunun sınıflarına göre yeni veri noktasının sınıfını belirler. Genellikle çoğunluk oylaması kullanılır.
Regresyon: En yakın k komşunun değerlerinin ortalamasını alarak yeni veri noktasının değerini tahmin eder.
KNN Algoritmasının Özellikleri
Parametrik Olmayan: KNN, parametrik olmayan bir algoritmadır, yani belirli bir model yapısına bağlı değildir ve herhangi bir dağılım varsayımı gerektirmez.
Basitlik: Uygulaması kolay ve anlaşılması basit bir algoritmadır.
Lokal: KNN, lokal olarak veri noktalarının komşuluk ilişkilerine dayanır ve genel bir model oluşturmaz.
KNN Algoritmasının Avantajları
Kolay Uygulanabilirlik: Basitliği ve kolay anlaşılabilirliği sayesinde birçok farklı problemde rahatlıkla kullanılabilir.
Eğitim Hızlıdır: Eğitim aşamasında model sadece veriyi saklar ve öğrenme işlemi yapmaz, bu nedenle eğitim aşaması çok hızlıdır.
Esneklik: Parametrik olmaması sayesinde çeşitli veri setlerinde esnek bir şekilde kullanılabilir.
KNN Algoritmasının Dezavantajları
Hesaplama Maliyeti: Büyük veri setlerinde yavaş çalışabilir, çünkü her yeni veri noktası için tüm veri seti taranmalıdır.
Bellek Kullanımı: Tüm eğitim verisini saklaması gerektiğinden, bellek kullanımı yüksektir.
Özellik Ölçeklendirme: Özelliklerin ölçeklendirilmesine duyarlıdır, bu yüzden normalizasyon veya standardizasyon gereklidir.
Gürültüye Duyarlılık: Gürültülü veri setlerinde performansı düşebilir, çünkü komşuluk ilişkileri gürültüden etkilenebilir.