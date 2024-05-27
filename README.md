<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>K-Nearest Neighbors (KNN) Algoritması</title>
</head>
<body>
    <h1><strong>K-Nearest Neighbors (KNN) Algoritması</strong></h1>

    <h2><strong>KNN Nedir?</strong></h2>
    <p>
        <strong>K-Nearest Neighbors (KNN)</strong>, denetimli öğrenme algoritmaları arasında yer alan ve hem sınıflandırma hem de regresyon problemlerinde kullanılan basit ama güçlü bir algoritmadır. KNN, yeni bir veri noktasını sınıflandırmak veya değerini tahmin etmek için en yakın <strong>k</strong> komşusunu temel alır. Komşular arasındaki mesafeyi hesaplayarak, yeni veri noktasının hangi sınıfa ait olduğunu veya hangi değeri alacağını belirler.
    </p>

    <h2><strong>KNN Nasıl Çalışır?</strong></h2>
    <ol>
        <li><strong>Veri Hazırlığı:</strong> Verisetini eğitim ve test seti olarak ayırın.</li>
        <li><strong>Mesafe Hesaplama:</strong> Yeni bir veri noktası ile mevcut veri noktaları arasındaki mesafeyi hesaplayın. En yaygın kullanılan mesafe ölçütü Öklidyen mesafedir.</li>
        <li><strong>En Yakın Komşuların Seçimi:</strong> Hesaplanan mesafelere göre en yakın <strong>k</strong> komşuyu seçin.</li>
        <li><strong>Sınıflandırma veya Tahmin:</strong>
            <ul>
                <li><strong>Sınıflandırma:</strong> En yakın <strong>k</strong> komşunun sınıflarına göre yeni veri noktasının sınıfını belirleyin. Genellikle çoğunluk oylaması kullanılır.</li>
                <li><strong>Regresyon:</strong> En yakın <strong>k</strong> komşunun değerlerinin ortalamasını alın.</li>
            </ul>
        </li>
    </ol>

    <h2><strong>KNN'in Avantajları ve Dezavantajları</strong></h2>
    <h3><strong>Avantajlar:</strong></h3>
    <ul>
        <li>Basit ve kolay uygulanabilir.</li>
        <li>Parametrik olmayan bir model olduğu için herhangi bir dağılım varsayımı gerektirmez.</li>
        <li>Eğitim aşaması çok hızlıdır.</li>
    </ul>

    <h3><strong>Dezavantajlar:</strong></h3>
    <ul>
        <li>Büyük veri setlerinde yavaş çalışabilir, çünkü her yeni veri noktası için tüm veri seti taranır.</li>
        <li>Özelliklerin ölçeklendirilmesine duyarlıdır, bu yüzden normalizasyon gereklidir.</li>
        <li>Gürültülü veri setlerinde performansı düşebilir.</li>
    </ul>

    <h2><strong>KNN Uygulama Örneği</strong></h2>

    <h3><strong>Gerekli Kütüphaneler</strong></h3>
    <pre><code>import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
</code></pre>

    <h3><strong>Veri Kümesi</strong></h3>
    <p>Örnek olarak Iris veri setini kullanacağız. Bu veri seti, üç farklı iris çiçeği türünü (Setosa, Versicolor, Virginica) içerir ve her bir çiçeğin dört farklı özelliği (sepal length, sepal width, petal length, petal width) bulunmaktadır.</p>

    <pre><code># Veri setini yükleme
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Verilerin ölçeklendirilmesi
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN modelinin oluşturulması
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Tahmin
y_pred = knn.predict(X_test)

# Sonuçların değerlendirilmesi
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("Karışıklık Matrisi:\n", confusion_matrix(y_test, y_pred))
</code></pre>

    <h3><strong>Açıklama</strong></h3>
    <ol>
        <li><strong>Veri Kümesini Yükleme:</strong> <code>load_iris()</code> fonksiyonu ile Iris veri setini yüklüyoruz.</li>
        <li><strong>Veri Setini Ayırma:</strong> <code>train_test_split</code> fonksiyonu ile veri setini eğitim ve test setlerine ayırıyoruz.</li>
        <li><strong>Özelliklerin Ölçeklendirilmesi:</strong> <code>StandardScaler</code> kullanarak özellikleri ölçeklendiriyoruz. Bu adım, KNN algoritmasının performansını artırmak için önemlidir.</li>
        <li><strong>KNN Modeli Oluşturma:</strong> <code>KNeighborsClassifier</code> sınıfını kullanarak KNN modelimizi oluşturuyoruz. Bu örnekte, <strong>k</strong> değerini 5 olarak seçtik.</li>
        <li><strong>Modeli Eğitme ve Tahmin:</strong> Eğitim verisi ile modeli eğitiyoruz ve test verisi üzerinde tahmin yapıyoruz.</li>
        <li><strong>Sonuçların Değerlendirilmesi:</strong> Doğruluk skoru, sınıflandırma raporu ve karışıklık matrisi ile modelimizin performansını değerlendiriyoruz.</li>
    </ol>
</body>
</html>
