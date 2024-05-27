import pandas as pd  # Verileri işleme ve analiz etme kütüphanesi
from sklearn.model_selection import train_test_split  # Verileri eğitim ve test kümelerine ayırma fonksiyonu
from sklearn.neighbors import KNeighborsClassifier  # K-En Yakın Komşu sınıflandırma algoritması

# Verileri yükle
data = pd.read_csv("data.csv")  # CSV dosyasından veri okunur

# Özellikleri ve hedef değişkeni ayır
X = data[["öznitelik1", "öznitelik2", ...]]  # Özellikler seçilir
y = data["hedef_değişken"]  # Hedef değişken seçilir

# Verileri eğitim ve test kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # Veriler %80 eğitim, %20 test kümesi olarak ayrılır

# KNN modelini oluştur ve eğit
knn = KNeighborsClassifier(n_neighbors=5)  # KNN modeli oluşturulur, k değeri 5 olarak belirlenir
knn.fit(X_train, y_train)  # Model eğitim verisi üzerinde eğitilir

# Yeni bir veri noktasını sınıflandır
yeni_değer1 = 1.2  # Yeni veri noktasının ilk değeri
yeni_değer2 = 1.5  # Yeni veri noktasının ikinci değeri
yeni_veri = [yeni_değer1, yeni_değer2, ...]  # Yeni veri noktası bir listeye dönüştürülür

# Tahmini yazdır
tahmin = knn.predict(yeni_veri)  # Yeni veri noktası için sınıf tahmini yapılır
print("Tahmin edilen sınıf:", tahmin)  # Tahmin ekrana yazdırılır


'''
 Bu kod, pandas kütüphanesini kullanarak CSV dosyasından veri okur.
sklearn.model_selection modülünden train_test_split fonksiyonu ile veriler eğitim ve test kümelerine ayrılır.
sklearn.neighbors modülünden KNeighborsClassifier sınıfı kullanılarak KNN modeli oluşturulur.
n_neighbors parametresi, dikkate alınacak en yakın komşu sayısını belirler.
fit fonksiyonu ile model eğitim kümesi üzerinde eğitilr.
predict fonksiyonu ile yeni bir veri noktası için sınıf tahmini yapılır.
'''