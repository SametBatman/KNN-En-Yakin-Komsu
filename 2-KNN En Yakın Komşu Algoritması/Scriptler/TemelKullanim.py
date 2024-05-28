import pandas as pd  # Verileri işleme ve analiz etme kütüphanesi
from sklearn.model_selection import train_test_split  # Verileri eğitim ve test kümelerine ayırma fonksiyonu içerir.
from sklearn.neighbors import KNeighborsClassifier  # K-En Yakın Komşu sınıflandırma algoritmasını içerir.

# Verileri yüklemek
data = pd.read_csv("data.csv")  # CSV dosyasından veri okunur

# data değişkeninden "öznitelik1", "öznitelik2", ... gibi özelliklerin bulunduğu X adlı bir veri çerçevesi oluşturulur.
# data değişkeninden "hedef_değişken" adlı hedef değişken ayrılarak y adlı bir değişkene atanır.
X = data[["öznitelik1", "öznitelik2", ...]]  # Özellikler seçilir
y = data["hedef_değişken"]  # Hedef değişken seçilir

# Eğitim kümesi X_train ve y_train değişkenlerine, test kümesi ise X_test ve y_test değişkenlerine atanır.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # Veriler train_test_split %80 eğitim, %20 test kümesi olarak ayrılır

# KNeighborsClassifier sınıfı kullanılarak bir KNN modeli oluşturulur.
# n_neighbors parametresi 5 olarak ayarlanır, bu da modelin sınıflandırma işleminde en yakın 5 komşunun oy kullanacağı anlamına gelir.

knn = KNeighborsClassifier(n_neighbors=5)  # KNN modeli oluşturulur, k değeri 5 olarak belirlenir
knn.fit(X_train, y_train)  # Oluşturulan model X_train ve y_train eğitim verisi üzerinde eğitilir.

# yeni_değer1 ve yeni_değer2 gibi yeni veri noktasının değerleri belirlenir.
yeni_değer1 = 1.2  # Yeni veri noktasının ilk değeri
yeni_değer2 = 1.5  # Yeni veri noktasının ikinci değeri
yeni_veri = [yeni_değer1, yeni_değer2, ...]  # Yeni veri noktası "yeni_veri" adlı bir listeye dönüştürülür.
#Eğitimli KNN modeli yeni_veri listesini kullanarak yeni veri noktasının sınıfını tahmin eder.

# Tahmini yazdır
tahmin = knn.predict(yeni_veri)  # Yeni veri noktası için sınıf tahmini yapılır
print("Tahmin edilen sınıf:", tahmin)  # Tahmin ekrana yazdırılır


'''
Tahmin edilen sınıf: 1
Doğruluk skoru: 0.92

Bu kod, pandas kütüphanesini kullanarak CSV dosyasından veri okur.
sklearn.model_selection modülünden train_test_split fonksiyonu ile veriler eğitim ve test kümelerine ayrılır.
sklearn.neighbors modülünden KNeighborsClassifier sınıfı kullanılarak KNN modeli oluşturulur.
n_neighbors parametresi, dikkate alınacak en yakın komşu sayısını belirler.
fit fonksiyonu ile model eğitim kümesi üzerinde eğitilr.
predict fonksiyonu ile yeni bir veri noktası için sınıf tahmini yapılır.
KNN sınıflandırma algoritmasının temel kullanımını gösterir ,veri setinin özelliklerine dayalı olarak verilerin sınıflandırılması vb.
'''
