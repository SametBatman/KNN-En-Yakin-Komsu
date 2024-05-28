import pandas as pd #veri analizi ve manipülasyonu için kullanılan kütüphanedir
from sklearn.model_selection import train_test_split # Verileri eğitim ve test kümelerine ayırmak için fonksiyon
from sklearn.preprocessing import StandardScaler # Verileri ölçeklendirmek için sınıf
from sklearn.neighbors import KNeighborsClassifier # K-En Yakın Komşu sınıflandırma algoritması
from sklearn.metrics import accuracy_score # Doğruluk hesaplamak için fonksiyon

# Hasta verileri ile ilgili veri setini yükler
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
data = pd.read_csv(url)

# Bağımsız ve bağımlı değişkenleri ayırır
X = data.drop('class', axis=1) # Özellikler (bağımsız değişkenler)
y = data['class'] # Hedef (bağımlı değişken)

# Eğitim ve test veri kümelerine ayırır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# "train_test_split" fonksiyonu ile verileri %80 eğitim, %20 test kümesi olarak ayırıyor.
# "test_size" parametresi test kümesinin oranını belirliyor (bu örnekte %20).
# "random_state" parametresi her çalıştırmada farklı bölünmeler olmaması için rastgele sayı üreteci için bir başlangıç değeri ayarlıyor

# Veriyi ölçeklendirir
scaler = StandardScaler()  # "StandardScaler" sınıfı, verilerin ortalaması sıfır ve varyansı bire olacak şekilde ölçeklenmesini sağlıyor.
X_train = scaler.fit_transform(X_train) # "fit_transform" metodu önce ölçekleme parametrelerini hesaplıyor (fit) sonra verileri ölçeklendiriyor (transform).
X_test = scaler.transform(X_test) # Eğitim verisine göre hesaplanan ölçekleme parametreleri test verisine de uygulanıyor ("scaler.transform(X_test)").

# KNN sınıflandırma modelini oluştur ve eğitir
k = 5  # K değeri (komşu sayısı)
knn_clf = KNeighborsClassifier(n_neighbors=k)  # "KNeighborsClassifier" sınıfı, K-En Yakın Komşu algoritmasını kullanarak bir sınıflandırma modeli oluşturuyor.
knn_clf.fit(X_train, y_train)
# "n_neighbors" parametresi, sınıflandırma işleminde en yakın kaç komşunun oy kullanacağını belirliyor
# "fit" metodu, modeli eğitim verileri üzerinde eğiterek öğrenmesini sağlıyor.

# Test verileri üzerinde tahmin yapar
y_pred = knn_clf.predict(X_test)

# Doğruluk hesaplar
accuracy = accuracy_score(y_test, y_pred) # Doğruluğu hesaplar
print(f'Accuracy: {accuracy * 100:.2f}%') # Doğruluğu yüzde formatında yazdırır

"""
Bu kod örneği, KNN sınıflandırmasını kullanarak 
Veri seti, üç farklı çiçek türünü içeren ünlü Iris veri setidir. 
Veri seti, çiçeklerin dört farklı özelliğini içerir (çiçek boyutları gibi) 
ve her bir çiçeğin sınıfını (çiçek türünü) belirtir. Kod, veri setini yükler, 
bağımsız ve bağımlı değişkenleri ayırır, eğitim ve test veri kümelerine ayırır, 
veriyi ölçeklendirir, KNN sınıflandırma modelini oluşturur, eğitir, test 
verileri üzerinde tahmin yapar ve doğruluk oranını hesaplar.
"""
