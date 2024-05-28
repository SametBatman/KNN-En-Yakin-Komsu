import numpy as np #Sayısal işlemler için kullanılan güçlü bir kütüphanedir
from sklearn.model_selection import train_test_split # Verileri eğitim ve test kümelerine ayırmak için fonksiyon
from sklearn.preprocessing import StandardScaler # Verileri ölçeklendirmek için sınıf
from sklearn.neighbors import KNeighborsClassifier # K-En Yakın Komşu sınıflandırma algoritması
from sklearn.metrics import accuracy_score  # Doğruluk hesaplamak için fonksiyon

# Rastgele veri oluşturma için tohum belirle
# Bu, her çalıştırmada aynı rastgele sayı dizisinin üretilmesini sağlar.
np.random.seed(42)

# Örnek veri seti oluşturma
num_samples = 1000  # Örnek sayısı
num_features = 100  # Özellik sayısı

# Özellik matrisi X ve hedef vektörü y oluşturma
# X rastgele normal dağılım gösteren sayılardan oluşur
X = np.random.randn(num_samples, num_features)
# y rastgele 0 veya 1 değerlerinden oluşur (binary sınıflandırma)
y = np.random.randint(0, 2, size=num_samples)

# Eğitim ve test veri kümelerine ayırma
# Verinin %80'i eğitim, %20'si test için kullanılır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendirme (standardizasyon)
# StandardScaler kullanarak özelliklerin ortalamasını 0, standart sapmasını 1 yapar
scaler = StandardScaler()
# Eğitim verisi üzerinde fit ve transform uygular
X_train = scaler.fit_transform(X_train)
# Test verisine sadece transform uygular
X_test = scaler.transform(X_test)

# KNN sınıflandırma modelini oluşturma ve eğitir
k = 5  # K değeri (komşu sayısı)
knn_clf = KNeighborsClassifier(n_neighbors=k)
# Eğitim verisi ile modeli eğitir
knn_clf.fit(X_train, y_train)

# Test verileri üzerinde tahmin yapar
y_pred = knn_clf.predict(X_test)

# Doğruluk hesaplama
# accuracy_score fonksiyonu ile tahminlerin doğruluğunu hesaplar
accuracy = accuracy_score(y_test, y_pred)

# Doğruluğu yüzde formatında yazdırır
print(f'Accuracy: {accuracy * 100:.2f}%')


"""
Çıktı:Accuracy: 52.50%

MRI görüntüsünden bir hastalığın varlığını veya 
yokluğunu belirlemek için KNN sınıflandırmasını 
kullanabiliriz. Bu durumda, veri seti MRI görüntülerini 
ve her görüntünün hastalığın varlığını veya yokluğunu 
içeren etiketleri içerecektir.
"""
