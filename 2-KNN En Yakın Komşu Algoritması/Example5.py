import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Örnek veri seti oluşturma
np.random.seed(42)
num_samples = 1000
num_features = 100
X = np.random.randn(num_samples, num_features)
y = np.random.randint(0, 2, size=num_samples)  # Sınıflar: 0 veya 1

# Eğitim ve test veri kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendir
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN sınıflandırma modelini oluştur ve eğit
k = 5  # K değeri
knn_clf = KNeighborsClassifier(n_neighbors=k)
knn_clf.fit(X_train, y_train)

# Test verileri üzerinde tahmin yap
y_pred = knn_clf.predict(X_test)

# Doğruluk hesapla
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

"""
Çıktı:Accuracy: 52.50%

MRI görüntüsünden bir hastalığın varlığını veya 
yokluğunu belirlemek için KNN sınıflandırmasını 
kullanabiliriz. Bu durumda, veri seti MRI görüntülerini 
ve her görüntünün hastalığın varlığını veya yokluğunu 
içeren etiketleri içerecektir.
"""
