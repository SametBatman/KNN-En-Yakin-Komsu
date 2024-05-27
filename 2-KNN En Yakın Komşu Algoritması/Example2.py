from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Iris veri setini yükle
iris = load_iris()
X = iris.data
y = iris.target

# Eğitim ve test veri kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=422)

# Veriyi ölçeklendir
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN modelini oluştur ve eğit
k = 3  # K değeri
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Test verileri üzerinde tahmin yap
y_pred = knn.predict(X_test)

# Doğruluk hesapla
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


"""
Çıktı:Accuracy: 93.33%

Bu kod, Iris veri seti üzerinde KNN algoritmasının en yakın komşu versiyonunu uygular.
Veri seti, üç farklı iris çiçeği türünü sınıflandırmak için kullanılır. Kullanıcıdan gelen girişlere dayalı 
olarak, kod, eğitim ve test veri kümelerini oluşturur, veriyi ölçeklendirir, 
KNN modelini oluşturur, eğitir ve son olarak test verileri üzerinde tahmin yapar ve doğruluğu hesaplar.

"""