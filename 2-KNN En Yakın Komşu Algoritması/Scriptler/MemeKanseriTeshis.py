from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Meme kanseri veri setini yükle
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Eğitim ve test veri kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Veriyi ölçeklendir
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN modelini oluştur ve eğit
k = 5  # K değeri
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Test verileri üzerinde tahmin yap
y_pred = knn.predict(X_test)

# Doğruluk hesapla
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

"""
Çıktı:Accuracy: 95.91%

Bu kod, Wisconsin Meme Kanseri veri seti üzerinde KNN 
sınıflandırmasını uygular. Veri seti, meme kanseri hücrelerinin 
bazı özelliklerine dayalı olarak kanserin iyi huylu veya kötü huylu 
olup olmadığını sınıflandırmak için kullanılır. Kod, eğitim ve
test veri kümelerini oluşturur, veriyi ölçeklendirir, KNN modelini 
oluşturur, eğitir, test verileri üzerinde tahmin yapar ve doğruluk oranını hesaplar.
"""
