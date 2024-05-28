import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Hasta verileri ile ilgili veri setini yükle
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
data = pd.read_csv(url)

# Bağımsız ve bağımlı değişkenleri ayır
X = data.drop('class', axis=1)
y = data['class']

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
Bu kod örneği, KNN sınıflandırmasını kullanarak 
Veri seti, üç farklı çiçek türünü içeren ünlü Iris veri setidir. 
Veri seti, çiçeklerin dört farklı özelliğini içerir (çiçek boyutları gibi) 
ve her bir çiçeğin sınıfını (çiçek türünü) belirtir. Kod, veri setini yükler, 
bağımsız ve bağımlı değişkenleri ayırır, eğitim ve test veri kümelerine ayırır, 
veriyi ölçeklendirir, KNN sınıflandırma modelini oluşturur, eğitir, test 
verileri üzerinde tahmin yapar ve doğruluk oranını hesaplar.
"""
