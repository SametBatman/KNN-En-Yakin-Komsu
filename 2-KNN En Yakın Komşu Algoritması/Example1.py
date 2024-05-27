import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Verileri yükle
data = pd.read_csv("data.csv")

# Özellikleri ve hedef değişkeni ayır
X = data[["öznitelik1", "öznitelik2", ...]]
y = data["hedef_değişken"]

# Verileri eğitim ve test kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# KNN modelini oluştur ve eğit
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Yeni bir veri noktasını sınıflandır
yeni_değer1 = 1.2
yeni_değer2 = 1.5  # Ör
yeni_veri = [yeni_değer1, yeni_değer2, ...]
tahmin = knn.predict(yeni_veri)

# Tahmini yazdır
print("Tahmin edilen sınıf:", tahmin)
