from sklearn.datasets import load_breast_cancer  # Meme kanseri veri seti yüklemek için kütüphane
from sklearn.model_selection import train_test_split  # Verileri eğitim ve test kümelerine ayırmak için fonksiyon
from sklearn.preprocessing import StandardScaler  # Verileri ölçeklendirmek için sınıf
from sklearn.neighbors import KNeighborsClassifier  # K-En Yakın Komşu sınıflandırma algoritması
from sklearn.metrics import accuracy_score  # Doğruluk hesaplamak için fonksiyon

# Meme kanseri veri setini yükle
cancer = load_breast_cancer()
X = cancer.data  # Veri setindeki özellikler "X" değişkenine atanıyor
y = cancer.target  # Veri setindeki hedef değişken (kanserli/sağlıklı) "y" değişkenine atanıyor

# Eğitim ve test veri kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.13, random_state=2)
  # "train_test_split" fonksiyonu ile verileri %87 eğitim, %13 test kümesi olarak ayırıyor.
  # "test_size" parametresi test kümesinin oranını belirliyor (bu örnekte %13).
  # "random_state" parametresi her çalıştırmada farklı bölünmeler olmaması için rastgele sayı üreteci için bir başlangıç değeri ayarlıyor (bu örnekte 2).

# Veriyi ölçeklendir
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Eğitim verileri ölçeklendiriliyor
X_test = scaler.transform(X_test)  # Test verileri ölçeklendiriliyor
  # "StandardScaler" sınıfı, verilerin ortalaması sıfır ve varyansı bire olacak şekilde ölçeklenmesini sağlıyor.
  # Bu işlem, farklı birimlerdeki özelliklerin birbirini etkilememesini ve modelin daha iyi öğrenmesini sağlar.
  # "fit_transform" metodu önce ölçekleme parametrelerini hesaplıyor (fit) sonra verileri ölçeklendiriyor (transform).
  # Eğitim verisine göre hesaplanan ölçekleme parametreleri test verisine de uygulanıyor ("scaler.transform(X_test)").

# KNN modelini oluştur ve eğit
k = 5  # K değeri (en yakın kaç komşu kullanılacak)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)  # Model, eğitim verileri üzerinde eğitiliyor
  # "KNeighborsClassifier" sınıfı, K-En Yakın Komşu algoritmasını kullanarak bir sınıflandırma modeli oluşturuyor.
  # "n_neighbors" parametresi, sınıflandırma işleminde en yakın kaç komşunun oy kullanacağını belirliyor (bu örnekte 5).
  # "fit" metodu, modeli eğitim verileri üzerinde eğiterek öğrenmesini sağlıyor.

# Test verileri üzerinde tahmin yap
y_pred = knn.predict(X_test)  # Model, test verileri üzerinde tahmin yapıyor
  # "predict" metodu, eğitilmiş modele yeni bir veri noktası (veya noktaları) verildiğinde, bu noktanın hangi sınıfa ait olabileceğini tahmin ediyor.

# Doğruluk hesapla
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')  # Doğruluk yüzdesi ekrana yazdırılıyor
  # "accuracy_score" fonksiyonu, tahminlerin gerçek değerlerle karşılaştırılarak modelin doğruluk oranını hesaplıyor.
  # "f-string" ile hesaplanan doğruluk değeri "%" sembolü ile birlikte ekrana yazdırılıyor.

"""
Çıktı:Accuracy: 97.30%

Bu kod, Wisconsin Meme Kanseri veri seti üzerinde KNN 
sınıflandırmasını uygular. Veri seti, meme kanseri hücrelerinin 
bazı özelliklerine dayalı olarak kanserin iyi huylu veya kötü huylu 
olup olmadığını sınıflandırmak için kullanılır. Kod, eğitim ve
test veri kümelerini oluşturur, veriyi ölçeklendirir, KNN modelini 
oluşturur, eğitir, test verileri üzerinde tahmin yapar ve doğruluk oranını hesaplar.
"""
