
# Python Kurulum Rehberi

**Python'u İndirin:**

•Resmi Python web sitesine gidin: https://www.python.org/downloads/  en son Python 3 sürümünü seçin.
64 bit işletim sistemi kullanıyorsanız, Windows x86-64 dosyasını indirin. 32 bit işletim sistemi kullanıyorsanız, Windows x86 dosyasını indirin.
İndirme işlemi tamamlandıktan sonra, kurulum dosyasını çalıştırın.

**Kurulumu Yapın**

Kurulum sihirbazında, "Add Python 3.x to PATH" seçeneğinin işaretli olduğundan emin olun. Bu, Python'u komut satırından çalıştırabilmenizi sağlayacaktır.
Diğer seçenekleri varsayılan değerlerde bırakabilirsiniz.
"Install Now" düğmesine tıklayın ve kurulumun tamamlanmasını bekleyin.

**Kurulumun Doğrulanması**
Başlat menüsünü açın ve "Komut İstemi"ni arayın.
Komut isteminde "python3 --version" yazın ve Enter tuşuna basın.
Python sürümünün doğru şekilde kurulduğunu gösteren bir mesaj görmelisiniz.

# KNN İçin Gerekli Paket ve Kütüphanelerin Kurulumu

K-En Yakın Komşu (KNN) algoritmasını kullanmak için genellikle scikit-learn kütüphanesi kullanılır. Bu kütüphane, Python'da makine öğrenimi modelleri oluşturmak için yaygın olarak kullanılan bir kütüphanedir ve KNN gibi çeşitli algoritmaları içerir.

<code> pip install scikit-learn </code>

Bu komut, Python'un pip paket yöneticisi aracılığıyla scikit-learn kütüphanesini indirir ve kurar. Bu kütüphane, KNN modelini oluşturmak ve eğitmek için gerekli olan tüm işlevleri sağlar.

Ek olarak genellikle veri manipülasyonu ve görselleştirmesi için pandas ve matplotlib gibi başka kütüphaneler de kullanılır. Bunları da yüklemek istiyorsanız, aşağıdaki komutları kullanabilirsiniz:

<code>pip install pandas matplotlib </code>
