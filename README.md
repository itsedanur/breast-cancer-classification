#  Göğüs Kanseri Sınıflandırması (Breast Cancer Classification)

Bu projede, göğüs kanseri verisi kullanılarak **K-Nearest Neighbors (KNN)** algoritmasıyla sınıflandırma yapılmıştır. Veriye öncelikle **PCA (Principal Component Analysis)** uygulanarak boyut indirgeme gerçekleştirilmiştir. Ardından en iyi KNN parametreleri `GridSearchCV` ile bulunmuş ve model başarı metrikleri görselleştirilmiştir.

---

##  Proje Yapısı

breast-cancer-project/
│
├── breast-cancer-project.py # Ana Python kodu (Spyder)
├── README.md # açıklama dosyası


##  Kullanılan Yöntemler

-  **Veri Ön İşleme**
  - Eksik verilerin silinmesi
  - StandardScaler ile normalize etme
-  **Aykırı Değer Tespiti**
  - LocalOutlierFactor (LOF)
-  **Boyut İndirgeme**
  - PCA (2 bileşene indirgeme)
-  **Modelleme**
  - KNN
  - GridSearchCV ile hiperparametre optimizasyonu
-  **Değerlendirme**
  - Confusion Matrix
  - Accuracy Score
  - Karar sınırı (decision boundary) görselleştirmesi

---

## Kullanılan Kütüphaneler

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Kurmak için:
```bash
pip install -r requirements.txt

� Veri Kümesi

Veri, UCI Machine Learning Repository'deki Breast Cancer Wisconsin (Diagnostic) Data Set'ten alınmıştır:

 Veri kümesi linki

https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

 Çalıştırmak İçin

python breast-cancer-project.py


Eda Nur Unal
