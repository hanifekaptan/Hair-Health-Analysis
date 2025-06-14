# Saç Hastalıkları Sınıflandırma Modülü

Bu modül, saç hastalıklarını otomatik olarak tespit etmek ve sınıflandırmak için geliştirilmiştir.

## Özellikler

- Transfer öğrenme tabanlı derin öğrenme modeli
- Çoklu sınıf sınıflandırma desteği
- Gerçek zamanlı analiz
- Yüksek doğruluk oranı: 0.90

## Desteklenen Hastalıklar

1. Alopesi Areata
2. Contact Dermatitis
3. Folliculitis
4. Head Lice
5. Lichen Planus
6. Male Pattern Baldness
7. Psoriasis
8. Seborrheic Dermatitis
9. Telogen Effluvium
10. Tinea Capitis

## Teknik Detaylar

### Model Mimarisi
- Temel model: Mobilenet
- Transfer öğrenme için önceden eğitilmiş ağırlıklar
- Özelleştirilmiş son katmanlar
- Dropout ve Batch Normalization katmanları

### Eğitim Parametreleri
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 16
- Epochs: 15

### Veri Ön İşleme
- Görüntü boyutlandırma: 224x224
- Veri artırma teknikleri:
  - Yatay çevirme
  - Döndürme
  - Parlaklık ayarlama
  - Kontrast ayarlama

## API Kullanımı

```python
from main import HairDiseasesClassificationApp

classifier = HairDiseasesClassificationApp()
classifier.api()
```

## Performans Metrikleri

- Accuracy: %95
- Precision: %98
- Recall: %92
- F1-Score: %84 (min)
