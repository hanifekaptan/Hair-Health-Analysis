# Saç Stili Sınıflandırma Modülü

Bu modül, farklı saç stillerini otomatik olarak tanımlamak ve sınıflandırmak için geliştirilmiştir.

## Özellikler

- Derin öğrenme tabanlı sınıflandırma
- Çoklu stil tanıma
- Gerçek zamanlı analiz
- Yüksek doğruluk oranı

## Desteklenen Saç Stilleri

1. Braids
2. Curly
3. Dreadlocks
4. Kinky
5. Short-men
6. Straight
7. Wavy

## Teknik Detaylar

### Model Mimarisi
- Temel model: Mobilenet
- Özelleştirilmiş son katmanlar
- Global Average Pooling

### Eğitim Parametreleri
- Optimizer: Adam
- Learning Rate: 0.0001
- Batch Size: 16
- Epochs: 50

### Veri Ön İşleme
- Görüntü boyutlandırma: 224x224
- Veri artırma teknikleri: Kullanılmadı

## API Kullanımı

```python
from main import HairstyleClassificationApp

classifier = HairstyleClassificationApp()

result = classifier.api()
```

## Performans Metrikleri

- Accuracy: %94
- Precision: %93
- Recall: %92
- F1-Score: %92.5
