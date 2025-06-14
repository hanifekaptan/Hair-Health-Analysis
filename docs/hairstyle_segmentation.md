# Saç Stili Segmentasyon Modülü

Bu modül, saç bölgelerini otomatik olarak tespit etmek ve segmentasyon yapmak için geliştirilmiştir.

## Özellikler

- Mobilenet tabanlı derin öğrenme modeli
- Piksel seviyesinde segmentasyon
- Gerçek zamanlı analiz
- Yüksek doğruluk oranı

## Teknik Detaylar

### Model Mimarisi
- Encoder-Decoder yapısı
- U-Net tabanlı mimari
- Skip connections

### Eğitim Parametreleri
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 50
- Epochs: 50
- Loss Function: Categorical Crossentropy

### Veri Ön İşleme
- Görüntü boyutlandırma: 224x224
- Veri artırma teknikleri: Kullanılmadı

## API Kullanımı

```python
from main import HairstyleSegmentationApp

segmenter = HairstyleSegmentationApp()

mask = segmenter.api()
```

## Performans Metrikleri

- Ortalama IoU (Intersection over Union): 0.9188
- Accuracy: 0.98
