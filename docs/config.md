# Konfigürasyon Modülü

Bu modül, projenin tüm yapılandırma ayarlarını yönetmek için geliştirilmiştir.

## Özellikler

- Merkezi yapılandırma yönetimi
- Ortam bazlı ayarlar
- Dinamik konfigürasyon değişiklikleri
- Güvenli ayar yönetimi

## Yapılandırma Bileşenleri

### 1. Model Ayarları
```python
MODEL_CONFIG = {
    "hair_disease_classification": {
        "model_type": "mobilenet",
        "pretrained": True,
        "num_classes": 10,
        "input_size": (224, 224)
    },
    "hairstyle_classification": {
        "model_type": "mobilenet",
        "pretrained": True,
        "num_classes": 7,
        "input_size": (224, 224)
    },
    "hairstyle_segmentation": {
        "model_type": "unet",
        "encoder": "mobilenet",
        "input_size": (224, 224)
    }
}
```

### 2. Veri İşleme Ayarları
```python
DATA_CONFIG = {
    "train_data_path": "data/train",
    "val_data_path": "data/val",
    "test_data_path": "data/test",
    "batch_size": 16
}
```

### 3. Eğitim Ayarları
```python
TRAINING_CONFIG = {
    "learning_rate": flexible,
    "epochs": 50,
    "early_stopping": 10,
    "optimizer": "adam"
}
```

### 4. API Ayarları
```python
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False
}

```
