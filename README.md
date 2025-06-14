# Saç Sağlığı Analizi Projesi

Bu proje, yapay zeka kullanarak saç sağlığı analizi yapan bir sistemdir. Proje, saç hastalıklarının sınıflandırılması, saç stillerinin sınıflandırılması ve saç segmentasyonu gibi farklı görevleri içermektedir.

## 📁 Klasör Yapısı

```
hair_health_analysis/
├── src/
│   ├── config/
│   │   ├── config_data_loading.py
│   │   ├── config_model.py
│   │   ├── config_data_path.py
│   │   └── config_saved_model_path.py
│   ├── hair_diseases_classification/
│   │   ├── data_sample/
│   │   ├── experiences/
│   │   ├── saved_models/
│   │   │   ├── logs/
│   │   │   ├── results/
│   │   │   └── best_model.keras
│   │   ├── api.py
│   │   ├── inference.py
│   │   ├── model.py
│   │   └── training.py
│   ├── hairstyle_classification/
│   │   ├── original_data_sample/
│   │   ├── overlayed_data_sample/
│   │   ├── experiences/
│   │   ├── saved_models/
│   │   │   ├── logs/
│   │   │   ├── results/
│   │   │   └── best_model.keras
│   │   ├── api.py
│   │   ├── inference.py
│   │   ├── model.py
│   │   └── training.py
│   ├── hairstyle_segmentation/
│   │   ├── data_sample/
│   │   ├── experiences/
│   │   ├── saved_models/
│   │   │   ├── logs/
│   │   │   ├── results/
│   │   │   └── best_model.keras
│   │   ├── api.py
│   │   ├── inference.py
│   │   ├── model.py
│   │   └── training.py
│   └── utils/
│       ├── class_evaluation.py
│       ├── data_loading.py
│       ├── data_preprocessing.py
│       └── training.py
├── docs/
│   ├── references.md
│   ├── hair_diseases_classification.md
│   ├── hairstyle_classification.md
│   ├── hairstyle_segmentation.md
│   ├── config.md
│   └── usage.md
├── main.py
├── setup.py
├── README.md
└── LICENSE
```

## 🚀 Özellikler

- **Saç Hastalıkları Sınıflandırma**: Saç hastalığı olduğu bilinen kişilerde 10 farklı saç hastalığını tespit eder.
- **Saç Stili Sınıflandırma**: Görüntüdeki saçı tespit ederek saç stilini tahmin eder.
- **Saç Segmentasyonu**: Saç bölgelerini görüntüden ayırır
- **Folikül Tespiti**: Henüz eklenmedi
- **API Desteği**: RESTful API ile kolay entegrasyon

## 🛠️ Kurulum

**1. Python 3.12.3 sürümü yükleyin.**

**2. Projeyi klonlayın:**
```bash
git clone https://github.com/hanifekaptan/hair_health_analysis.git
cd hair_health_analysis
```

**3. Sanal ortam oluşturun ve aktifleştirin:**
```bash
# Windows için
python -m venv hair_health_analysis
hair_health_analysis\Scripts\activate

# Linux/Mac için
python -m venv hair_health_analysis
source hair_health_analysis/bin/activate
```

**4. Gerekli paketleri yükleyin:**
```bash
pip install -e .
```


## 💻 Kullanım

### Model Eğitimi

```python
from main import HairDiseasesClassificationApp, HairstyleClassificationApp, HairStyleSegmentationApp

# Saç hastalıkları sınıflandırma modelini eğitme
hair_diseases = HairDiseasesClassificationApp()
hair_diseases.train()

# Saç stili sınıflandırma modelini eğitme
hairstyle = HairstyleClassificationApp()
hairstyle.train()

# Saç segmentasyon modelini eğitme
segmentation = HairStyleSegmentationApp()
segmentation.train()
```

### Model Değerlendirme

```python
# Model değerlendirme
hair_diseases.evaluate()
hairstyle.evaluate()
segmentation.evaluate()
```

### API Kullanımı

```python
# hair diseases classification için API'yi başlatma
hair_diseases.api()  # http://127.0.0.1:8000 adresinde çalışır
```

```python
# hairstle classification için API'yi başlatma
hairstyle.api()  # http://127.0.0.1:8000 adresinde çalışır
```

```python
# hairstyle segmentation için API'yi başlatma
segmentation.api()  # http://127.0.0.1:8000 adresinde çalışır
```
## 📊 Model Performansı

- Saç Hastalıkları Sınıflandırma: 0.90 accuracy
- Saç Stili Sınıflandırma: 0.67 accuracy
- Saç Segmentasyonu: 0.95 accuracy


## 📝 Lisans

Bu proje Apache License 2.0 altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.


## 📞 İletişim

Hanife Kaptan
- Email: [hanifekaptan.dev@gmail.com](mailto:hanifekaptan.dev@gmail.com)
- LinkedIn: [Hanife Kaptan](https://www.linkedin.com/in/hanifekaptan/)

