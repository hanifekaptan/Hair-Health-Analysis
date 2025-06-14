# Kullanım Kılavuzu

Bu belge, Saç Sağlığı Analiz Sistemi'nin nasıl kullanılacağını detaylı olarak açıklar.


### Adımlar

1. Projeyi klonlayın:
```bash
git clone https://github.com/hanifekaptan/hair-health-analysis.git
cd hair-health-analysis
```

2. Sanal ortam oluşturun ve aktifleştirin:
```bash
python -m venv hair-health-analysis
source hair-health-analysis/bin/activate  # Linux/Mac
hair-health-analysis\Scripts\activate     # Windows
```

3. Gerekli paketleri yükleyin:
```bash
pip install -e .
```

## Temel Kullanım

### 1. Saç Deri Hastalığı Analizi

```python
from main import HairDiseasesClassificationApp
classifier = HairDiseasesClassificationApp()
classifier.api()
```

### 2. Saç Stili Analizi

```python
from main import HairstyleClassificationApp
classifier = HairstyleClassificationApp()
classifier.api()
```

### 3. Saç Segmentasyonu

```python
from main import HairstyleSegmentationApp
segmenter = HairstyleSegmentationApp()
segmenter.api()
```
