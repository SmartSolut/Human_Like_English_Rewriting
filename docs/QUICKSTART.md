# دليل البدء السريع / Quick Start Guide

## البدء السريع / Quick Start

### 1. التثبيت / Installation

```bash
# إنشاء بيئة افتراضية / Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# تثبيت المكتبات / Install dependencies
pip install -r requirements.txt
```

### 2. تحميل البيانات / Download Data

```bash
python src/data/downloader.py
```

**ملاحظة**: قد يستغرق هذا بعض الوقت حسب سرعة الإنترنت.

**Note**: This may take some time depending on internet speed.

### 3. تجهيز البيانات / Preprocess Data

```bash
python src/data/preprocessor.py
```

### 4. تدريب النموذج / Train Model

```bash
python src/training/trainer.py
```

**ملاحظة**: التدريب قد يستغرق عدة ساعات حسب حجم البيانات والـ GPU.

**Note**: Training may take several hours depending on data size and GPU.

### 5. تشغيل API / Run API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

افتح المتصفح على: `http://localhost:8000`

Open browser at: `http://localhost:8000`

## استخدام سكريبتات التشغيل / Using Pipeline Scripts

### Linux/Mac:

```bash
chmod +x scripts/run_pipeline.sh
./scripts/run_pipeline.sh
```

### Windows:

```bash
scripts\run_pipeline.bat
```

## استخدام Docker / Using Docker

```bash
# بناء الصورة / Build image
docker build -t paraphrase-api .

# تشغيل الحاوية / Run container
docker-compose up -d
```

## استخدام API / Using API

### مثال باستخدام curl:

```bash
curl -X POST "http://localhost:8000/api/rewrite" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog.",
    "tone": "casual",
    "strength": "medium"
  }'
```

### مثال باستخدام Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/rewrite",
    json={
        "text": "The quick brown fox jumps over the lazy dog.",
        "tone": "casual",
        "strength": "medium"
    }
)

print(response.json())
```

## استكشاف الأخطاء / Troubleshooting

### المشكلة: النموذج غير موجود / Model Not Found

**الحل**: تأكد من تدريب النموذج أولاً:

```bash
python src/training/trainer.py
```

**Solution**: Make sure to train the model first:

```bash
python src/training/trainer.py
```

### المشكلة: خطأ في الذاكرة / Out of Memory

**الحل**: قلل batch_size في `config.yaml`:

```yaml
training:
  batch_size: 4  # بدلاً من 8
  gradient_accumulation_steps: 8  # بدلاً من 4
```

**Solution**: Reduce batch_size in `config.yaml`:

```yaml
training:
  batch_size: 4  # Instead of 8
  gradient_accumulation_steps: 8  # Instead of 4
```

### المشكلة: بطء في التحميل / Slow Download

**الحل**: قلل max_samples في `config.yaml` للاختبار:

```yaml
data:
  datasets:
    - name: "jpwahle/machine-paraphrase-dataset"
      max_samples: 10000  # بدلاً من 50000
```

**Solution**: Reduce max_samples in `config.yaml` for testing:

```yaml
data:
  datasets:
    - name: "jpwahle/machine-paraphrase-dataset"
      max_samples: 10000  # Instead of 50000
```

## الخطوات التالية / Next Steps

1. جرب النموذج على نصوصك الخاصة
2. اضبط معاملات التدريب في `config.yaml`
3. أضف بيانات تدريب إضافية
4. حسّن النموذج بناءً على التقييم

1. Try the model on your own texts
2. Adjust training parameters in `config.yaml`
3. Add additional training data
4. Improve the model based on evaluation








