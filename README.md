# Human-Like English Rewriting System

نظام لإعادة صياغة النصوص الإنجليزية بأسلوب بشري طبيعي مع الحفاظ الكامل على المعنى.

A system for rewriting English text in a natural, human-like manner while fully preserving meaning.

## المميزات / Features

- ✅ إعادة صياغة طبيعية بأسلوب بشري / Natural human-like rewriting
- ✅ الحفاظ على المعنى / Meaning preservation
- ✅ مستويات مختلفة لإعادة الصياغة / Multiple rewriting levels (light/medium/strong)
- ✅ نبرات أسلوبية متنوعة / Various style tones (formal, academic, casual)
- ✅ تصحيح لغوي ونحوي / Grammar and fluency correction
- ✅ API وواجهة ويب / REST API and web interface
- ✅ تدريب باستخدام LoRA لتقليل الموارد / LoRA-based training for efficiency

## المعمارية / Architecture

```
Input → Normalization → Meaning-Preserving Rewrite (LLM) → 
Style Controller → Grammar & Fluency Pass → Quality Gates → Output
```

## المتطلبات / Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- 8GB+ RAM (16GB+ recommended)
- GPU with 6GB+ VRAM (recommended for training)

## التثبيت / Installation

### 1. استنساخ المشروع / Clone Repository

```bash
git clone <repository-url>
cd Paraphrase
```

### 2. إنشاء بيئة افتراضية / Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. تثبيت المكتبات / Install Dependencies

```bash
pip install -r requirements.txt
```

## الاستخدام / Usage

### الخطوة 1: تحميل البيانات / Step 1: Download Data

```bash
python src/data/downloader.py
```

سيقوم هذا بتحميل البيانات من:
- Machine Paraphrase Dataset
- Quora Question Pairs
- PAWS

This will download data from:
- Machine Paraphrase Dataset
- Quora Question Pairs
- PAWS

### الخطوة 2: تجهيز البيانات / Step 2: Preprocess Data

```bash
python src/data/preprocessor.py
```

أو مع ملف محدد:

```bash
python src/data/preprocessor.py data/raw/combined_raw.json
```

### الخطوة 3: تدريب النموذج / Step 3: Train Model

**الطريقة السريعة (موصى به) / Quick Method (Recommended):**

```bash
# من scripts directory
scripts\START_TRAINING.bat

# أو مباشرة
scripts\train_with_book1.bat
```

**الطريقة اليدوية / Manual Method:**

```bash
python src/training/trainer.py data/processed/mpc_cleaned_combined_train_with_book1.json data/processed/mpc_cleaned_combined_val.json
```

راجع [`docs/HOW_TO_RUN_TRAINING.md`](docs/HOW_TO_RUN_TRAINING.md) للتفاصيل الكاملة.

### الخطوة 4: تقييم النموذج / Step 4: Evaluate Model

```bash
python src/evaluation/evaluator.py models/final
```

### الخطوة 5: تشغيل API / Step 5: Run API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

ثم افتح المتصفح على: `http://localhost:8000`

Then open browser at: `http://localhost:8000`

## استخدام Docker / Using Docker

### بناء الصورة / Build Image

```bash
docker build -t paraphrase-api .
```

### تشغيل الحاوية / Run Container

```bash
docker run -p 8000:8000 -v $(pwd)/models:/app/models paraphrase-api
```

أو باستخدام docker-compose:

```bash
docker-compose up -d
```

## التكوين / Configuration

يمكن تعديل الإعدادات في ملف `config.yaml`:

You can modify settings in `config.yaml`:

- **Model**: اختيار النموذج الأساسي (T5-base, BART, etc.)
- **Training**: معاملات التدريب (learning rate, batch size, etc.)
- **Data**: مصادر البيانات وحدودها
- **API**: إعدادات الخادم

## API Documentation

### Endpoint: POST `/api/rewrite`

إعادة صياغة نص:

Rewrite text:

```json
{
  "text": "Your text here",
  "tone": "casual",  // Options: formal, academic, casual
  "strength": "medium",  // Options: light, medium, strong
  "max_length": 512  // Optional
}
```

Response:

```json
{
  "original": "Your text here",
  "rewritten": "Rewritten text",
  "tone": "casual",
  "strength": "medium",
  "processing_time": 1.23
}
```

### Endpoint: GET `/api/health`

فحص حالة النظام:

Check system health:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0"
}
```

## هيكل المشروع / Project Structure

```
Paraphrase/
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies (CPU)
├── requirements-cuda.txt    # Python dependencies (CUDA)
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose config
├── README.md               # This file
├── src/                    # Source code
│   ├── api/               # API server code
│   ├── training/          # Training code
│   ├── data/              # Data processing code
│   └── evaluation/        # Evaluation code
├── scripts/                # Executable scripts
│   ├── train_with_book1.bat    # Main training script
│   ├── START_TRAINING.bat      # Training wrapper
│   ├── start_api.bat           # API server
│   ├── reset_training.bat      # Reset training environment
│   └── utils/                  # Utility Python scripts
├── data/                    # Data files
│   ├── raw/               # Raw input data
│   ├── processed/         # Processed training data
│   └── cache/             # Tokenized data cache
├── models/                  # Model files
│   ├── checkpoints/       # Training checkpoints
│   └── final/             # Final trained model
├── tests/                   # Test files
└── docs/                    # Documentation
    ├── PROJECT_STRUCTURE.md
    ├── HOW_TO_RUN_TRAINING.md
    └── ...
```

للتفاصيل الكاملة: راجع [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)

## التقييم / Evaluation

النظام يستخدم المقاييس التالية:

The system uses the following metrics:

- **BERTScore**: Semantic similarity
- **ROUGE**: Overlap-based metrics
- **Fluency Metrics**: Grammar and readability
- **Diversity**: Distinct-n for variety

## الأداء المتوقع / Expected Performance

- **Processing Time**: 1-5 seconds for short texts
- **Quality**: High semantic preservation with natural fluency
- **Diversity**: Varied paraphrases without distortion

## الترخيص / License

هذا المشروع مفتوح المصدر.

This project is open source.

## المساهمة / Contributing

نرحب بالمساهمات! يرجى فتح issue أو pull request.

Contributions welcome! Please open an issue or pull request.

## الدعم / Support

للأسئلة والدعم، يرجى فتح issue في المستودع.

For questions and support, please open an issue in the repository.

## المراجع / References

- T5: Text-To-Text Transfer Transformer
- BART: Denoising Autoencoder for Pre-training
- LoRA: Low-Rank Adaptation for Efficient Fine-tuning
- Datasets: Hugging Face Datasets Library

