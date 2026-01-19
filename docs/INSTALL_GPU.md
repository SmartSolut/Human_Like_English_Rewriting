# دليل تثبيت PyTorch مع دعم CUDA / PyTorch CUDA Installation Guide

## المشكلة / Problem

عند تثبيت PyTorch من `requirements.txt` العادي، يتم تثبيت النسخة الخاصة بالـ CPU فقط بدون دعم CUDA.

When installing PyTorch from the regular `requirements.txt`, it installs the CPU-only version without CUDA support.

## الحل / Solution

### الخطوة 1: إزالة PyTorch القديم / Step 1: Remove Old PyTorch

```bash
pip uninstall torch torchvision torchaudio -y
```

### الخطوة 2: تثبيت PyTorch مع CUDA / Step 2: Install PyTorch with CUDA

اختر الإصدار المناسب حسب CUDA لديك:

#### للـ CUDA 12.1 (موصى به) / For CUDA 12.1 (Recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### للـ CUDA 11.8 / For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### الخطوة 3: تثبيت باقي المكتبات / Step 3: Install Other Libraries

```bash
pip install -r requirements-cuda.txt
```

أو تثبيت يدوي:
```bash
pip install transformers>=4.35.0 datasets>=2.14.0 accelerate>=0.24.0 peft>=0.6.0
pip install fastapi uvicorn[standard] pydantic
pip install scikit-learn numpy pandas tqdm sentencepiece
pip install bert-score sentence-transformers rouge-score
pip install python-dotenv pyyaml requests
```

### الخطوة 4: التحقق / Step 4: Verify

```bash
python check_gpu.py
```

يجب أن ترى:
- `CUDA Available: True`
- `PyTorch Version: 2.x.x+cu121` (أو cu118)

## ملاحظات مهمة / Important Notes

1. **لا تستخدم `pip install -r requirements.txt` مباشرة** - سيتم تثبيت PyTorch بدون CUDA
2. **تأكد من أن CUDA drivers مثبتة** - تحقق بـ `nvidia-smi`
3. **استخدم `requirements-cuda.txt`** بعد تثبيت PyTorch مع CUDA

## استكشاف الأخطاء / Troubleshooting

### الخطأ: "CUDA is not available"
- تأكد من تثبيت PyTorch مع CUDA (ليس CPU version)
- تحقق من `nvidia-smi` يعمل
- أعد تشغيل Python بعد التثبيت

### الخطأ: "No module named torch"
- تأكد من تثبيت PyTorch في نفس البيئة الافتراضية
- تحقق من `pip list | findstr torch`

### الخطأ: "CUDA version mismatch"
- تأكد من أن إصدار CUDA في PyTorch متوافق مع CUDA drivers
- استخدم `torch.version.cuda` للتحقق


