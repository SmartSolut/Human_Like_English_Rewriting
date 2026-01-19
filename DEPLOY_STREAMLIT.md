# دليل رفع المشروع على Streamlit / Streamlit Deployment Guide

## الخطوات / Steps

### 1. رفع المشروع على GitHub

```bash
# تهيئة Git (إذا لم يكن موجوداً)
git init

# إضافة جميع الملفات
git add .

# عمل commit
git commit -m "Initial commit: Human-Like English Rewriting System"

# إضافة remote repository
git remote add origin https://github.com/SmartSolut/Human_Like_English_Rewriting.git

# رفع المشروع
git branch -M main
git push -u origin main
```

### 2. رفع على Streamlit Cloud

1. اذهب إلى [Streamlit Cloud](https://streamlit.io/cloud)
2. سجل الدخول بحساب GitHub
3. اضغط على "New app"
4. اختر المستودع: `SmartSolut/Human_Like_English_Rewriting`
5. اختر Branch: `main`
6. Main file path: `streamlit_app.py`
7. اضغط "Deploy"

### 3. إعدادات Streamlit Cloud

- **Python version**: 3.10+
- **Requirements file**: `requirements-streamlit.txt`
- **Main file**: `streamlit_app.py`

### 4. ملاحظات مهمة / Important Notes

⚠️ **تحذير**: النماذج كبيرة الحجم (models/) لن تُرفع على GitHub
- استخدم Git LFS للملفات الكبيرة
- أو ارفع النماذج على Google Drive / HuggingFace Hub
- أو استخدم Streamlit Secrets لتخزين مسارات النماذج

⚠️ **Warning**: Large model files won't be uploaded to GitHub
- Use Git LFS for large files
- Or upload models to Google Drive / HuggingFace Hub
- Or use Streamlit Secrets for model paths

### 5. استخدام Git LFS للملفات الكبيرة

```bash
# تثبيت Git LFS
git lfs install

# تتبع ملفات النماذج
git lfs track "models/**/*.pt"
git lfs track "models/**/*.bin"
git lfs track "models/**/*.safetensors"

# إضافة .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

### 6. بديل: استخدام HuggingFace Hub

```python
# في streamlit_app.py
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="SmartSolut/Human_Like_English_Rewriting",
    filename="adapter_model.safetensors"
)
```

## استكشاف الأخطاء / Troubleshooting

### المشكلة: النموذج لا يُحمّل
**الحل**: تأكد من أن مسار النموذج صحيح في `streamlit_app.py`

### المشكلة: خطأ في الذاكرة
**الحل**: استخدم `torch.float16` أو قلل `max_length`

### المشكلة: بطء في التحميل
**الحل**: استخدم `@st.cache_resource` للتحميل مرة واحدة
