# دليل التدريب المحسّن / Enhanced Training Guide

## التحسينات المضافة

### ✅ 1. التقييم أثناء التدريب
- تم تفعيل `eval_strategy="steps"` لتقييم النموذج أثناء التدريب
- يتم تقييم النموذج كل `eval_steps` خطوة
- يتم حفظ أفضل نموذج بناءً على `eval_loss`
- إضافة `compute_metrics` لحساب metrics إضافية

### ✅ 2. فحص جودة البيانات
- يتم فحص البيانات تلقائياً قبل التدريب
- يقارن بين `input` و `target` لضمان وجود إعادة صياغة كافية
- يحذر إذا كانت التغييرات قليلة جداً

### ✅ 3. تحسين Logging
- عرض معلومات مفصلة عن النموذج المحمّل
- عرض حجم النموذج ونوعه (LoRA vs Full)
- عرض مسار الملفات المستخدمة في التدريب

### ✅ 4. مسح الكاش
- إضافة وظيفة `clear_cache()` لمسح الكاش القديم
- سكريبت `clear_cache.bat` لمسح الكاش يدوياً

---

## خطوات التدريب

### الخطوة 1: التحقق من النموذج الحالي

#### أ) شغّل API وافحص اللوج:
```bash
python src/api/main.py
```

ابحث عن:
```
Model Loaded Successfully!
📁 Model Path: models/final
📦 Base Model: t5-base
🔧 Model Type: LoRA Adapter
📦 Model Size: X.XX MB
```

#### ب) تحقق من الملفات:
- `models/final/adapter_config.json` → LoRA adapter
- `models/final/adapter_model.safetensors` → LoRA weights
- الحجم المتوقع: **5-15 MB** (إذا كان أكبر، قد يكون هناك مشكلة)

---

### الخطوة 2: مسح الكاش القديم

#### أ) استخدام السكريبت:
```bash
scripts\clear_cache.bat
```

#### ب) أو يدوياً:
```bash
# احذف المجلدات التالية:
data\cache\train_tokenized\
data\cache\val_tokenized\
data\cache\cache_marker.txt
```

**⚠️ مهم:** هذا يمسح فقط الكاش، لا يمسح البيانات الأصلية!

---

### الخطوة 3: التحقق من ملفات التدريب

#### أ) تحقق من الملف المستخدم:
افتح `scripts/train_part_1.bat` وتحقق من:
```batch
set TRAIN_FILE=%SPLITS_DIR%\train_part_1.json
```

#### ب) إذا كنت تريد النسخة المنظفة:
```batch
set TRAIN_FILE=%SPLITS_DIR%\train_part_1_cleaned.json
```

#### ج) فحص عينة من البيانات:
```python
import json

# افتح ملف التدريب
with open('data/processed/splits_5_parts/train_part_1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# افحص أول 5 عينات
for i, sample in enumerate(data[:5]):
    input_text = sample['input'].replace('humanize: ', '')
    target_text = sample['target']
    
    print(f"\n=== Sample {i+1} ===")
    print(f"Input: {input_text[:100]}...")
    print(f"Target: {target_text[:100]}...")
    print(f"Input words: {len(input_text.split())}")
    print(f"Target words: {len(target_text.split())}")
```

**✅ يجب أن يكون الفرق واضحاً بين input و target!**

---

### الخطوة 4: التدريب على الأجزاء

#### أ) تدريب الجزء الأول:
```bash
scripts\train_part_1.bat
```

#### ب) مراقبة اللوج:
ابحث عن:
```
Checking Data Quality
Total samples in file: XXXX
Average word changes per sample: XX.X
✅ Data quality looks good - sufficient variation

Training Configuration
📁 Train file: data/processed/splits_5_parts/train_part_1.json
📁 Validation file: data/processed/mpc_cleaned_combined_val.json
📊 Train samples: X,XXX
📊 Validation samples: X,XXX

✅ Evaluation enabled: eval_strategy='steps', eval_steps=XXX
✅ Best model selection: metric_for_best_model='eval_loss'
```

#### ج) مراقبة Loss:
```
Step 100: train_loss=2.XXX, eval_loss=2.XXX
Step 200: train_loss=1.XXX, eval_loss=1.XXX
...
```

**⚠️ إذا كان `eval_loss` لا يتناقص، قد يكون هناك overfitting!**

---

### الخطوة 5: التحقق من النموذج بعد التدريب

#### أ) فحص الحجم:
```
Saving Final Model
📁 Saving to: models/final
✅ LoRA adapter saved successfully
📦 Model size: X.XX MB
```

**✅ الحجم المتوقع: 5-15 MB**

#### ب) إعادة تحميل النموذج:
```bash
# إعادة تشغيل API
python src/api/main.py

# أو استخدام endpoint
curl -X POST http://localhost:8000/api/reload-model
```

---

## نصائح مهمة

### 1. إذا كان النموذج ينتج تغييرات بسيطة:

#### أ) تحقق من البيانات:
- تأكد أن الفرق بين `input` و `target` كبير
- إذا كان الفرق < 10%، النموذج سيتعلم النسخ

#### ب) تحقق من Loss:
- إذا كان `eval_loss` لا يتناقص، قد تحتاج:
  - تقليل `learning_rate`
  - زيادة `num_epochs`
  - تغيير `lora_r` أو `lora_alpha`

#### ج) تحقق من معاملات التوليد:
- في `src/api/main.py`، راجع:
  - `temperature` (حاول رفعه)
  - `do_sample` (فعّله لـ medium)
  - `repetition_penalty` (حاول رفعه)

### 2. إذا كان الحجم كبير (>> 15 MB):

#### أ) تحقق من طريقة الحفظ:
```python
# يجب أن يكون:
model.save_pretrained(path)  # يحفظ adapter فقط

# وليس:
model.merge_and_unload().save_pretrained(path)  # يحفظ النموذج الكامل
```

#### ب) تحقق من `adapter_config.json`:
```json
{
  "peft_type": "LORA",
  "r": 16,
  "lora_alpha": 32,
  ...
}
```

### 3. إذا كان التدريب بطيء:

#### أ) قلل `eval_steps`:
```python
eval_steps = 1000  # بدلاً من 500
```

#### ب) قلل `save_steps`:
```python
save_steps = 2000  # بدلاً من 1000
```

#### ج) استخدم `eval_strategy="no"` مؤقتاً:
```python
eval_strategy="no"  # لتدريب أسرع (لكن بدون تقييم)
```

---

## الأوامر السريعة

### مسح الكاش:
```bash
scripts\clear_cache.bat
```

### تدريب الجزء الأول:
```bash
scripts\train_part_1.bat
```

### فحص النموذج:
```bash
python src/api/main.py
# ثم افتح http://localhost:8000
```

### إعادة تحميل النموذج:
```bash
curl -X POST http://localhost:8000/api/reload-model
```

---

## استكشاف الأخطاء

### المشكلة: النموذج ينتج نفس المدخل
**الحل:**
1. تحقق من جودة البيانات (الفرق بين input/target)
2. مسح الكاش وإعادة التدريب
3. رفع `temperature` و `repetition_penalty`

### المشكلة: الحجم كبير (>> 15 MB)
**الحل:**
1. تحقق من `adapter_config.json`
2. تأكد أن `save_pretrained()` يحفظ adapter فقط
3. لا تستخدم `merge_and_unload()`

### المشكلة: Loss لا يتناقص
**الحل:**
1. تحقق من البيانات (قد تكون متشابهة جداً)
2. قلل `learning_rate`
3. زد `num_epochs`
4. غيّر `lora_r` أو `lora_alpha`

---

## الخلاصة

✅ **تم إضافة:**
- التقييم أثناء التدريب
- فحص جودة البيانات
- تحسين logging
- وظيفة مسح الكاش

✅ **الخطوات المطلوبة:**
1. تحقق من النموذج الحالي
2. مسح الكاش
3. تحقق من ملفات التدريب
4. تدريب على الأجزاء
5. التحقق من النموذج بعد التدريب

✅ **النتيجة المتوقعة:**
- نموذج بحجم 5-15 MB (LoRA adapter)
- إعادة صياغة أفضل (تغييرات واضحة)
- تقييم أثناء التدريب (eval_loss)

---

**تاريخ التحديث:** 2025-01-XX
**الإصدار:** 2.0
