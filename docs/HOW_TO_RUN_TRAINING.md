# كيفية تشغيل التدريب

## المشكلة: النافذة تغلق بسرعة

إذا كانت النافذة تغلق بسرعة، اتبع هذه الخطوات:

## الحل 1: تشغيل من Command Prompt (موصى به)

### الخطوات:
1. افتح Command Prompt (cmd)
   - اضغط `Win + R`
   - اكتب `cmd` واضغط Enter

2. انتقل إلى مجلد المشروع:
   ```cmd
   cd E:\project_student\Paraphrase
   ```

3. شغّل السكريبت:
   ```cmd
   train_with_book1.bat
   ```
   أو
   ```cmd
   scripts\train_with_book1.bat
   ```

## الحل 2: إضافة pause في نهاية السكريبت

تم إضافة `pause` في نهاية السكريبتات المحدثة، لكن إذا استمرت المشكلة:

1. اضغط `Shift + Right Click` على الملف `.bat`
2. اختر "Run as administrator"
3. أو افتح Command Prompt أولاً ثم شغّل السكريبت

## الحل 3: تشغيل مباشر من Python

إذا استمرت المشكلة، شغّل Python مباشرة:

```cmd
python src\training\trainer.py "data\processed\mpc_cleaned_combined_train_with_book1.json" "data\processed\mpc_cleaned_combined_val.json"
```

## ما تم إصلاحه في السكريبتات:

1. ✅ إضافة `setlocal enabledelayedexpansion` للمتغيرات
2. ✅ إضافة `pause` بعد كل خطأ
3. ✅ التحقق من وجود الملفات قبل التشغيل
4. ✅ رسائل خطأ واضحة
5. ✅ عرض المجلد الحالي عند الخطأ

## التحقق من المشاكل الشائعة:

### 1. GPU غير متاح:
```
ERROR: CUDA not available!
```
**الحل**: تأكد من تثبيت PyTorch مع CUDA

### 2. الملفات غير موجودة:
```
ERROR: Training file not found!
```
**الحل**: تأكد من وجود الملفات في المسار الصحيح

### 3. خطأ في Python:
**الحل**: افتح Command Prompt وافحص رسالة الخطأ

---

## الخطوات الموصى بها:

1. افتح Command Prompt أولاً
2. انتقل إلى مجلد المشروع
3. شغّل السكريبت
4. راقب الرسائل للأخطاء
5. إذا ظهر خطأ، اقرأ الرسالة قبل أن تختفي

---

## ملاحظة مهمة:

السكريبتات الآن تحتوي على:
- `pause` بعد كل خطأ
- رسائل خطأ واضحة
- التحقق من الملفات قبل التشغيل

لكن من الأفضل دائماً تشغيلها من Command Prompt لرؤية جميع الرسائل.
