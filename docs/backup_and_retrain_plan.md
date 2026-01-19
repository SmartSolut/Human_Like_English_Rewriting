# خطة النسخ الاحتياطي وإعادة التدريب على 256 Tokens

## الوضع الحالي:
- النموذج الحالي مدرب على **128 tokens** (قصير)
- النموذج محفوظ في `models/final`
- البيانات تحتاج تنظيف أكثر

## الخطة المقترحة:

### الخطوة 1: نسخ احتياطي للنموذج الحالي (128 tokens)
```bash
# إنشاء مجلد للنسخ الاحتياطي
mkdir models\backup_128_tokens

# نسخ النموذج الحالي
xcopy models\final models\backup_128_tokens\ /E /I
```

### الخطوة 2: تنظيف جميع البيانات (Parts 1-5)
```bash
# تنظيف جميع الأجزاء
python clean_all_data_parts.py
```

### الخطوة 3: تعديل trainer.py لاستخدام 256 tokens
- تغيير `max_target_length` من 200 إلى 256
- تغيير `max_source_length` من 256 إلى 256 (أو 512)

### الخطوة 4: حذف النموذج القديم والـ cache
```bash
# حذف النموذج الحالي
rmdir /s /q models\final

# حذف الـ cache
rmdir /s /q data\cache
```

### الخطوة 5: إعادة التدريب من الصفر على 256 tokens
```bash
# استخدام البيانات النظيفة
# نسخ البيانات النظيفة إلى المجلد الأصلي
copy data\processed\splits_5_parts_cleaned\train_part_*_cleaned.json data\processed\splits_5_parts\

# إعادة تسمية الملفات
ren data\processed\splits_5_parts\train_part_1_cleaned.json train_part_1.json
ren data\processed\splits_5_parts\train_part_2_cleaned.json train_part_2.json
ren data\processed\splits_5_parts\train_part_3_cleaned.json train_part_3.json
ren data\processed\splits_5_parts\train_part_4_cleaned.json train_part_4.json
ren data\processed\splits_5_parts\train_part_5_cleaned.json train_part_5.json

# إعادة التدريب
scripts\train_part_1.bat
scripts\train_part_2.bat
scripts\train_part_3.bat
scripts\train_part_4.bat
scripts\train_part_5.bat
```

## المزايا:
1. ✅ النموذج القديم (128) محفوظ كـ backup
2. ✅ البيانات نظيفة أكثر
3. ✅ النموذج الجديد على 256 tokens (نصوص أطول)
4. ✅ تدريب من الصفر (لا مشاكل في الاستقرار)

## الوقت المتوقع:
- تنظيف البيانات: ~10-15 دقيقة
- Part 1: ~1.5-2 ساعة
- Part 2: ~1.5-2 ساعة
- Part 3: ~1.5-2 ساعة
- Part 4: ~1.5-2 ساعة
- Part 5: ~1.5-2 ساعة
- **المجموع: ~7.5-10 ساعات**

## التوصية:
**نعم، هذا هو الحل الأفضل!**

