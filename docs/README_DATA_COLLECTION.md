# دليل جمع البيانات / Data Collection Guide

## قالب CSV / CSV Template

استخدم ملف `template_data_collection.csv` لجمع البيانات.

### الأعمدة المطلوبة:

1. **input**: النص المدخل (AI text) مع بادئة `humanize: `
   - مثال: `humanize: The system processes data efficiently.`

2. **target**: النص المستهدف (Human text) - بدون بادئة
   - مثال: `The system handles data well.`

3. **source**: مصدر البيانات
   - قيم مقترحة: `book1`, `mpc-wikipedia`, `custom`, إلخ

4. **tone**: نبرة النص
   - القيم المسموحة: `formal`, `casual`, `academic`

5. **strength**: قوة إعادة الصياغة
   - القيم المسموحة: `light`, `medium`, `strong`

---

## أمثلة / Examples

### مثال 1: Formal Tone, Light Strength
```csv
input,target,source,tone,strength
humanize: The application utilizes advanced algorithms.,The app uses sophisticated methods.,book1,formal,light
```

### مثال 2: Casual Tone, Medium Strength
```csv
input,target,source,tone,strength
humanize: The system processes information through multiple layers.,The system works through several steps.,book1,casual,medium
```

### مثال 3: Academic Tone, Strong Strength
```csv
input,target,source,tone,strength
humanize: Utilization of computational methodologies facilitates outcomes.,Using better methods helps improve results.,book1,academic,strong
```

---

## ملاحظات مهمة / Important Notes

### ✅ صح / Correct:
- Input = نص AI (مع `humanize: `)
- Target = نص بشري
- الأزواج متطابقة (نفس النص الأصلي)

### ❌ خطأ / Wrong:
- Input = نص بشري (خطأ!)
- Target = نص AI (خطأ!)
- الأزواج غير متطابقة

---

## كيفية الاستخدام / How to Use

1. افتح `template_data_collection.csv` في Excel أو Google Sheets
2. املأ الأعمدة بنصوصك
3. تأكد من:
   - Input يبدأ بـ `humanize: `
   - Input = نص AI
   - Target = نص بشري
   - الأزواج متطابقة
4. احفظ الملف
5. استخدم سكريبت التحويل لتحويل CSV إلى JSON

---

## التحويل إلى JSON / Convert to JSON

بعد جمع البيانات في CSV، استخدم:
```bash
python convert_csv_to_json.py template_data_collection.csv output.json
```

---

## القيم المسموحة / Allowed Values

### Tone:
- `formal` - رسمي
- `casual` - عادي/غير رسمي
- `academic` - أكاديمي

### Strength:
- `light` - إعادة صياغة خفيفة
- `medium` - إعادة صياغة متوسطة
- `strong` - إعادة صياغة قوية

---

## مثال كامل / Complete Example

```csv
input,target,source,tone,strength
humanize: The system processes data through multiple algorithmic layers to generate optimal outcomes.,The system works through several steps to produce the best results.,book1,formal,medium
humanize: This application utilizes advanced computational methodologies to enhance user experience.,This app uses smart methods to make things better for users.,book1,casual,light
humanize: Implementation of sophisticated techniques facilitates improved performance metrics.,Using better approaches helps boost performance.,book1,academic,strong
```
