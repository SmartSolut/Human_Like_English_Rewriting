# خريطة السكربتات والملفات / Scripts and Files Mapping

## 📋 ملخص السكربتات والملفات المستخدمة

### السكربتات الرئيسية:

| السكربت | الملف المستخدم | العينات | ملاحظات |
|---------|---------------|---------|----------|
| `TRAIN_NOW.bat` | → `train_with_book1.bat` | `mpc_cleaned_combined_train_with_book1.json` | ~108,611 | MPC + Book1 |
| `START_TRAINING.bat` | → `train_with_book1.bat` | `mpc_cleaned_combined_train_with_book1.json` | ~108,611 | MPC + Book1 |
| `train_with_book1.bat` | `mpc_cleaned_combined_train_with_book1.json` | ~108,611 | MPC + Book1 |
| `train_full_data.bat` | `mpc_cleaned_combined_train.json` | ~108,606 | MPC فقط (بدون Book1) |
| `train_book1.bat` | `book1_train.json` | ? | Book1 فقط |
| `train_part_1.bat` | `splits_5_parts_cleaned/train_part_1_cleaned.json` | ~21,722 | **✅ cleaned (مع fallback)** |
| `train_part_2.bat` | `splits_5_parts_cleaned/train_part_2_cleaned.json` | ~21,721 | **✅ cleaned (مع fallback)** |
| `train_part_3.bat` | `splits_5_parts_cleaned/train_part_3_cleaned.json` | ~21,721 | **✅ cleaned (مع fallback)** |
| `train_part_4.bat` | `splits_5_parts_cleaned/train_part_4_cleaned.json` | ~21,721 | **✅ cleaned (مع fallback)** |
| `train_part_5.bat` | `splits_5_parts_cleaned/train_part_5_cleaned.json` | ~21,721 | **✅ cleaned (مع fallback)** |

### الملفات المتوفرة:

#### في `data/processed/`:
- ✅ `mpc_cleaned_combined_train.json` (MPC فقط)
- ✅ `mpc_cleaned_combined_train_with_book1.json` (MPC + Book1)
- ✅ `mpc_cleaned_combined_val.json` (Validation)

#### في `data/processed/splits_5_parts/`:
- ✅ `train_part_1.json` (غير منظف)
- ✅ `train_part_2.json` (غير منظف)
- ✅ `train_part_3.json` (غير منظف)
- ✅ `train_part_4.json` (غير منظف)
- ✅ `train_part_5.json` (غير منظف)

#### في `data/processed/splits_5_parts_cleaned/`:
- ✅ `train_part_1_cleaned.json` (منظف) ⭐
- ✅ `train_part_2_cleaned.json` (منظف) ⭐
- ✅ `train_part_3_cleaned.json` (منظف) ⭐
- ✅ `train_part_4_cleaned.json` (منظف) ⭐
- ✅ `train_part_5_cleaned.json` (منظف) ⭐

---

## ✅ الحالة الحالية:

### 1. سكربتات الأجزاء محدثة ✅
- السكربتات تستخدم `train_part_X_cleaned.json` (منظف) أولاً
- إذا لم توجد النسخة المنظفة، تستخدم `train_part_X.json` كـ fallback
- النسخ المنظفة موجودة في `splits_5_parts_cleaned/` ومستخدمة الآن

### 2. عدم توحيد الملفات
- بعض السكربتات تستخدم `mpc_cleaned_combined_train_with_book1.json`
- بعضها يستخدم `mpc_cleaned_combined_train.json`
- بعضها يستخدم `train_part_X.json`

---

## ✅ الحلول المطبقة:

### ✅ تم تحديث سكربتات الأجزاء لاستخدام النسخ المنظفة
```batch
set SPLITS_DIR=data/processed/splits_5_parts_cleaned
set TRAIN_FILE=%SPLITS_DIR%\train_part_1_cleaned.json
# مع fallback تلقائي إذا لم توجد النسخة المنظفة
```

### ✅ تم إصلاح prepare_clean_data.bat
- المسار الصحيح: `scripts\utils\clean_all_data_parts.py`

---

## 📝 التوصيات:

1. **للتجربة السريعة:** استخدم `train_part_X_cleaned.json`
2. **للتجربة الكاملة:** استخدم `mpc_cleaned_combined_train_with_book1.json`
3. **للتجربة على أجزاء:** استخدم `train_part_X_cleaned.json` (وليس `train_part_X.json`)

---

**تاريخ التحديث:** 2025-01-XX
