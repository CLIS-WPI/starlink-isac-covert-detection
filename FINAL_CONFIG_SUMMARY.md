# ✅ تنظیمات نهایی - Semi-Fixed Pattern با پارامترهای بهینه

## 📊 تغییرات اعمال شده در `config/settings.py`

### 🎯 پارامترهای اصلی

```python
# =======================================================
# 💡 Covert Injection Settings (Semi-Fixed Pattern)
# =======================================================

COVERT_AMP = 1.7                      # ↑ از 1.5 به 1.7 (هدف: 3-4% power diff)
USE_SEMI_FIXED_PATTERN = True         # ✅ فعال
NUM_COVERT_SUBCARRIERS = 16           # ✅ 16 subcarrier
BAND_SIZE = 8                         # ✅ باندهای پیوسته 8تایی
BAND_START_OPTIONS = [0, 16, 32, 48] # ↑ تغییر: فاصله بیشتر بین موقعیت‌ها
SYMBOL_PATTERN_OPTIONS = [
    [1, 3, 5, 7],    # الگوی فرد
    [2, 4, 6, 8]     # الگوی زوج
]
ADD_NOISE = True                      # ✅ فعال
NOISE_STD = 0.015                     # ✅ نویز ملایم
```

---

## 📈 مقایسه با نسخه قبلی

| پارامتر | قبل | حالا | دلیل تغییر |
|---------|-----|------|-----------|
| **COVERT_AMP** | 1.5 | **1.7** | افزایش power diff به 3-4% |
| **BAND_START_OPTIONS** | [0, 8, 16, 24] | **[0, 16, 32, 48]** | پخش بهتر در طیف |
| **ADD_NOISE** | True | True | بدون تغییر ✓ |
| **NOISE_STD** | 0.015 | 0.015 | بدون تغییر ✓ |

---

## 🎯 انتظار نتایج

### Power Difference
```
Target: 3-4%
با COVERT_AMP=1.7 انتظار می‌ره:
  - Benign: ~0.25
  - Attack: ~0.26
  - Diff: ~3.5%
```

### Pattern Coverage
```
4 band positions × 2 symbol patterns = 8 unique patterns
Coverage: 48 subcarriers (از 64) = 75% طیف
```

### Expected Performance
```
✅ AUC: 0.85-0.92
✅ Precision: 0.82-0.90
✅ Recall: 0.80-0.88
✅ F1 Score: 0.81-0.89
```

---

## 🔧 Aliases اضافه شده

برای سازگاری با مستندات مختلف:

```python
SYMBOL_PATTERNS = SYMBOL_PATTERN_OPTIONS  # Alias
SUBBAND_SIZE = BAND_SIZE                  # Alias (8)
```

---

## 📊 ویژگی‌های کلیدی این تنظیمات

### 1️⃣ Semi-Fixed Pattern
- ✅ 8 pattern منحصر به فرد
- ✅ قابل یادگیری برای CNN
- ✅ تنوع کافی برای generalization

### 2️⃣ Spectral Distribution
```
Band positions: [0, 16, 32, 48]
  - Band 1: subcarriers 0-7, 8-15
  - Band 2: subcarriers 16-23, 24-31
  - Band 3: subcarriers 32-39, 40-47
  - Band 4: subcarriers 48-55, 56-63
```
**مزیت:** پوشش یکنواخت‌تر طیف

### 3️⃣ Symbol Patterns
```
Pattern A: [1, 3, 5, 7] → فرد
Pattern B: [2, 4, 6, 8] → زوج
```
**مزیت:** تفکیک واضح در temporal domain

### 4️⃣ Noise Injection
```
ADD_NOISE = True
NOISE_STD = 0.015 → SNR ~ 40 dB
```
**مزیت:** robustness بدون از دست دادن signature

---

## 🚀 آماده برای اجرا

### چک کردن تنظیمات:
```bash
python3 test_semi_fixed_pattern.py
```

انتظار output:
```
🎯 SEMI-FIXED PATTERN CONFIGURATION
========================================
  COVERT_AMP:             1.7
  NUM_COVERT_SUBCARRIERS: 16
  BAND_SIZE:              8
  BAND_START_OPTIONS:     [0, 16, 32, 48]
  
  Total unique patterns:  8
  Expected power diff:    ~3-4%
  Expected AUC:           0.85-0.92
```

### تست سریع (10 دقیقه):
```bash
chmod +x quick_test_cnn.sh
./quick_test_cnn.sh
```

### تست کامل (25 دقیقه):
```bash
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```

---

## 🎯 مراحل بعدی (بعد از موفقیت)

### اگر AUC ≥ 0.85 شد:

#### مرحله 1: کاهش تدریجی amplitude
```python
COVERT_AMP = 1.5  # کاهش از 1.7
# Target: 2-3% power diff
```

#### مرحله 2: افزایش diversity
```python
BAND_START_OPTIONS = [0, 8, 16, 24, 32, 40, 48, 56]  # 8 موقعیت
# Total patterns: 8 × 2 = 16
```

#### مرحله 3: افزایش نویز
```python
NOISE_STD = 0.02  # افزایش از 0.015
# SNR ~ 35 dB
```

---

## 📝 نکات مهم

### ✅ DO's:
1. همیشه قبل از training بررسی کن:
   ```bash
   python3 check_balance.py
   python3 analyze_power.py
   ```

2. SEED=42 رو تغییر نده

3. نتایج رو document کن

4. تغییرات تدریجی انجام بده

### ❌ DON'Ts:
1. COVERT_AMP رو یکباره خیلی تغییر نده (max ±0.2)

2. هم‌زمان چند پارامتر رو تغییر نده

3. بدون check کردن dataset جدید نساز

4. نتایج قبلی رو حذف نکن

---

## 🔍 Verification Checklist

قبل از شروع training:

```bash
# 1. تنظیمات
grep "COVERT_AMP" config/settings.py
# Expected: COVERT_AMP = 1.7 ✅

# 2. Pattern config
grep "BAND_START_OPTIONS" config/settings.py
# Expected: [0, 16, 32, 48] ✅

# 3. Noise
grep "NOISE_STD" config/settings.py
# Expected: 0.015 ✅

# 4. Semi-fixed enabled
grep "USE_SEMI_FIXED_PATTERN" config/settings.py
# Expected: True ✅
```

---

## 📊 Expected Timeline

```
⏱️ Quick Test (500 samples):
  - Dataset generation: ~8 min
  - Training (20 epochs): ~2 min
  - Total: ~10 min

⏱️ Full Test (1500 samples):
  - Dataset generation: ~15 min
  - Training (50 epochs): ~8 min
  - Total: ~23 min
```

---

## ✅ فایل‌های به‌روز شده

1. ✅ `config/settings.py` - پارامترهای نهایی
   - COVERT_AMP = 1.7
   - BAND_START_OPTIONS = [0, 16, 32, 48]
   - Aliases اضافه شده

2. ✅ `core/covert_injection.py` - inject_covert_semi_fixed()

3. ✅ `core/dataset_generator.py` - استفاده از semi-fixed

4. ✅ `model/detector_cnn.py` - class_weight support

5. ✅ ابزارهای تست:
   - test_semi_fixed_pattern.py
   - analyze_power.py
   - check_balance.py

---

## 🎊 آماده برای تست!

همه چیز آماده‌ست با تنظیمات بهینه:
- ✅ COVERT_AMP = 1.7 (3-4% power diff)
- ✅ باندهای پخش شده در طیف
- ✅ 8 pattern قابل یادگیری
- ✅ نویز ملایم برای robustness
- ✅ Class balance support
- ✅ Reproducible (SEED=42)

**می‌تونیم شروع کنیم!** 🚀
