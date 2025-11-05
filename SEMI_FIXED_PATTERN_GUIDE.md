# 🎯 Semi-Fixed Pattern Strategy - Implementation Summary

## ❌ مشکل قبلی

```
Strategy: Fully random injection
COVERT_AMP = 1.2-1.4
Power diff = 0.84-1.5%
AUC = 0.50-0.55 (random guessing)
```

**علت شکست:**
- الگوهای کاملاً تصادفی → CNN نمی‌تونه pattern مشترک پیدا کنه
- Subcarrier های پراکنده → سیگنال طیفی ضعیف
- Power diff خیلی کم (< 2%) → قابل تشخیص نیست

---

## ✅ راه‌حل جدید: Semi-Fixed Pattern

### استراتژی اصلی

#### 1️⃣ باندهای پیوسته (Contiguous Bands)
```python
BAND_SIZE = 8                    # 8 subcarrier پشت سر هم
NUM_COVERT_SUBCARRIERS = 16      # مجموعاً 16 subcarrier (2 باند)
BAND_START_OPTIONS = [0, 8, 16, 24]  # 4 موقعیت شروع محدود
```

**مثال:**
- Sample 1: subcarriers [0-7, 8-15]
- Sample 2: subcarriers [8-15, 16-23]
- Sample 3: subcarriers [16-23, 24-31]
- Sample 4: subcarriers [0-7, 8-15] (تکرار الگو)

#### 2️⃣ الگوهای Symbol نیمه‌ثابت
```python
SYMBOL_PATTERN_OPTIONS = [
    [1, 3, 5, 7],    # Symbol های فرد
    [2, 4, 6, 8]     # Symbol های زوج
]
```

**ترکیبات:**
- 4 موقعیت band × 2 الگوی symbol = **8 pattern منحصر به فرد**

#### 3️⃣ افزایش Amplitude
```python
COVERT_AMP = 1.5  # ↑ از 1.4
```
**انتظار:** Power diff ~4-6%

#### 4️⃣ نویز ملایم
```python
ADD_NOISE = True
NOISE_STD = 0.015  # ↑ از 0.01
```

---

## 📊 مقایسه رویکردها

| پارامتر | Random | Semi-Fixed |
|---------|--------|------------|
| **Subcarrier placement** | پراکنده در 64 subcarrier | 2 باند پیوسته 8تایی |
| **Symbol pattern** | تصادفی از 10 symbol | 2 الگوی ثابت (فرد/زوج) |
| **Total patterns** | ~1000+ (خیلی زیاد) | 8 (قابل یادگیری) |
| **Spectral signature** | ضعیف و پراکنده | قوی و پیوسته |
| **Power diff** | 0.5-2% | 4-6% |
| **Expected AUC** | 0.50-0.60 | 0.80-0.90 |
| **CNN learning** | ❌ نمی‌تونه یاد بگیره | ✅ pattern واضح |

---

## 🎯 چرا این کار می‌کنه؟

### 1. Pattern Recognition
- CNN می‌بینه: "همیشه یک باند پیوسته 8تایی وجود داره"
- می‌تونه این ویژگی رو یاد بگیره

### 2. Spectral Signature
- باندهای پیوسته → انرژی متمرکز در یک ناحیه
- قابل تشخیص‌تر از subcarrier های پراکنده

### 3. Controlled Diversity
- 8 pattern کافی برای جلوگیری از overfitting
- کم کافی برای یادگیری pattern مشترک

### 4. Higher Power Difference
- COVERT_AMP = 1.5 → ~4-6% power diff
- قابل تشخیص ولی هنوز covert (< 10%)

---

## 🚀 نحوه استفاده

### 1️⃣ تست تنظیمات
```bash
python3 test_semi_fixed_pattern.py
```
این کار می‌کنه:
- ✅ نمایش تنظیمات فعلی
- ✅ نمایش 5 نمونه pattern
- ✅ مقایسه با رویکرد قبلی

### 2️⃣ تست سریع (500 samples)
```bash
chmod +x quick_test_cnn.sh
./quick_test_cnn.sh
```
**زمان:** ~8-10 دقیقه

### 3️⃣ تست کامل (1500 samples)
```bash
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```
**زمان:** ~20-25 دقیقه

### 4️⃣ فقط تحلیل power
```bash
python3 generate_dataset_parallel.py
python3 analyze_power.py
```

---

## 📈 انتظارات

### Power Difference
```
Target: 4-6%
Min for learning: 3%
Max for covert: 8%
```

### AUC Performance
```
✅ Excellent: AUC ≥ 0.85
✅ Good:      AUC ≥ 0.75
⚠️  Moderate:  AUC ≥ 0.65
❌ Poor:      AUC < 0.65
```

### Training Time
```
Quick test (500):  8-10 min
Full test (1500): 20-25 min
```

---

## 🔍 Debug Checklist

اگر AUC پایین شد، بررسی کن:

### 1. Power Analysis
```bash
python3 analyze_power.py
```
✅ باید: power diff ≥ 3%

### 2. Pattern Config
```bash
python3 test_semi_fixed_pattern.py
```
✅ باید: USE_SEMI_FIXED_PATTERN = True

### 3. Dataset Logs
در لاگ dataset generation:
```
[Covert-SemiFix] Sample #1: band_start=0, pattern=0, amp=1.5
symbols=[1, 3, 5, 7], subcarriers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]
```

### 4. CNN Training
در epochs اول:
- Validation AUC باید شروع به بالا رفتن کنه (epoch 5-10)
- اگر تا epoch 20 همچنان 0.5 موند → مشکل داره

---

## ⚙️ تنظیمات فعلی

```python
# Power
COVERT_AMP = 1.5

# Semi-fixed pattern
USE_SEMI_FIXED_PATTERN = True
NUM_COVERT_SUBCARRIERS = 16
BAND_SIZE = 8
BAND_START_OPTIONS = [0, 8, 16, 24]
SYMBOL_PATTERN_OPTIONS = [[1,3,5,7], [2,4,6,8]]

# Noise
ADD_NOISE = True
NOISE_STD = 0.015

# Dataset
NUM_SAMPLES_PER_CLASS = 1500
```

---

## 🎊 مراحل بعدی (بعد از موفقیت)

وقتی AUC ≥ 0.85 شد:

### 1️⃣ کاهش تدریجی amplitude
```python
COVERT_AMP = 1.3  # کاهش از 1.5
```

### 2️⃣ افزایش diversity
```python
BAND_START_OPTIONS = [0, 4, 8, 12, 16, 20, 24, 28]  # 8 موقعیت
```

### 3️⃣ افزایش نویز
```python
NOISE_STD = 0.02  # افزایش از 0.015
```

### 4️⃣ تست با شرایط واقعی‌تر
- Doppler shift
- Multipath fading
- Time-varying channels

---

## 📚 فایل‌های مرتبط

- `config/settings.py` - تنظیمات اصلی
- `core/covert_injection.py` - الگوریتم semi-fixed
- `core/dataset_generator.py` - استفاده از semi-fixed
- `test_semi_fixed_pattern.py` - تست و نمایش pattern ها
- `analyze_power.py` - تحلیل power difference
- `quick_test_cnn.sh` - تست سریع
- `run_full_pipeline.sh` - پایپلاین کامل

---

## 💡 نکات کلیدی

1. **Pattern باید قابل یادگیری باشه**: 8 pattern کافیه
2. **Contiguous bands قوی‌تر از scattered**: باندهای پیوسته بهتر قابل تشخیصن
3. **Power diff مهمه**: حداقل 3% نیازه
4. **Noise کمکه نه مانع**: نویز ملایم برای robustness خوبه
5. **صبور باش**: ممکنه 2-3 iteration نیاز باشه

---

## ✅ Success Criteria

```
✅ Power diff: 4-6%
✅ AUC: ≥ 0.85
✅ Precision: ≥ 0.80
✅ Recall: ≥ 0.75
✅ Training converges in < 30 epochs
```

**وقتی این شرایط برقرار شد → موفق بودیم!** 🎉
