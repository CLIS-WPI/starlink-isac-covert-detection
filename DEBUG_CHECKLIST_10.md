# ✅ چک‌لیست ۱۰‌گانه DEBUG - اعمال شده

## تاریخ: November 4, 2025

---

## 🎯 وضعیت: همه 10 مورد اعمال شد!

---

### ✅ مورد 1: Dataset Load Check
**فایل**: `main_detection.py`

```python
# اضافه شد:
- File modification timestamp
- DEBUG: dataset path و تعداد نمونه‌ها
```

**چک کنید**:
```
📅 File modified: 2025-11-04 15:30:45
🔢 DEBUG dataset path = dataset/... → n = 200
```

---

### ✅ مورد 2: Injection Alignment Check
**فایل‌ها**: `main_detection.py`, `covert_injection.py`

```python
# در main_detection.py:
- DEBUG injected_symbols = [1,2,3,4,5,6,7]
- DEBUG mask.nonzero count
- DEBUG mask shape
- DEBUG mask symbols (unique)
- DEBUG mask subcarriers range

# در covert_injection.py:
- DEBUG injection: symbols, subcarriers, step
```

**چک کنید**:
```
🔍 DEBUG injected_symbols = [1,2,3,4,5,6,7]
🔍 DEBUG mask symbols (unique) = [1,2,3,4,5,6,7]  ← باید match کنند
🔍 DEBUG mask.nonzero count = XXX
```

---

### ✅ مورد 3: Spectral Difference Check
**فایل**: `debug_spectral_diff.py` (جدید)

```bash
python3 debug_spectral_diff.py
```

**خروجی مورد انتظار**:
```
🔍 مورد 3: تفاوت طیفی
  Δmag stats:
    Mean:   0.XXX  ← باید > 0.01
    Std:    0.XXX
    Max:    0.XXX
  
  Power analysis:
    Power diff: 4-5%  ← target
```

---

### ✅ مورد 4: Noise Disable (Temporary)
**فایل**: `config/settings.py`

```python
# تغییر داده شد:
ADD_NOISE = False  # ⚠️ موقتاً غیرفعال برای debug
```

**هدف**: بررسی اینکه آیا noise مشکل ایجاد می‌کند

---

### ✅ مورد 5: Shape & Axes Check
**فایل**: `main_detection.py`

```python
# اضافه شد:
🔍 DEBUG tx_grids shape = (N, ...)
  First sample (squeezed) shape: (10, 64)
  Expected: (n_symbols, n_subcarriers)
```

**چک کنید**: shape باید `(10, 64)` یا مشابه باشد

---

### ✅ مورد 6: Normalization Check
**فایل**: `model/detector_frequency.py`

```python
# اضافه شد:
🔍 DEBUG scaler fitted:
    mean = X.XXXXXX
    std  = X.XXXXXX
```

**مهم**: این فقط روی train data fit می‌شود، نه test

---

### ✅ مورد 7: Train/Test Split
**وضعیت**: قبلاً OK بود

```python
# در main_detection.py:
train_test_split(..., stratify=Y, random_state=SEED)
```

تضمین می‌کند classes balanced باشند.

---

### ✅ مورد 8: Red-Line Test
**فایل**: `redline_test.py` (جدید)

```bash
python3 redline_test.py
```

**تنظیمات**:
- `NUM_SAMPLES = 25` (کل 50)
- `COVERT_AMP = 0.80` (خیلی بالا)
- `ADD_NOISE = False`

**انتظار**: AUC > 0.95

---

### ✅ مورد 9: Energy Ratio Check
**فایل**: `debug_spectral_diff.py` (بخش دوم)

```bash
python3 debug_spectral_diff.py
```

**خروجی مورد انتظار**:
```
🔍 مورد 9: Energy Ratio
  Energy inside mask:   X.XXXXXX
  Energy outside mask:  X.XXXXXX
  Ratio (in/out):       X.XXX
  
  ✅ PASS: Ratio > 1.3  ← هدف
```

---

### ✅ مورد 10: Feature Importance Check
**فایل**: `model/detector_frequency.py`

```python
# اضافه شد در train():
🔍 DEBUG Top-10 RF feature importances:
    Indices: [...]
    Values:  [...]
```

**چک کنید**: importances نباید همه ~0 باشند

---

## 🚀 نحوه اجرا

### مرحله 1: تست سریع با تنظیمات فعلی
```bash
# تولید dataset با ADD_NOISE=False
python3 generate_dataset_parallel.py

# بررسی طیفی و energy ratio
python3 debug_spectral_diff.py

# detection (با همه debug prints)
python3 main_detection.py
```

---

### مرحله 2: Red-Line Test (اگر AUC هنوز پایین است)
```bash
python3 redline_test.py
```

این تست با COVERT_AMP=0.80 اجرا می‌شود.
- ✅ AUC > 0.95 → Pipeline صحیح، تنظیمات عادی مشکل دارند
- ❌ AUC < 0.95 → مشکل اساسی در feature/mask/axes

---

## 📊 نقاط کلیدی برای چک کردن

### 1. Dataset Load:
```
📅 File modified: [باید امروز باشد]
🔢 n = 200  [یا 50 در red-line test]
```

### 2. Mask Alignment:
```
DEBUG injected_symbols = [1,2,3,4,5,6,7]
DEBUG mask symbols = [1,2,3,4,5,6,7]  ← match!
```

### 3. Spectral Difference:
```
Δmag mean > 0.01  ← visible
Power diff = 4-5%  ← target
```

### 4. Energy Ratio:
```
Ratio (in/out) > 1.3  ← aligned
```

### 5. Feature Importance:
```
Top importances > 0.01  ← effective
```

---

## 🔍 علائم مشکلات رایج

### ❌ Problem 1: Mask Misalignment
**علائم**:
- Mask symbols ≠ Injected symbols
- Energy ratio < 1.0

**راه حل**:
```python
# در detector._build_default_focus_mask():
# باید دقیقاً با symbols تزریقی match کند
selected_symbols = [1,2,3,4,5,6,7]  # همین‌ها
```

---

### ❌ Problem 2: Wrong Axes
**علائم**:
- Shape نامنظم
- Δmag خیلی کم

**راه حل**:
```python
# مطمئن شوید:
grid = np.squeeze(tx_grid)  # → (symbols, subcarriers)
# نه: (subcarriers, symbols) ← اشتباه!
```

---

### ❌ Problem 3: Old Dataset
**علائم**:
- File modified: دیروز یا قبل‌تر
- COVERT_AMP در log با settings.py match نمی‌کند

**راه حل**:
```bash
rm dataset/*.pkl
python3 generate_dataset_parallel.py
```

---

### ❌ Problem 4: Weak Signal
**علائم**:
- Power diff < 3%
- Δmag mean < 0.01

**راه حل**:
```python
# در settings.py:
COVERT_AMP = 0.80  # افزایش موقت برای تست
ADD_NOISE = False
```

---

### ❌ Problem 5: Feature Extraction Failed
**علائم**:
- All importances ≈ 0
- Training AUC ≈ 0.5

**راه حل**:
- چک کنید محورها درست باشند
- چک کنید mask alignment درست باشد
- red-line test را اجرا کنید

---

## 📋 چک‌لیست نهایی قبل از اجرا

- [ ] `ADD_NOISE = False` (موقت برای debug)
- [ ] `COVERT_AMP = 0.45` (یا 0.80 برای red-line)
- [ ] `NUM_SAMPLES_PER_CLASS = 100` (یا 25 برای red-line)
- [ ] Dataset قدیمی را حذف کرده‌اید
- [ ] همه debug prints فعال هستند

---

## 🎯 معیارهای موفقیت

### تست عادی (COVERT_AMP=0.45):
```
✅ Power diff: 4-5%
✅ Energy ratio: > 1.3
✅ AUC: > 0.90
```

### Red-Line Test (COVERT_AMP=0.80):
```
✅ Power diff: > 10%
✅ Energy ratio: > 2.0
✅ AUC: > 0.95
```

---

## 📞 اگر هنوز مشکل دارید

### اگر Red-Line Test هم fail شد:
1. محورها را دوباره چک کنید (`debug_spectral_diff.py`)
2. mask alignment را دستی بسازید
3. feature extraction را ساده‌تر کنید (فقط magnitude)

### اگر Red-Line Test pass شد اما تست عادی fail:
1. `COVERT_AMP` را به 0.55-0.60 افزایش دهید
2. `ADD_NOISE` را غیرفعال نگه دارید یا کاهش دهید
3. `mask_weight` را به 15.0 افزایش دهید

---

**Status**: ✅ همه 10 مورد اعمال شد  
**Next**: Dataset بسازید و debug outputs را چک کنید!

