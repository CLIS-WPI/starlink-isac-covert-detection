# 🔍 Debugging Summary: CNN Detection Failure

## تاریخ: 5 نوامبر 2025

## 📊 خلاصه مشکل
CNN نمی‌تواند covert channel را تشخیص دهد. با وجود تلاش‌های متعدد، AUC همیشه حدود 0.47-0.54 (worse than or barely better than random) بوده است.

---

## 🛠️ تغییرات اعمال شده (به ترتیب زمانی)

### 1. **Power Preservation Fix**
- **مشکل**: `ABLATION_CONFIG['power_preserving_covert']` به `False` override می‌شد
- **راه‌حل**: Override را حذف کردیم، حالا از `settings.py` استفاده می‌کند
- **فایل‌ها**: `core/dataset_generator.py`, `core/covert_injection.py`
- **نتیجه**: Power difference از 62% به ~0.3% کاهش یافت ✅

### 2. **Amplitude Consistency Fix**
- **مشکل**: `covert_syms` در `inject_covert_channel` با `* 2.0` ضرب می‌شد
- **راه‌حل**: ضریب `* 2.0` را حذف کردیم
- **فایل**: `core/covert_injection.py` (line 57)
- **نتیجه**: Amplitude consistent شد ✅

### 3. **rx_grids Merge Fix**
- **مشکل**: `rx_grids` در `generate_dataset_parallel.py` merge نمی‌شد
- **راه‌حل**: `'rx_grids'` را به merge pipeline اضافه کردیم
- **فایل**: `generate_dataset_parallel.py`
- **نتیجه**: rx_grids در dataset ذخیره می‌شود ✅

### 4. **rx_grids Shape Fix**
- **مشکل**: rx_grids shape (12000, ...) بود چون از همه 12 satellites ذخیره می‌شد
- **راه‌حل**: فقط satellite اول (`sat_idx == 0`) را ذخیره کنیم
- **فایل**: `core/dataset_generator.py`
- **نتیجه**: Shape (400, 1, 10, 64) درست شد ✅

### 5. **rx_grids Signal Source Fix**
- **مشکل**: از `rx_grid_cropped` استفاده می‌شد (degraded signal با power 0.011)
- **راه‌حل**: از `y_grid_noisy` استفاده کنیم (بعد از channel + noise + injection)
- **فایل**: `core/dataset_generator.py` (line 527)
- **نتیجه**: Signal quality بهتر شد اما power هنوز 0.017 بود ⚠️

### 6. **Per-Sample Power Normalization (اول)**
- **مشکل**: rx_grids power خیلی پایین (0.017)
- **راه‌حل**: Normalize هر sample به unit power
- **نتیجه**: Power = 1.0 شد اما **pattern از بین رفت!** (9.47% → 1.99%) ❌

### 7. **حذف Per-Sample Normalization**
- **مشکل**: Per-sample normalization pattern را destroy می‌کرد
- **راه‌حل**: Normalization را حذف کردیم
- **نتیجه**: Pattern برگشت (9.47%) اما power پایین ماند (0.0095) ⚠️

### 8. **افزایش COVERT_AMP**
- **مشکل**: Pattern با amp=0.2 خیلی ضعیف بود
- **راه‌حل**: COVERT_AMP: 0.2 → 0.3
- **نتیجه**: Pattern همچنان 9.47% بود ⚠️

### 9. **Fixed Injection Pattern**
- **مشکل**: Semi-fixed injection روی 12 band مختلف، pattern inconsistent بود
- **راه‌حل**: 
  - `USE_SEMI_FIXED_PATTERN = False`
  - `inject_covert_channel_fixed` را تغییر دادیم تا **همیشه** subcarriers 0-15 و symbols [1,3,5,7] استفاده کند
- **فایل‌ها**: `config/settings.py`, `core/covert_injection.py`
- **نتیجه**: Pattern consistent شد، 9.47% در covert band ✅

### 10. **CNN Preprocessing Fix**
- **مشکل**: Per-sample magnitude normalization در CNN pattern را reverse می‌کرد!
  - Before: +9.47%
  - After: -3.57% (برعکس شده!)
- **راه‌حل**: از **global normalization** استفاده کنیم (روی همه samples)
- **فایل**: `model/detector_cnn.py` (line 258)
- **نتیجه**: Pattern حفظ شد (9.47% → 9.47%) ✅

### 11. **افزایش تعداد Samples**
- **مشکل**: 400 samples برای CNN کم است
- **راه‌حل**: `NUM_SAMPLES_PER_CLASS`: 200 → 500 (total: 1000)
- **نتیجه**: Training stability بهتر شد اما AUC همچنان 0.47 ❌

### 12. **افزایش COVERT_AMP به 0.5**
- **مشکل**: با amp=0.3، pattern visible بود اما learnable نبود (Logistic Regression هم AUC=0.47)
- **راه‌حل**: COVERT_AMP: 0.3 → 0.5
- **نتیجه**: Pattern **برعکس شد!** (-7.17%) ❌❌❌

---

## 🔴 مشکلات باقی‌مانده

### مشکل اصلی: Pattern Direction Inconsistency

با `COVERT_AMP=0.5`, pattern **معکوس** شده است:
- انتظار: Attack > Benign در covert band
- واقعیت: Attack < Benign (-7.17%)

این نشان می‌دهد که:
1. **یا** injection به درستی اعمال نمی‌شود
2. **یا** channel/noise pattern را تخریب می‌کند
3. **یا** مشکل fundamental در روش injection وجود دارد

### کشفیات کلیدی

1. **Logistic Regression هم fail کرد** (AUC=0.47)
   - این ثابت می‌کند مشکل در **data** است، نه CNN architecture

2. **Sample-level variance خیلی زیاد است**
   - بعضی samples: +4046%
   - بعضی samples: -8%
   - **Overlap کامل** بین benign و attack classes

3. **Per-sample normalization مخرب است**
   - در CNN preprocessing باعث reverse شدن pattern می‌شود
   - باید از global normalization استفاده کرد

4. **Channel attenuation شدید**
   - rx_grids power: 0.01-0.02 (100× کمتر از tx_grids)
   - این noise را dominant می‌کند

---

## 📋 تست‌های انجام شده

| Test | Result | AUC | Pattern |
|------|--------|-----|---------|
| Initial (400 samples, amp=0.2) | ❌ | 0.44 | Not measured |
| After fixes (400 samples, amp=0.3) | ❌ | 0.54-0.58 | 9.47% |
| Global norm (400 samples, amp=0.3) | ❌ | 0.54 | 9.47% preserved |
| More data (1000 samples, amp=0.3) | ❌ | 0.47 | 9.47% |
| Logistic Regression (1000, amp=0.3) | ❌ | 0.47 | 9.47% |
| Stronger amp (1000, amp=0.5) | ❌ | Not tested | -7.17% (REVERSED!) |

---

## 🎯 توصیه‌های بعدی

### گزینه 1: Debug Injection Direction
بررسی کنید که injection واقعاً در **صحیح** direction اتفاق می‌افتد:
```python
# Check if attack samples actually have MORE energy in covert band
attack_covert_mean = np.mean(np.abs(attack_rx[:, :, :16]))
benign_covert_mean = np.mean(np.abs(benign_rx[:, :, :16]))
assert attack_covert_mean > benign_covert_mean, "Injection direction wrong!"
```

### گزینه 2: Inject PRE-CHANNEL
به جای POST-CHANNEL injection, inject کنید BEFORE channel:
- Pattern قوی‌تر خواهد بود
- Channel attenuation pattern را کمتر تخریب می‌کند

### گزینه 3: استفاده از tx_grids
اگر injection را pre-channel انجام دهیم، می‌توانیم از `tx_grids` استفاده کنیم:
- Power بالا (1.0)
- Noise کم
- Pattern واضح

### گزینه 4: تغییر روش Injection
به جای additive/weighted injection, از **subcarrier replacement** استفاده کنید:
```python
# Replace specific subcarriers completely
ofdm_frame[:, :, :, symbols, subcarriers] = covert_signal
```

---

## 📊 فایل‌های تغییر یافته

1. `config/settings.py`
   - COVERT_AMP: 0.2 → 0.5
   - USE_SEMI_FIXED_PATTERN: False
   - NUM_SAMPLES_PER_CLASS: 500

2. `core/dataset_generator.py`
   - rx_grids source: y_grid_noisy
   - rx_grids filtering: فقط sat_idx==0
   - Power normalization حذف شد

3. `core/covert_injection.py`
   - inject_covert_channel_fixed: subcarriers 0-15 fixed
   - inject_covert_channel_fixed: symbols [1,3,5,7] fixed
   - Amplitude × 2.0 حذف شد

4. `model/detector_cnn.py`
   - Preprocessing: global normalization به جای per-sample

5. `generate_dataset_parallel.py`
   - rx_grids به merge pipeline اضافه شد

---

## 🔚 نتیجه‌گیری

با وجود تمام تلاش‌ها و fix های متعدد:
- ✅ مشکلات technical (shape, merge, power preservation) حل شدند
- ✅ Pattern visible شد (9.47%)
- ❌ **اما ML نمی‌تواند یاد بگیرد!**

این نشان می‌دهد مشکل **fundamental** در روش injection یا data generation وجود دارد که نیاز به بازنگری اساسی دارد.

**تست بعدی پیشنهادی**: بررسی injection direction و debugging کامل injection pipeline.
