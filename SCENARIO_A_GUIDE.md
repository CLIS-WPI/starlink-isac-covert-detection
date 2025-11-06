# Scenario A — Insider@Satellite (Downlink) — راهنمای اجرا

## 📋 پیش‌نیازها

✅ مطمئن شوید که `INSIDER_MODE = 'sat'` در `config/settings.py` است.

```python
# config/settings.py
INSIDER_MODE = 'sat'  # ✅ برای Scenario A
```

## 🚀 دستورات اجرا

### مرحله 1: ساخت دیتاست

```bash
python3 generate_dataset_parallel.py \
  --num-samples 500 \
  --num-satellites 12
```

**توضیح:**
- `--num-samples 500`: 500 نمونه per class = 1000 نمونه کل
- `--num-satellites 12`: 12 ماهواره برای TDoA
- دیتاست در `dataset/dataset_samples500_sats12.pkl` ذخیره می‌شود

**زمان تقریبی:** ~10-15 دقیقه (بسته به GPU)

---

### مرحله 2: بررسی صحت دیتاست (اختیاری اما توصیه می‌شود)

```bash
# بررسی کلی دیتاست
python3 validate_dataset.py

# چک صحت تزریق (pre-channel, power_diff_pct, pattern_boost, doppler_hz)
python3 verify_injection_correctness.py

# چک consistency (برای multi-GPU)
python3 check_dataset_consistency.py
```

**انتظار:**
- ✅ Power diff < 5%
- ✅ Pattern boost در subcarriers 24-39
- ✅ Doppler non-zero و reasonable
- ✅ Labels: 50/50 split

---

### مرحله 3: Train CNN-only

```bash
python3 main_detection_cnn.py \
  --epochs 50 \
  --batch-size 512
```

**توضیح:**
- `--epochs 50`: حداکثر 50 epochs (با early stopping)
- `--batch-size 512`: بهینه برای H100 GPU
- نتایج در `result/scenario_a/detection_results_cnn.json`
- مدل در `model/scenario_a/cnn_detector.keras`

**زمان تقریبی:** ~2-3 دقیقه

---

### مرحله 4: Train CNN+CSI

```bash
python3 main_detection_cnn.py \
  --use-csi \
  --epochs 50 \
  --batch-size 512
```

**توضیح:**
- `--use-csi`: فعال‌سازی CSI fusion (real/imag channels)
- نتایج در `result/scenario_a/detection_results_cnn_csi.json`
- مدل در `model/scenario_a/cnn_detector_csi.keras`

**زمان تقریبی:** ~3-5 دقیقه

---

### مرحله 5: بررسی نتایج

```bash
# مشاهده نتایج CNN-only
cat result/scenario_a/detection_results_cnn.json | jq '.metrics'

# مشاهده نتایج CNN+CSI
cat result/scenario_a/detection_results_cnn_csi.json | jq '.metrics'

# مشاهده meta log (per-sample metadata)
head result/scenario_a/run_meta_log.csv
head result/scenario_a/run_meta_log_csi.csv
```

---

## 📊 نتایج مورد انتظار

بر اساس اجرای قبلی با `COVERT_AMP=0.5` و `POWER_PRESERVING_COVERT=True`:

### CNN-only:
- **AUC:** ~0.9997 ✅
- **Precision:** ~1.0000
- **Recall:** ~0.4000
- **F1 Score:** ~0.5714

### CNN+CSI:
- **AUC:** ~0.9814 ✅
- **Precision:** ~0.5379
- **Recall:** ~0.9933
- **F1 Score:** ~0.6979

### Physical Metrics:
- **Power diff:** ~0.14% (ultra-covert) ✅
- **Doppler:** ~-4920 Hz (mean), ±395516 Hz (std)
- **CSI variance:** ~1.64e-02

---

## 📁 ساختار فایل‌های خروجی

```
result/scenario_a/
├── detection_results_cnn.json      # نتایج CNN-only
├── detection_results_cnn_csi.json   # نتایج CNN+CSI
├── run_meta_log.csv                 # Meta log CNN-only
└── run_meta_log_csi.csv             # Meta log CNN+CSI

model/scenario_a/
├── cnn_detector.keras               # مدل CNN-only
└── cnn_detector_csi.keras           # مدل CNN+CSI
```

---

## 🔄 انتقال فایل‌های قدیمی (اگر لازم باشد)

اگر فایل‌های قدیمی در `result/` دارید:

```bash
python3 organize_results.py
```

این اسکریپت فایل‌های `result/*_sat.*` را به `result/scenario_a/` منتقل می‌کند.

---

## ⚠️ نکات مهم

1. **Normalization:** mean/std فقط از train data محاسبه می‌شود (no data leakage) ✅
2. **Injection Location:** Subcarriers 24-39 (middle band) ✅
3. **Power Preserving:** `POWER_PRESERVING_COVERT = True` ✅
4. **CSI:** Real/imag channels (dual-channel) ✅

---

## 🐛 عیب‌یابی

اگر AUC پایین بود (< 0.70):

1. چک کنید که `INSIDER_MODE = 'sat'` است
2. چک کنید که `COVERT_AMP = 0.5` است
3. چک کنید که `POWER_PRESERVING_COVERT = True` است
4. دیتاست را دوباره بسازید
5. `verify_injection_correctness.py` را اجرا کنید

---

## ✅ آماده برای مقاله

پس از اجرای موفق:
- ✅ نتایج در `result/scenario_a/` ذخیره شده
- ✅ مدل‌ها در `model/scenario_a/` ذخیره شده
- ✅ کاملاً جدا از Scenario B
- ✅ آماده برای استفاده در مقاله

