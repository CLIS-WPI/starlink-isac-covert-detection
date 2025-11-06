# Scenario B — Insider@Ground (Uplink → Relay → Downlink) — راهنمای اجرا

## 📋 پیش‌نیازها

✅ مطمئن شوید که `INSIDER_MODE = 'ground'` در `config/settings.py` است.

```python
# config/settings.py
INSIDER_MODE = 'ground'  # ✅ برای Scenario B
```

## 🔄 تفاوت با Scenario A

| ویژگی | Scenario A (Satellite) | Scenario B (Ground) |
|--------|------------------------|---------------------|
| **Injection Point** | Satellite downlink | Ground terminal uplink |
| **Signal Path** | Direct downlink | Uplink → Relay → Downlink |
| **Channel Effects** | Single channel | Double channel (uplink + downlink) |
| **Noise** | Single noise | Double noise (relay amplifies noise) |
| **Expected AUC** | ~1.0 (CNN-only) | ~0.85-0.95 (CNN-only) |
| | ~0.96 (CNN+CSI) | ~0.90+ (CNN+CSI) |

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
- ⚠️ **نکته:** اگر دیتاست Scenario A را می‌خواهید نگه دارید، ابتدا rename کنید:

```bash
# نگه داشتن دیتاست Scenario A
mv dataset/dataset_samples500_sats12.pkl dataset/dataset_scenario_a.pkl

# بعد از ساخت دیتاست Scenario B
mv dataset/dataset_samples500_sats12.pkl dataset/dataset_scenario_b.pkl
```

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
- ✅ Insider mode: 'ground'

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
- نتایج در `result/scenario_b/detection_results_cnn.json`
- مدل در `model/scenario_b/cnn_detector.keras`

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
- نتایج در `result/scenario_b/detection_results_cnn_csi.json`
- مدل در `model/scenario_b/cnn_detector_csi.keras`

**زمان تقریبی:** ~3-5 دقیقه

---

### مرحله 5: بررسی نتایج

```bash
# مشاهده نتایج CNN-only
cat result/scenario_b/detection_results_cnn.json | jq '.metrics'

# مشاهده نتایج CNN+CSI
cat result/scenario_b/detection_results_cnn_csi.json | jq '.metrics'

# مشاهده meta log (per-sample metadata)
head result/scenario_b/run_meta_log.csv
head result/scenario_b/run_meta_log_csi.csv
```

---

## 📊 نتایج مورد انتظار

بر اساس تفاوت‌های Scenario B با A:

### CNN-only:
- **AUC:** ~0.85-0.95 (پایین‌تر از Scenario A به‌دلیل رله)
- **Precision:** ~0.70-0.90
- **Recall:** ~0.30-0.50
- **F1 Score:** ~0.40-0.60

### CNN+CSI:
- **AUC:** ~0.90+ ✅ (هدف: ≥ 0.90)
- **Precision:** ~0.60-0.80
- **Recall:** ~0.90-0.99
- **F1 Score:** ~0.70-0.85

### Physical Metrics:
- **Power diff:** < 5% (ultra-covert) ✅
- **Doppler:** Similar to Scenario A
- **CSI variance:** ممکن است کمی بالاتر باشد (به‌دلیل رله)

---

## 📁 ساختار فایل‌های خروجی

```
result/scenario_b/
├── detection_results_cnn.json      # نتایج CNN-only
├── detection_results_cnn_csi.json   # نتایج CNN+CSI
├── run_meta_log.csv                 # Meta log CNN-only
└── run_meta_log_csi.csv             # Meta log CNN+CSI

model/scenario_b/
├── cnn_detector.keras               # مدل CNN-only
└── cnn_detector_csi.keras           # مدل CNN+CSI
```

---

## 🔄 مقایسه با Scenario A

پس از اجرای Scenario B، می‌توانید نتایج را مقایسه کنید:

```bash
# مقایسه AUC
echo "Scenario A - CNN-only:"
cat result/scenario_a/detection_results_cnn.json | jq '.metrics.auc'
echo "Scenario B - CNN-only:"
cat result/scenario_b/detection_results_cnn.json | jq '.metrics.auc'

echo "Scenario A - CNN+CSI:"
cat result/scenario_a/detection_results_cnn_csi.json | jq '.metrics.auc'
echo "Scenario B - CNN+CSI:"
cat result/scenario_b/detection_results_cnn_csi.json | jq '.metrics.auc'
```

---

## ⚠️ نکات مهم

1. **Normalization:** mean/std فقط از train data محاسبه می‌شود (no data leakage) ✅
2. **Injection Location:** Subcarriers 24-39 (middle band) ✅
3. **Power Preserving:** `POWER_PRESERVING_COVERT = True` ✅
4. **CSI:** Real/imag channels (dual-channel) ✅
5. **Relay Effect:** Amplify-and-Forward (نویز مضاعف) ⚠️

---

## 🐛 عیب‌یابی

اگر AUC پایین بود (< 0.85):

1. چک کنید که `INSIDER_MODE = 'ground'` است
2. چک کنید که `COVERT_AMP = 0.5` است
3. چک کنید که `POWER_PRESERVING_COVERT = True` است
4. دیتاست را دوباره بسازید
5. `verify_injection_correctness.py` را اجرا کنید
6. توجه: Scenario B به‌طور طبیعی AUC پایین‌تری دارد (به‌دلیل رله)

---

## ✅ آماده برای مقاله

پس از اجرای موفق:
- ✅ نتایج در `result/scenario_b/` ذخیره شده
- ✅ مدل‌ها در `model/scenario_b/` ذخیره شده
- ✅ کاملاً جدا از Scenario A
- ✅ آماده برای مقایسه و استفاده در مقاله

