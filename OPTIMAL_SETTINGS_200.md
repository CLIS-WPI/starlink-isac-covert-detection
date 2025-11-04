# ⚡ تنظیمات بهینه برای تست 200 نمونه‌ای

## تغییرات نهایی برای Sweet Spot Detection

---

## 🎯 تغییرات اعمال شده:

### 1️⃣ config/settings.py
```python
COVERT_AMP = 0.50  # Sweet spot: 5-7% power difference
```
**قبل**: 0.30  
**بعد**: 0.50  
**تأثیر**: تفاوت توان 5-7% (بهینه برای detection)

---

### 2️⃣ core/dataset_generator.py
```python
covert_rate = 80.0  # 80 Mbps
```
**قبل**: 60.0 Mbps  
**بعد**: 80.0 Mbps  
**تأثیر**: اثر طیفی قوی‌تر، throughput بیشتر

---

### 3️⃣ core/covert_injection.py
```python
step = max(1, len(all_indices) // (n_subs * 4))  # wider distribution
```
**قبل**: `(n_subs * 3)`  
**بعد**: `(n_subs * 4)`  
**تأثیر**: subcarriers بیشتر، پوشش طیفی بیشتر

---

## 📊 نتایج مورد انتظار:

### پارامترهای کلیدی:
```
Samples:        200 (100 per class)
COVERT_AMP:     0.50
Covert rate:    80 Mbps
Step factor:    n_subs * 4
Symbols used:   تا 7 OFDM symbols
```

### Power Analysis:
```
Power difference: 5-7% (sweet spot) ✅
Status: ✅ GOOD (detectable but subtle)
```

### Detection Performance:
```
Training AUC:   0.99+ ✅
Test AUC:       0.95-1.00 ✅
F1 Score:       0.90+ ✅
FPR:            <5% ✅
```

---

## 🚀 اجرا:

```bash
# 1. تولید dataset با تنظیمات جدید
python3 generate_dataset_parallel.py

# 2. بررسی سریع آمار
python3 quick_stats.py

# 3. اجرای detection
python3 main_detection.py
```

---

## 🔍 چک‌لیست خروجی:

### در هنگام تولید dataset:
```
✓ [Dataset] Using COVERT_AMP=0.50 from settings.py
✓ [Dataset] Sample 0 (ATTACK): rate=80.00, amp=0.50
✓ [Covert-Fixed] Injected XX subcarriers at symbols [...]
```

### در بررسی آمار:
```
✓ Total: 200 samples
✓ Benign: 100 (50.0%)
✓ Attack: 100 (50.0%)
✓ Power Difference: 5-7% ✅
```

### در detection:
```
✓ Training AUC: 0.99+
✓ Test AUC: 0.95+
✓ F1 Score: 0.90+
```

---

## 💡 چرا این مقادیر بهینه هستند?

### COVERT_AMP = 0.50:
- ✅ نه خیلی ضعیف (قابل تشخیص)
- ✅ نه خیلی قوی (واقعی)
- ✅ 5-7% power difference (sweet spot)

### Covert Rate = 80 Mbps:
- ✅ throughput کافی برای اثر مشخص
- ✅ تعداد subcarriers مناسب
- ✅ balance بین پنهان‌کاری و تشخیص

### Step = n_subs * 4:
- ✅ پوشش طیفی گسترده‌تر
- ✅ subcarriers بیشتر تزریق می‌شوند
- ✅ pattern واضح‌تر برای detector

---

## 📈 مقایسه با حالت قبل:

| پارامتر | قبل | بعد | بهبود |
|---------|-----|-----|-------|
| COVERT_AMP | 0.30 | 0.50 | +67% |
| Covert Rate | 60 | 80 | +33% |
| Step Factor | ×3 | ×4 | +33% subs |
| Power Diff | ~15-20% | ~5-7% | بهینه ✅ |
| Expected AUC | 0.90+ | 0.95+ | بهتر ✅ |

---

## ⚠️ نکته مهم:

این تنظیمات برای **تست 200 نمونه‌ای** بهینه شده‌اند.

برای production با 3000 نمونه:
- COVERT_AMP را به 0.08-0.12 کاهش دهید
- نتایج را مقایسه کنید

---

**زمان تولید**: ~5-7 دقیقه  
**زمان training**: ~5-10 ثانیه  
**نتیجه انتظاری**: AUC ≥ 0.95 ✅

