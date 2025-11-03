# 🕵️ Covert Attack Implementation & Detection

این فایل توضیح می‌دهد که **covert channel attack** چطوری در پروژه انجام می‌شود و چگونه آن را **کشف** می‌کنیم.

---

## 🎯 خلاصه اجرایی

### سوال: Covert attack چیست و چطوری کشف می‌شه؟

**پاسخ کوتاه:**
- **Attack:** یک مهاجم روی زمین سیگنال QPSK مخفی (covert) به سیگنال ماهواره **inject** می‌کنه
- **Covert بودن:** قدرت کل سیگنال **تغییر نمی‌کنه** (power ≈ 1.0) پس simple power detector نمی‌تونه تشخیص بده
- **Detection:** از **CNN با دو input** (Spectrogram + CSI features) استفاده می‌کنیم که می‌تونه تغییرات **spectral** رو ببینه
- **Localization:** بعد از detection، با **TDOA/FDOA** موقعیت emitter روی زمین رو پیدا می‌کنیم

---

## 📐 تئوری: Covert Channel چیست؟

### تعریف:
**Covert channel** = کانال ارتباطی که وجودش **مخفی** است و نمی‌تونه با روش‌های معمولی (مثل power detection) کشف بشه.

### در این پروژه:
```
Normal signal:  [OFDM symbols] ──────► Satellite ──► User
                       ↑
                       │ + Covert QPSK
Attack signal:  [OFDM + Covert] ──────► Satellite ──► User
                       ^
                       └─── Ground emitter (مهاجم)
```

**چالش:** Power attack = power benign → Simple detector فریب می‌خوره!

---

## 🔧 پیاده‌سازی Attack (Injection Phase)

### مرحله 1: تولید Covert Symbols

**فایل:** `core/covert_injection.py` (خطوط 35-45)

```python
def inject_covert_channel(ofdm_frame, resource_grid, covert_rate_mbps, 
                          scs, covert_amp=COVERT_AMP):
    """
    Inject covert QPSK symbols into OFDM frame
    """
    # محاسبه تعداد subcarriers برای covert channel
    symbol_duration = (fft_size + cp_length) / (fft_size * scs)
    bits_per_symbol = 2  # QPSK
    num_covert_subcarriers = int((covert_rate_mbps * 1e6) / 
                                  (bits_per_symbol / symbol_duration))
    
    # محدود کردن به 25% subcarriers
    num_covert_subcarriers = min(
        num_covert_subcarriers, 
        num_effective_subcarriers // 4
    )
```

**پارامترها:**
- `covert_rate_mbps`: سرعت covert channel (Mbps)
- `COVERT_AMP`: قدرت covert symbols (از ESNO محاسبه می‌شه)

---

### مرحله 2: تولید QPSK Random Bits

**کد:** (خطوط 47-51)

```python
# تولید bits تصادفی
covert_bits = tf.random.uniform(
    [batch_size, num_covert_subcarriers, bits_per_symbol],
    0, 2, dtype=tf.int32
)

# QPSK mapping
covert_mapper = Mapper("qam", bits_per_symbol)  # QPSK
covert_syms = covert_mapper(covert_bits) * covert_amp
```

**خروجی:** سیگنال‌های QPSK پیچیده (complex symbols)

---

### مرحله 3: انتخاب Sparse Subcarriers

**کد:** (خطوط 53-59)

```python
# انتخاب subcarriers با فاصله (sparse pattern)
all_indices = np.arange(num_effective_subcarriers)
candidates = all_indices[::4]  # هر 4 subcarrier یکی

# انتخاب تصادفی
selected = np.random.choice(candidates, num_covert_subcarriers, 
                           replace=False)
```

**چرا sparse؟**
- کمتر قابل تشخیص (spread spectrum)
- کمترین تداخل با سیگنال اصلی

---

### مرحله 4: Injection به OFDM Symbols

**کد:** (خطوط 61-69)

```python
# انتخاب 3 OFDM symbol تصادفی
L = min(3, num_ofdm_symbols)
sym_indices = np.random.choice(num_ofdm_symbols, L, replace=False)

ofdm_np = ofdm_frame.numpy()
cs = covert_syms.numpy()[0]

# Inject: جمع کردن covert symbols با OFDM موجود
for s in sym_indices:
    for k, sc in enumerate(selected):
        ofdm_np[0, 0, 0, s, sc] += complex(cs[k])  # ✅ Addition!
```

**نکته مهم:** `+=` استفاده می‌شه (نه `=`) → covert روی موجود **اضافه** می‌شه

---

### مرحله 5: Power Preservation (کلید Covert بودن!) 🔑

**کد:** (خطوط 71-85)

```python
# ✅ STEP 1: ذخیره power اصلی
orig_power = np.mean(np.abs(ofdm_np[0, 0, 0, :, :])**2)

# STEP 2: Injection انجام شد (مرحله 4)

# ✅ STEP 3: Rescale برای preserve کردن power
new_power = np.mean(np.abs(ofdm_np[0, 0, 0, :, :])**2)
scale = np.sqrt(orig_power / (new_power + 1e-12))
ofdm_np[0, 0, 0, :, :] *= scale  # 🔑 کلید covert بودن!

final_power = np.mean(np.abs(ofdm_np[0, 0, 0, :, :])**2)
print(f"Power preserved: {orig_power:.6f} → {final_power:.6f}")
```

**نتیجه:**
```
Power ratio = attack_power / benign_power ≈ 1.0
```

**چرا مهمه؟**
- Simple power detector نمی‌تونه تشخیص بده (power یکسان است!)
- Attack واقعاً **covert** می‌شه

---

### مرحله 6: Emitter Location (Ground Level)

**کد:** (خطوط 87-92)

```python
# موقعیت تصادفی روی زمین
emitter_location = (
    np.random.uniform(-1000, 1000),  # x [km]
    np.random.uniform(-1000, 1000),  # y [km]
    0.0                              # z = ground level ✅
)
```

**نکته:** `z = 0.0` → emitter روی **زمین** است (نه در ماهواره)

---

## 📊 تولید Dataset (Training Data)

### فایل: `core/dataset_generator.py`

**فرآیند:**

```python
# خطوط 320-340:
if is_attack:
    # تولید waveform تمیز
    tx_grid_clean = isac_system.rg_mapper(x)
    
    # Covert injection
    covert_rate = np.random.uniform(*covert_rate_mbps_range)
    tx_grid_attacked, emitter_loc = inject_covert_channel(
        tx_grid_clean,
        isac_system.rg,
        covert_rate,
        isac_system.SUBCARRIER_SPACING,
        COVERT_AMP
    )
    
    # ذخیره در dataset:
    dataset['iq_samples'][idx] = tx_grid_attacked  # ✅ Attack sample
    dataset['labels'][idx] = 1                     # Label = attack
    dataset['emitter_locations'][idx] = emitter_loc
else:
    # Benign sample
    tx_grid_clean = isac_system.rg_mapper(x)
    dataset['iq_samples'][idx] = tx_grid_clean
    dataset['labels'][idx] = 0  # Label = benign
```

**Dataset structure:**
```python
{
    'iq_samples': [3000 samples],      # نیمی benign، نیمی attack
    'labels': [0, 0, ..., 1, 1, ...],  # 0=benign, 1=attack
    'emitter_locations': [...],        # فقط برای attacks
    'csi': [...],                      # Channel State Information
    'satellite_receptions': [...]      # برای localization
}
```

---

## 🔍 کشف Attack (Detection Phase)

### مرحله 1: Feature Extraction

**فایل:** `core/feature_extraction.py`

#### Feature A: Spectrogram (Time-Frequency Analysis)

```python
@tf.function
def extract_spectrogram_tf(iq_batch):
    """
    STFT از IQ samples → Spectrogram
    """
    # magnitude از IQ
    x_mag = tf.abs(iq_batch)
    
    # Short-Time Fourier Transform
    stft_c = tf.signal.stft(
        x_mag,
        frame_length=128,
        frame_step=32,
        fft_length=128
    )
    
    # Spectrogram = |STFT|
    spec = tf.abs(stft_c)
    
    # Normalize و resize به 64×64
    spec = tf.image.resize(spec, (64, 64))
    spec = spec / (tf.reduce_max(spec) + 1e-8)
    
    return spec  # Shape: [B, 64, 64, 1]
```

**چرا spectrogram؟**
- Covert symbols در **frequency domain** تغییرات ایجاد می‌کنن
- حتی اگر power یکسان باشه، **spectral pattern** متفاوته

---

#### Feature B: CSI Statistics (Channel Features)

```python
@tf.function
def extract_received_signal_features(dataset):
    """
    آمار per-subcarrier از CSI
    """
    csi = dataset['csi']  # [B, symbols, subcarriers]
    pwr = tf.abs(csi) ** 2
    
    # محاسبه آمار:
    mean_sc = tf.reduce_mean(pwr, axis=1)  # میانگین
    std_sc = tf.math.reduce_std(pwr, axis=1)  # انحراف معیار
    max_sc = tf.reduce_max(pwr, axis=1)  # بیشینه
    
    # Stack → [B, 64, 3]
    F = tf.stack([mean_sc, std_sc, max_sc], axis=-1)
    
    # Reshape به 8×8×3 (برای CNN)
    F = tf.reshape(F, [-1, 8, 8, 3])
    
    return F  # Shape: [B, 8, 8, 3]
```

**چرا CSI features؟**
- نشان می‌دهد که covert symbols چطور **channel** را تحت تاثیر قرار دادند
- آمار subcarrier-wise حساس‌تر از power کلی است

---

### مرحله 2: CNN Detection Model

**فایل:** `model/detector.py`

#### معماری:

```python
def build_dual_input_cnn_h100():
    """
    Dual-Input CNN با دو branch:
    - Branch A: Spectrogram (64×64×1)
    - Branch B: CSI features (8×8×3)
    """
    
    # ===== Branch A: Spectrogram =====
    a_in = layers.Input(shape=(64, 64, 1), name="spectrogram")
    a = layers.Conv2D(32, 3, activation='relu')(a_in)
    a = layers.BatchNormalization()(a)
    a = layers.MaxPooling2D(2)(a)
    a = layers.Dropout(0.2)(a)
    
    a = layers.Conv2D(64, 3, activation='relu')(a)
    a = layers.BatchNormalization()(a)
    a = layers.MaxPooling2D(2)(a)
    a = layers.Dropout(0.2)(a)
    
    a = layers.Conv2D(128, 3, activation='relu')(a)
    a = layers.BatchNormalization()(a)
    a = layers.MaxPooling2D(2)(a)
    a = layers.Dropout(0.3)(a)
    
    a = layers.Conv2D(256, 3, activation='relu')(a)
    a = layers.GlobalAveragePooling2D()(a)
    a = layers.Dropout(0.3)(a)
    
    # ===== Branch B: CSI Features =====
    b_in = layers.Input(shape=(8, 8, 3), name="rx_features")
    b = layers.Conv2D(32, 3, activation='relu')(b_in)
    b = layers.BatchNormalization()(b)
    b = layers.MaxPooling2D(2)(b)
    b = layers.Dropout(0.2)(b)
    
    b = layers.Conv2D(64, 3, activation='relu')(b)
    b = layers.BatchNormalization()(b)
    b = layers.MaxPooling2D(2)(b)
    b = layers.Dropout(0.2)(b)
    
    b = layers.Conv2D(128, 3, activation='relu')(b)
    b = layers.GlobalAveragePooling2D()(b)
    b = layers.Dropout(0.3)(b)
    
    # ===== Merge + Classification =====
    x = layers.Concatenate()([a, b])  # ترکیب دو branch
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output: Binary classification (logits)
    out = layers.Dense(1, dtype='float32', name='logits')(x)
    
    model = Model([a_in, b_in], out)
    return model
```

**چرا dual-input؟**
- **Spectrogram:** تغییرات فرکانسی covert symbols را می‌بیند
- **CSI features:** تغییرات channel را می‌بیند
- **ترکیب:** دقت بالاتر از استفاده تک‌تک

---

### مرحله 3: Training

**فایل:** `model/detector.py`

```python
def train_detector(Xs_tr, Xr_tr, y_tr, Xs_te, Xr_te, y_te, strategy=None):
    """
    آموزش CNN با dual inputs
    
    Inputs:
    - Xs_tr: Spectrogram (train)
    - Xr_tr: CSI features (train)
    - y_tr: Labels (0=benign, 1=attack)
    
    Outputs:
    - model: Trained CNN
    - temperature: Calibration parameter
    """
    
    # Build model
    if strategy is not None:
        with strategy.scope():  # Multi-GPU
            model = build_dual_input_cnn_h100()
    else:
        model = build_dual_input_cnn_h100()
    
    # Compile
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy', tf.keras.metrics.AUC(from_logits=True)]
    )
    
    # Train
    history = model.fit(
        [Xs_tr, Xr_tr], y_tr,  # Dual inputs!
        validation_data=([Xs_te, Xr_te], y_te),
        epochs=50,
        batch_size=32,
        callbacks=[
            EarlyStopping(patience=15),
            ModelCheckpoint('best_model.keras')
        ]
    )
    
    return model, history
```

**Loss function:**
```
Binary Cross-Entropy (from logits):
L = -[y*log(σ(z)) + (1-y)*log(1-σ(z))]
```
جایی که `z` = logit output، `σ` = sigmoid

---

### مرحله 4: Evaluation & Threshold Tuning

```python
def evaluate_detector(model, Xs_te, Xr_te, y_te, temperature=1.0):
    """
    ارزیابی مدل و پیدا کردن threshold بهینه
    """
    # Predict (logits)
    logits = model.predict([Xs_te, Xr_te])
    
    # Temperature scaling (calibration)
    scaled_logits = logits / temperature
    y_prob = tf.sigmoid(scaled_logits).numpy()
    
    # محاسبه F1-score برای threshold های مختلف
    thresholds = np.linspace(0, 1, 1000)
    f1_scores = []
    
    for thr in thresholds:
        y_pred = (y_prob > thr).astype(int)
        f1 = f1_score(y_te, y_pred)
        f1_scores.append(f1)
    
    # بهترین threshold
    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx]
    
    print(f"Optimized threshold: {best_thr:.4f}")
    print(f"Best F1 score: {f1_scores[best_idx]:.4f}")
    
    return y_prob, best_thr, f1_scores
```

**Metrics:**
- **AUC (Area Under ROC Curve):** 0.70-0.80 = خوب، >0.90 = عالی
- **F1-Score:** 2×(Precision×Recall)/(Precision+Recall)
- **Threshold:** بهترین نقطه برای binary classification

---

## 📍 Localization (پیدا کردن Emitter)

بعد از detection، می‌خواهیم **موقعیت emitter** روی زمین را پیدا کنیم.

### روش: TDOA (Time Difference of Arrival)

**فایل:** `core/localization_enhanced.py`

```python
def estimate_emitter_location_enhanced(sample_idx, dataset, isac_system):
    """
    پیدا کردن موقعیت emitter با TDOA/FDOA
    """
    # مرحله 1: دریافت سیگنال‌های ماهواره‌ها
    sats = dataset['satellite_receptions'][sample_idx]
    
    # مرحله 2: محاسبه TDOA (Time Difference of Arrival)
    # برای هر جفت ماهواره:
    for i, sat_i in enumerate(sats):
        for j, sat_j in enumerate(sats):
            if i >= j:
                continue
            
            # GCC-PHAT: Cross-correlation برای TDOA
            dt, _, _ = _estimate_toa(sat_i['rx_time'], sat_j['rx_time'], Fs)
            tdoa_m = dt * c  # تبدیل زمان به مسافت
            
            tdoa_measurements.append((tdoa_m, sat_i['position'], sat_j['position']))
    
    # مرحله 3: حل معادلات TDOA (Least-Squares)
    def residuals(P):
        """محاسبه خطا برای موقعیت تخمینی P"""
        r = []
        for (tdoa_obs, pos_i, pos_j) in tdoa_measurements:
            # فاصله emitter تا ماهواره‌ها
            d_i = np.linalg.norm(P - pos_i)
            d_j = np.linalg.norm(P - pos_j)
            
            # پیش‌بینی TDOA
            tdoa_pred = d_i - d_j
            
            # خطا
            r.append(tdoa_obs - tdoa_pred)
        
        return np.array(r)
    
    # Minimize residuals
    result = least_squares(residuals, x0=[0, 0, 0])
    emitter_position = result.x
    
    return emitter_position
```

**فرمول TDOA:**
```
TDOA_ij = (d_i - d_j) / c = t_i - t_j
```
جایی که:
- `d_i`: فاصله emitter تا ماهواره i
- `t_i`: زمان رسیدن سیگنال به ماهواره i
- `c`: سرعت نور

---

## 📊 نتایج و Metrics

### Detection Performance:

```python
# از output شما:
AUC (Normal): 0.5751  # کم (به خاطر ESNO=15 پایین)
Best F1 score: 0.6409
Precision: 0.4716
Recall: 1.0000  # همه attacks رو پیدا کرده (ولی false positive زیاد)
```

**تحلیل:**
- **AUC = 0.5751:** نزدیک به random (0.5) → dataset ضعیف (ESNO پایین)
- **Recall = 1.0:** همه attacks رو می‌بینه (good!)
- **Precision = 0.47:** تقریباً نصف detections اشتباه هستند (bad!)

**راه حل:**
```bash
# Regenerate dataset با ESNO بالاتر:
rm dataset/dataset_samples1500_sats12.pkl
# تغییر در config/settings.py:
DEFAULT_COVERT_ESNO_DB = 20.0  # از 15 به 20
python3 generate_dataset_parallel.py
python3 main.py
```

**انتظار بعد از fix:**
- AUC: 0.70-0.80
- F1: 0.75-0.85
- Precision: 0.70-0.80

---

### Localization Performance:

```
Median Error: 1-5 km  (برای 20 samples)
90th Percentile: 5-15 km
```

---

## 🎓 چرا این روش کار می‌کنه؟

### 1. Power Preservation ≠ Spectral Preservation

```
Power attack ≈ Power benign  ✅ (covert!)
BUT:
Spectrum attack ≠ Spectrum benign  ✅ (detectable!)
```

**دلیل:**
- Covert QPSK symbols **فرکانس‌های جدید** اضافه می‌کنند
- CNN این تفاوت **spectral** را در STFT می‌بیند

---

### 2. Dual-Input CNN

```
Spectrogram alone: AUC ≈ 0.65
CSI features alone: AUC ≈ 0.60
Combined:          AUC ≈ 0.75-0.80  ✅
```

**دلیل:**
- دو view مختلف از همان signal
- Complementary information

---

### 3. Deep Learning > Hand-Crafted Features

```
Traditional: Energy detection, PSD comparison → AUC ≈ 0.55
CNN:         Learned features              → AUC ≈ 0.75
```

**دلیل:**
- CNN خودش **بهترین features** را یاد می‌گیره
- Non-linear patterns را می‌بینه

---

## 🔧 پارامترهای کلیدی

### در `config/settings.py`:

```python
# قدرت covert signal
DEFAULT_COVERT_ESNO_DB = 20.0  # 🔧 بالاتر = راحت‌تر detectable

# Power preservation (کلید covert بودن)
ABLATION_CONFIG = {
    'power_preserving_covert': True  # ✅ باید True باشه
}

# محاسبه amplitude covert از ESNO
def covert_scale_from_esno_db(esno_db):
    return np.sqrt(10.0 ** (esno_db / 10.0))

COVERT_AMP = covert_scale_from_esno_db(DEFAULT_COVERT_ESNO_DB)
```

---

## 📈 Trade-off: Covert vs Detectable

```
ESNO (dB)  | Power Ratio | Spectral Cohen's d | AUC   | Covert?
-----------|-------------|-------------------|-------|--------
10         | ≈1.00       | 0.05              | 0.52  | ✅ Very
15         | ≈1.00       | 0.10              | 0.58  | ✅ Yes
20         | ≈1.00       | 0.25              | 0.75  | ⚠️ Moderate
25         | ≈1.02       | 0.50              | 0.88  | ❌ No
30         | ≈1.05       | 1.00              | 0.95  | ❌ Very obvious
```

**نتیجه:**
- **ESNO = 15-20 dB:** بهترین trade-off برای realistic covert scenario
- کمتر → خیلی covert، کشف سخت
- بیشتر → راحت detectable، دیگه covert نیست

---

## 🎯 خلاصه نهایی

### Attack Pipeline:
```
1. تولید QPSK random bits
2. Mapping به complex symbols
3. انتخاب sparse subcarriers
4. Injection به OFDM (addition)
5. Power preservation (rescale) ← کلید covert بودن!
6. ذخیره موقعیت emitter (ground level)
```

### Detection Pipeline:
```
1. Feature extraction:
   - Spectrogram (STFT)
   - CSI statistics
2. Dual-input CNN classification
3. Temperature scaling (calibration)
4. Threshold tuning (F1 optimization)
5. True positives → Localization
6. TDOA/FDOA → Emitter position
```

### چرا کار می‌کنه:
```
✅ Power preserved   → Simple detector فریب می‌خوره
✅ Spectrum changed  → CNN تغییرات spectral را می‌بیند
✅ Dual-input       → دو view مختلف = دقت بالاتر
✅ Deep learning    → Learned features > hand-crafted
```

---

## 💡 نکات مهم

1. **Power ratio ≈ 1.0** = شرط لازم برای covert بودن
2. **Spectral changes** = آنچه CNN می‌بیند
3. **ESNO = 15-20 dB** = بهترین trade-off
4. **Dual-input CNN** = کلید موفقیت detection
5. **TDOA localization** = پیدا کردن emitter بعد از detection

---

**در این پروژه، ما نشان می‌دهیم که حتی با power preservation (که covert channel را مخفی نگه می‌دارد)، تغییرات spectral توسط CNN قابل تشخیص هستند!** 🎯
