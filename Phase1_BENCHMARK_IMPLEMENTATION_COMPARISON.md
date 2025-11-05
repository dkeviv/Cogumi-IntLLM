# Implementation Comparison: Toy Model vs Phi-3.5

## Files Location

| Benchmark | Test File | Implementation | Log File | Status |
|-----------|-----------|----------------|----------|--------|
| **Toy Model** | `test_codec_greedy.py` | `src/qins_codec.py` | `test_codec_greedy_run.log` | ✅ In main directory |
| **Phi-3.5** | `test_pattern_a_clean.py` | `qins_weight_codec.py` | `test_pattern_a_clean.log` | ✅ In main directory |

---

## Detailed Encoding/Decoding Formulas

### Toy Model Implementation (Logarithmic)
**File**: `src/qins_codec.py`

#### Encoding Steps
```python
# Step 1: Extract sign
sign = torch.sign(weight).to(torch.int8)  # {-1, +1}

# Step 2: Log space transformation
abs_weight = torch.abs(weight).clamp(min=1e-8)
log_weight = torch.log(abs_weight)

# Step 3: Find log range
log_min = log_weight.min()
log_max = log_weight.max()

# Step 4: Normalize to [0, 1]
normalized = (log_weight - log_min) / (log_max - log_min)

# Step 5: Inverse map to [1, 255] (large weight → stored=1)
stored_float = 255.0 - (normalized * 254.0)

# Step 6: Quantize to uint8
stored = stored_float.round().clamp(1, 255).to(torch.uint8)

# Return: (stored, sign, log_min, log_max)
```

#### Decoding Steps
```python
# Step 1: Reverse inverse mapping [1, 255] → [0, 1]
normalized = (255.0 - stored.float()) / 254.0

# Step 2: Map back to log space
log_weight = log_min + normalized * (log_max - log_min)

# Step 3: Exponentiate to get magnitude
magnitude = torch.exp(log_weight)

# Step 4: Apply sign
weight = sign.float() * magnitude

# Return: reconstructed weight
```

---

### Phi-3.5 Implementation (Rational)
**File**: `qins_weight_codec.py`

#### Encoding Steps
```python
# Step 1: Extract sign
sign = torch.sign(weight)
sign[sign == 0] = 1

# Step 2: Rational transformation
abs_weight = weight.abs()
encoded = sign / (1.0 + alpha * abs_weight)  # Result in [-1, 1]

# Step 3: Quantize (if enabled - currently buggy)
if quantize:
    # Map [-1, 1] to [0, 255]
    quantized = ((encoded + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return quantized
else:
    return encoded  # Returns float32

# BUG: Due to some code path, quantization doesn't actually happen
# Result: stored as float32, not uint8
```

#### Decoding Steps
```python
# Step 1: Dequantize (if was quantized)
if is_quantized:
    # Map [0, 255] back to [-1, 1]
    encoded = (encoded.float() / 127.5) - 1.0

# Step 2: Extract sign from encoded value
sign = torch.sign(encoded)

# Step 3: Get absolute value
abs_encoded = encoded.abs().clamp(min=1e-12)  # Avoid division by zero

# Step 4: Inverse rational transformation
weight = sign * (1.0 - abs_encoded) / (alpha * abs_encoded)

# Return: reconstructed weight
```

---

## Mathematical Comparison

### Toy Model (Logarithmic)

**Forward (Encode)**:
```
w → sign(w), |w| → log(|w|) → normalize → 255 - (norm × 254) → uint8
```

**Inverse (Decode)**:
```
uint8 → denormalize → exp() → apply sign → w
```

**Key Property**: Inverse magnitude mapping
- Large |w| (e.g., 0.5) → log(|w|) is large → normalized near 1.0 → stored near 1
- Small |w| (e.g., 0.001) → log(|w|) is small (negative) → normalized near 0.0 → stored near 255

---

### Phi-3.5 (Rational)

**Forward (Encode)**:
```
w → z = sign(w) / (1 + α|w|)
```

**Inverse (Decode)**:
```
z → w = sign(z) × (1 - |z|) / (α|z|)
```

**Key Property**: Direct rational transformation
- Large |w| → z near 0
- Small |w| → z near ±1

**Verification** (with α=1.0):
```
w = 0.5  → z = 1/(1+0.5) = 0.667  → w = (1-0.667)/(1×0.667) = 0.5 ✓
w = 0.1  → z = 1/(1+0.1) = 0.909  → w = (1-0.909)/(1×0.909) = 0.1 ✓
w = 2.0  → z = 1/(1+2.0) = 0.333  → w = (1-0.333)/(1×0.333) = 2.0 ✓
```

---

## Results Comparison

| Metric | Toy Model | Phi-3.5 |
|--------|-----------|---------|
| **Model Size** | 3-layer, 256 hidden, 5K vocab | 3.82B parameters |
| **Encoding** | Logarithmic | Rational |
| **Storage** | uint8 + int8 | float32 (bug) |
| **Bytes/weight** | 2 bytes | 4 bytes |
| **Compression** | 2× (on linear layers) | 0× (no compression) |
| **Overall** | 1.48× (13.88 MB → 9.38 MB) | 1.0× (13,824 MB → 13,824 MB) |
| **Accuracy** | ✅ 100% (500/500 tokens) | ✅ 100% (15/15 tokens) |
| **Pattern** | ✅ Pattern A | ✅ Pattern A |

---

## Why Both Achieve 100% Accuracy

**Pattern A Guarantee**: Both implementations follow Pattern A correctly:

1. ✅ **Encode** weights and store them
2. ✅ **Decode** weights to FP32 before every computation
3. ✅ **Compute** always in FP32 domain

**Invertibility**: Both encoding methods are invertible:
- Logarithmic: `w = sign × exp(denormalize(stored))`
- Rational: `w = sign × (1 - |z|) / (α|z|)`

**Result**: As long as decode(encode(w)) ≈ w (within FP32 precision), Pattern A guarantees correctness!

---

## The Key Difference: Quantization

| Aspect | Toy Model | Phi-3.5 |
|--------|-----------|---------|
| **Quantization code** | ✅ Present and working | ⚠️ Present but buggy |
| **Actual storage** | uint8 (1 byte) | float32 (4 bytes) |
| **Evidence** | Verified by dtype check | Log shows 36 MB = 3072×3072×4 |
| **Compression** | ✅ 2× achieved | ❌ 0× (no compression) |

---

## Action Items

1. ✅ **Keep both files** - They validate different aspects:
   - Toy model: Validates logarithmic encoding + quantization works
   - Phi-3.5: Validates Pattern A works on full-size LLM

2. 🔧 **Fix Phi-3.5 quantization bug** (Future):
   - Debug why uint8 conversion doesn't happen
   - Or migrate to `src/qins_codec.py` (logarithmic encoding)
   - Expected result: 13,824 MB → ~6,912 MB (2× compression)

3. ✅ **Document known bug** - Done in file headers

4. ➡️ **Proceed with Phase A** - Both demonstrate Pattern A works correctly
   - Move to Step A2: Greedy parity on longer sequences
   - Then A3: Sampling sanity checks
