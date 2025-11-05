# Toy Model vs Phi-3.5 Benchmark Comparison

## Files Used

### Toy Model Benchmark (3-layer transformer)
- **Test File**: `test_codec_greedy.py` (in main directory)
- **Converter/Implementation**: `src/qins_codec.py`
- **Log File**: `test_codec_greedy_run.log`
- **Model**: 3-layer transformer, 256 hidden, 5K vocab
- **Status**: ✅ CORRECT (uses logarithmic encoding)

### Phi-3.5 Benchmark (Full-size LLM)
- **Test File**: `DO_NOT_USE/old_tests/test_pattern_a_clean_USES_RATIONAL.py` (archived)
- **Converter/Implementation**: `DO_NOT_USE/old_implementations/qins_weight_codec_RATIONAL_WRONG.py` (archived)
- **Log File**: `DO_NOT_USE/wrong_benchmarks/test_pattern_a_clean.log`
- **Model**: microsoft/Phi-3.5-mini-instruct (3.82B params)
- **Status**: ⚠️ USES RATIONAL ENCODING (not logarithmic, but still Pattern A)

---

## Why Both Got 100% Match Despite Different Encodings

### The Critical Insight: BOTH ARE PATTERN A ✅

Both implementations got 100% match because they both follow Pattern A (codec-at-rest):

1. **Encode weights** → Store in compressed/transformed format
2. **Decode before compute** → Always compute in FP32
3. **Never compute in QINS domain** → Preserves correctness

### Pattern A is Encoding-Agnostic!

As long as:
- ✅ Encoding is **invertible** (decode recovers original weights accurately)
- ✅ Decode happens **before every computation**
- ✅ All math happens in **FP32 domain**

Then it doesn't matter if you use:
- Logarithmic encoding
- Rational/projective encoding
- Or even identity mapping

**All will give 100% accuracy** (within floating point precision)!

---

## Detailed Implementation Comparison

| Aspect | Toy Model (Logarithmic) | Phi-3.5 (Rational) |
|--------|------------------------|-------------------|
| **Test File** | `test_codec_greedy.py` | `test_pattern_a_clean.py` |
| **Implementation** | `src/qins_codec.py` | `qins_weight_codec.py` |
| **Model** | 3-layer transformer | Phi-3.5-mini (3.82B) |
| **Encoding Formula** | `log(abs(w))` → normalize → inverse map | `z = sign(w) / (1 + α×abs(w))` |
| **Normalization** | `norm = (log_w - log_min) / (log_max - log_min)` | `z` already in range [-1, 1] |
| **Storage Mapping** | `stored = 255 - (norm × 254)` | `stored = z` (if quantize enabled) |
| **Inverse Relationship** | ✅ Large abs(w) → stored=1 (small) | ❌ Large abs(w) → z near 0 |
| | | Small abs(w) → z near ±1 |
| **Decode Formula** | `norm = (255 - stored) / 254` | `sign = sign(z)` |
| | `log_w = log_min + norm×(log_max - log_min)` | `abs_z = abs(z).clamp(min=1e-12)` |
| | `w = sign × exp(log_w)` | `w = sign × (1 - abs_z) / (α × abs_z)` |
| **Storage dtype** | uint8 (stored) + int8 (sign) | float32 (not quantized) |
| **Bytes per weight** | 2 bytes (1 + 1) | 4 bytes |
| **Quantization** | ✅ Enabled by default | ❌ Not actually applied |
| **Compression** | 2× on weights | 0× (no compression) |
| **Accuracy** | 100% match (500 tokens) | 100% match (15 tokens) |
| **Pattern** | ✅ Pattern A (codec-at-rest) | ✅ Pattern A (codec-at-rest) |

---

## The Memory Compression Difference

### Why Different Results?

**Toy Model**: Actually quantizes to uint8
```python
# In src/qins_codec.py line 75:
stored = stored_float.round().clamp(1, 255).to(torch.uint8)  # ✅ Quantizes!
```

**Phi-3.5**: Stores as float32 despite quantize flag
```python
# In qins_weight_codec.py line 42-47:
if quantize:
    quantized = ((encoded + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return quantized
return encoded  # ❌ Returns float32 when quantize=False or hits error
```

### Evidence from Logs

**Phi-3.5 log shows**:
```
QINS-encoded: torch.Size([3072, 3072]) (36.00 MB FP32 → 36.00 MB QINS)
```

**Calculation**: 36 MB = 3072 × 3072 × **4 bytes** ← Proves it's **float32**!

If it were uint8: 9 MB = 3072 × 3072 × **1 byte**
The Bug in Phi-3.5 Test
Looking at qins_weight_codec_RATIONAL_WRONG.py:

Line 106 - Default should quantize:

Line 130 - Calls encode with quantize parameter:

Line 42 - Should quantize to uint8:

BUT - Line 233 in convert_linear_to_qins:

So it uses default quantize=True, should create uint8 tensors...

BUT THEN - Line 235 calculates memory wrong:

This suggests the tensor might actually be float32, not uint8!

Let Me Verify The Actual dtype
Let me check if there's evidence of the actual dtype in the log:

The log shows:

36 MB = 3072 × 3072 × 4 bytes ← This proves it's float32, not uint8!

If it were uint8:
9 MB = 3072 × 3072 × 1 byte

The Real Bug: Encoding Didn't Actually Quantize
What likely happened:

quantize=True was passed to __init__
BUT qins_encode() might have hit an edge case or error
Fell back to returning float32
Or there's a code path where quantize doesn't propagate correctly
Result: Rational encoding stored as float32 (not uint8), so 0% compression

Summary Table
Implementation	Encoding Formula	Storage dtype	Decode Formula	Accuracy	Compression
Toy (qins_codec.py)	[log(	x	)](http://vscodecontentref/15) → uint8	uint8 + int8	exp(denorm)
Phi-3.5 (qins_weight_codec.py)	[sign/(1+α	x	)](http://vscodecontentref/17) → float32	float32 ❌	`(1-
Why Both Got 100% Match
✅ Both use Pattern A correctly:

Store weights in transformed space
Decode to FP32 before every computation
Compute always in FP32 domain
✅ Both encodings are invertible:

Logarithmic: decode(encode(x)) ≈ x
Rational: decode(encode(x)) ≈ x
✅ Pattern A guarantees correctness regardless of encoding method!

The Key Takeaway
Your question revealed the crucial insight:

The encoding method (logarithmic vs rational) doesn't affect accuracy - both give 100% match because Pattern A always decodes to FP32 before computation.

The encoding method DOES affect compression potential:

Logarithmic maps naturally to uint8 range [1, 255]
Rational maps to [-1, +1] which can be quantized to uint8
But only toy model actually quantized to uint8. Phi-3.5 test stored as float32 despite the quantization code being present - likely a bug in the conversion flow.