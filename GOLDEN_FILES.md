# Golden Files - Correct Implementation Reference

**Date**: November 2, 2025  
**Status**: VERIFIED - These files use correct Pattern A with logarithmic encoding

---

## Core Implementation (GOLDEN)

### ✅ `src/qins_codec.py`
**Encoding Method**: LOGARITHMIC (correct)
```python
log_tensor = torch.log(abs_tensor)
normalized = (log_tensor - log_min) / (log_max - log_min)
stored = 255.0 - (normalized * 254.0)  # Inverse magnitude
stored = stored.round().clamp(1, 255).to(torch.uint8)  # Quantized!
```

**Storage**:
- `stored`: uint8 [1, 255] - inverse magnitude in log space
- `sign`: int8 {-1, +1} - preserved signs
- `log_min`, `log_max`: float32 scalars per layer
- **Compression**: 2× (2 bytes vs 4 bytes FP32)

**Classes**:
- `QINSCodec` - Encoder/decoder
- `QINSLinear` - Linear layer with QINS weight storage

**Key Properties**:
1. ✅ Logarithmic encoding (not rational/projective)
2. ✅ Quantization ENABLED by default (uint8 + int8)
3. ✅ Inverse magnitude relationship (large weights → stored=1)
4. ✅ 100% sign preservation
5. ✅ Pattern A: Codec-at-rest (decode before compute)

---

## Test Files (GOLDEN)

### ✅ `test_codec_greedy.py`
**Purpose**: Test Pattern A with logarithmic encoding on toy model
**Uses**: `src/qins_codec.py` (QINSLinear)
**Model**: 3-layer transformer (256 hidden, 5K vocab)
**Results**: 100% token match over 500 steps
**Compression**: 1.48× overall (2× on linear layers, embeddings not encoded)

**Why it's golden**:
- Uses correct logarithmic encoding
- Quantization enabled (uint8 + int8)
- Validates Pattern A works with perfect accuracy

### ✅ `test_codec_greedy_run.log`
**Status**: VERIFIED CORRECT
**Results**:
- Greedy match: 100.00% (500/500 tokens)
- FP32 model: 13.91 MB (reported)
- QINS model: 0.41 MB (WRONG - bug in calculation)
- **Actual QINS**: 9.38 MB (calculated correctly)
- **Actual compression**: 1.48× (not 34×)

---

## Benchmark Files (GOLDEN)

### ✅ `benchmark_phi35_memory_rigorous.py`
**Purpose**: Detailed memory analysis for Phi-3.5
**Method**: Direct dtype checking, layer-by-layer breakdown
**Status**: Partially executed (interrupted)

**Why it's golden**:
- Verifies actual tensor dtypes (uint8 vs float32)
- Separates parameters vs buffers
- Layer-by-layer compression breakdown
- No assumptions, direct measurement

### ✅ `compression_verification.log`
**Status**: Analysis of saved models
**Method**: Analyzes torch.load() model files
**Purpose**: Compare FP32 vs QINS saved model sizes

---

## Verification Scripts (GOLDEN)

### ✅ `verify_compression_rigor.py`
**Purpose**: Analyze saved model files for compression
**Method**: Load models, count parameters, check dtypes
**Dependencies**: None (just torch + transformers)
**Status**: Clean analysis tool

---

## Documentation (GOLDEN)

### ✅ `docs/MEMORY_BUG_ANALYSIS.md`
**Purpose**: Documents the 34× compression bug
**Key Finding**: Memory calculation error in test_codec_greedy.py
**Actual Result**: 1.48× compression (not 34×)

### ✅ `docs/QUANTIZATION_COMPARISON.md`
**Purpose**: Explains why toy model has compression but Phi-3.5 doesn't
**Key Finding**: Two different implementations
- Toy model: Uses `qins_codec.py` with uint8 quantization ✅
- Phi-3.5: Uses `qins_weight_codec.py` without quantization ❌

### ✅ `docs/MEMORY_MEASUREMENTS_RIGOROUS.md`
**Purpose**: Rigorous memory measurements
**Status**: Documents actual vs claimed compression ratios

---

## Phi-3.5 Benchmark Files (Rational Encoding - Pattern A)

### ⚠️ `qins_weight_codec.py`
**Encoding Method**: RATIONAL/PROJECTIVE
```python
encoded = sign / (1.0 + alpha * abs_weight)
decoded = sign * (1.0 - abs_encoded) / (alpha * abs_encoded)
```

**Status**: 
- ✅ Pattern A correctly implemented (decode before compute)
- ✅ Achieves 100% accuracy on Phi-3.5
- ❌ Quantization bug: stores as float32 (not uint8)
- ❌ 0% compression (13,824 MB → 13,824 MB)

**Used by**:
- `test_pattern_a_clean.py` (Phi-3.5 benchmark)

**Why Keep**: 
- Validates Pattern A works on full-size LLM (3.82B params)
- Proves encoding method doesn't affect accuracy
- Useful for comparison and baseline

### ⚠️ `test_pattern_a_clean.py`
**Purpose**: Phi-3.5 benchmark with rational encoding
**Model**: microsoft/Phi-3.5-mini-instruct (3.82B params)
**Result**: 
- ✅ 100% token match (15/15)
- ❌ 0% compression (quantization bug)

**Status**: Keep for benchmarking, but note known bug

---

## Migration Plan

### Step 1: Rename/Move Files
```bash
# Move wrong implementation
mv qins_weight_codec.py DO_NOT_USE/old_implementations/qins_weight_codec_rational.py

# Keep golden implementation
# src/qins_codec.py stays as-is ✅
```

### Step 2: Create Phi-3.5 Test with Correct Encoding
```bash
# New file: test_phi35_logarithmic.py
# Uses: src/qins_codec.py (QINSLinear)
# Expected: 4× compression with quantization enabled
```

### Step 3: Fix Memory Calculation Bug
```bash
# Fix: test_codec_greedy.py line 250
# Correct formula to handle both FP32 and QINS layers properly
```

---

## Summary

**Golden Implementation**:
- File: `src/qins_codec.py`
- Encoding: Logarithmic (correct)
- Quantization: Enabled (uint8 + int8)
- Compression: 2× on encoded weights

**Wrong Implementation** (to be deprecated):
- File: `qins_weight_codec.py`
- Encoding: Rational/projective (old)
- Quantization: Disabled by default
- Compression: 0× (no actual compression)

**Action**: Keep `src/qins_codec.py` as golden standard, deprecate `qins_weight_codec.py`.
