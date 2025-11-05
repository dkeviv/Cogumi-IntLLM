# DO NOT USE - Archive of Incorrect Implementations

This folder contains old files that used incorrect methods or produced wrong results.

**Date Archived**: November 2, 2025  
**Reason**: Cleanup after identifying correct Pattern A implementation

---

## Why These Files Are Here

During development, we created multiple implementations and discovered that:

1. **Wrong encoding method**: Some files used rational/projective encoding instead of logarithmic
2. **No quantization**: Some implementations stored as float32 instead of uint8
3. **Calculation bugs**: Some benchmark calculations had bugs (e.g., 34× compression claim)
4. **Wrong patterns**: Some files tried to compute in QINS domain (Pattern B/C) instead of codec-at-rest (Pattern A)

---

## Folder Structure

### `old_implementations/`
**Files with wrong QINS encoding or no quantization**

- `qins_weight_codec_RATIONAL_WRONG.py` - Uses rational encoding `z = sign(x) / (1 + α|x|)` instead of logarithmic
- `calibrated_qins.py` - Tried to scale QINS activations (Pattern B, failed with 0% match)
- `projective_layer.py` - Old projective encoding (not logarithmic)
- `converter.py` - Converter for old implementations
- `compression.py` - Compression utilities for old format
- `model_loader.py` - Loader for old compressed format

### `old_tests/`
**Test files using wrong implementations**

- `test_pattern_a_clean_USES_RATIONAL.py` - Phi-3.5 test using rational encoding (0% compression)
- `test_calibrated_greedy.py` - Calibration approach (failed, 0% match)
- `test_divergence_analysis.py` - Analysis of divergence in wrong methods
- Many other test files that used incorrect implementations

### `wrong_benchmarks/`
**Benchmark results from wrong implementations**

- `test_pattern_a_clean.log` - Shows 0% compression (quantization not enabled)
- Various benchmark logs from wrong implementations
- Images/plots from failed tests

### `old_docs/`
**Documentation about failed approaches**

- `CALIBRATION_FAILURE_ANALYSIS.md` - Why calibration approach failed
- Various status reports and bug fix logs from wrong implementations

---

## What IS Correct (Still in Main Directory)

### ✅ `src/qins_codec.py`
**The GOLDEN implementation**
- Logarithmic encoding: `stored = 255 - normalize(log(|x|))`
- Quantization: uint8 + int8 storage (2 bytes per weight)
- Pattern A: Codec-at-rest (decode before compute)

### ✅ `test_codec_greedy.py`
**Golden test using correct implementation**
- Uses `src/qins_codec.py`
- 100% token match over 500 steps
- 1.48× compression on toy model

### ✅ `verify_compression_rigor.py`
**Rigorous compression verification**
- Analyzes saved models
- No implementation dependencies
- Clean analysis tool

---

## Key Lessons Learned

1. **Pattern A works**: Codec-at-rest with decode-before-compute gives 100% accuracy
2. **Logarithmic encoding is better**: More natural than rational/projective encoding
3. **Quantization must be enabled**: uint8 storage is essential for compression
4. **Never compute in QINS domain**: All computation must be in FP32 (Pattern A)

---

## DO NOT USE These Files

These files are kept only for historical reference. They contain:
- ❌ Wrong encoding methods (rational instead of logarithmic)
- ❌ Disabled quantization (float32 storage)
- ❌ Wrong patterns (compute in QINS domain)
- ❌ Calculation bugs (34× compression claim)

**Use the files in the main directory instead.**
