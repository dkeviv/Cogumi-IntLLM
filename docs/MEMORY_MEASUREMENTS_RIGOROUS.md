# QINS Pattern A: Rigorous Memory Measurements

**Date**: November 2, 2025  
**Purpose**: Verify actual compression ratios with no approximations

---

## Test 1: Toy Model (3-Layer Transformer)

### Configuration
```
Model: 3-layer transformer
  Vocabulary: 5,000 tokens
  Hidden dimension: 256
  Layers: 3
  Total linear layers: 12 (attn_qkv, attn_out, mlp_up, mlp_down × 3)
```

### Storage Format
```python
# QINSLinear stores:
stored: torch.uint8  # 1 byte per weight [1, 255]
sign:   torch.int8   # 1 byte per weight {-1, +1}
log_min: float32     # 4 bytes per layer
log_max: float32     # 4 bytes per layer
bias:   float32      # 4 bytes per bias (if exists)
```

### Memory Calculation
```python
# From test_codec_greedy.py line 243-246:
qins_params += module.stored.numel()  # Counts uint8 elements
qins_bytes = qins_params * 2  # 2 bytes per weight (uint8 + int8)
```

### Results
```
FP32 model: 13.91 MB
QINS model: 0.41 MB
Compression: 33.99×
Greedy match: 100.00% (500/500 tokens)
```

### Analysis
**Why 34× compression?**

The toy model likely has:
1. **Small vocabulary** (5K) → small embedding layer
2. **Only 12 linear layers** encoded
3. **Bias terms stay FP32** (not encoded)
4. **Embeddings/LM head not encoded** (stay FP32)

Breaking down the 13.91 MB:
```
FP32 model components:
  - Embedding: 5000 × 256 × 4 bytes = 5.00 MB
  - 12 Linear layers: ~8.91 MB (various sizes)
  - Bias terms: ~small
  - LM head: (tied to embeddings)
  Total: 13.91 MB
```

QINS model:
```
  - Embedding: 5.00 MB (not encoded, stays FP32)
  - 12 Linear weights: 8.91 MB → 0.41 MB (2 bytes/weight)
    → 21.7× compression on linear layers only
  - Bias terms: ~small (stays FP32)
  Total: 0.41 MB (after calculation error?)
```

**Issue**: The 0.41 MB seems too small even with 2 bytes/weight. Let me recalculate:

```
Layer dimensions from log:
  attn_qkv: 768 × 256 = 196,608 params
  attn_out: 256 × 256 = 65,536 params
  mlp_up:   1024 × 256 = 262,144 params
  mlp_down: 256 × 1024 = 262,144 params
  
Per layer: 786,432 params
3 layers: 2,359,296 params

QINS storage: 2,359,296 × 2 bytes = 4.72 MB (uint8 + int8)
```

**Wait!** 4.72 MB ≠ 0.41 MB

**Possible explanations:**
1. **Calculation bug** - the `count_params()` function may be counting wrong
2. **Mixed storage** - some layers not converted?
3. **Only counting `stored` tensor** - forgot to add `sign` tensor?

Let me check the actual calculation more carefully...

---

## Test 2: Phi-3.5-mini (Production Model)

### Configuration
```
Model: Phi-3.5-mini-instruct
  Parameters: 3.82 billion
  Hidden dimension: 3072
  Layers: 32
  Converted layers: v_proj, o_proj, gate_proj, up_proj, down_proj (128 total)
```

### Results (from test_pattern_a_clean.log)
```
Weight memory:
  FP32:  13,824.00 MB
  QINS:  13,824.00 MB
  Saved: 0.00 MB (0.0%)

Greedy match: 100% (15/15 tokens)
```

### Analysis
**NO compression!** Storage is still float32, not uint8.

From the log:
```
QINS-encoded: torch.Size([3072, 3072]) (36.00 MB FP32 → 36.00 MB QINS)
```

**Proof**: The QINS storage shows same size → float32 dtype

**Reason**: Quantization is NOT enabled in `qins_weight_codec.py`

Looking at the code:
```python
# qins_weight_codec.py line 18:
def qins_encode(weight, alpha=1.0, quantize=False):
    #                               ↑ Default is False!
```

When `quantize=False`:
- Returns float32 tensor in range [-1, 1]
- Size: 4 bytes per weight (same as original FP32)
- Compression: 0×

When `quantize=True` (not used):
- Maps to uint8 [0, 255]
- Size: 1 byte per weight
- Compression: 4×

---

## Test 3: Rigorous Phi-3.5 Benchmark (Running)

### Configuration
```
Script: benchmark_phi35_memory_rigorous.py
Measures:
  - Actual tensor sizes in bytes
  - Separate weight vs buffer memory
  - QINS layer-specific compression
  - Dtype verification (uint8 vs float32)
```

### Results
```
STEP 1: FP32 Baseline
  Parameters:  14,576.26 MB
  Buffers:     0.00 MB
  Total:       14,576.26 MB
  Param count: 3,821,079,552

STEP 2: QINS Model
  [Still running...]
```

---

## Summary: The Truth About Compression

| Test | Model | Claimed Compression | Actual Compression | Reason |
|------|-------|-------------------|-------------------|---------|
| Toy model | 3-layer transformer | 34× | **2×** (likely) | Bug in memory calculation |
| Phi-3.5 clean test | Phi-3.5-mini | 0× | **0×** ✅ | quantize=False (float32 storage) |
| Phi-3.5 rigorous | Phi-3.5-mini | TBD | **TBD** | Running now |

### The Real Numbers (Expected)

**Pattern A with quantization disabled** (current):
```
Storage format: float32 QINS values (range [-1, 1])
Compression: 0× (no compression)
Memory: Same as FP32
Quality: 100% match (lossless)
```

**Pattern A with quantization enabled** (target):
```
Storage format: uint8 [0, 255]
Compression: 4× (1 byte vs 4 bytes)
Memory: 25% of FP32
Quality: >99% match (small quantization error)
```

**Pattern A with bit-packing** (Phase 2 goal):
```
Storage format: 6-bit or 8-bit packed integers
Compression: 5-8× (depends on bit width)
Memory: 12-20% of FP32
Quality: >95% match (larger quantization error)
```

---

## Action Items

### Immediate (Fix Memory Calculation)
1. ✅ Run rigorous benchmark on Phi-3.5
2. ⏳ Verify actual dtype of stored weights
3. ⏳ Fix toy model memory calculation bug
4. ⏳ Re-run toy model with corrected calculation

### Phase 2 (Enable Compression)
1. ⏳ Enable `quantize=True` in qins_weight_codec.py
2. ⏳ Verify uint8 storage (dtype check)
3. ⏳ Measure actual 4× compression
4. ⏳ Validate quality (should still be >99%)

### Phase 3 (Optimize Further)
1. ⏳ Implement bit-packing (6-bit or 8-bit)
2. ⏳ Fused decode kernel
3. ⏳ KV cache compression

---

## Conclusion

**Current Reality**:
- ✅ Pattern A codec: **VALIDATED** (100% token match)
- ❌ Compression: **NOT ENABLED** (0% savings on Phi-3.5)
- ⚠️  34× claim: **SUSPICIOUS** (likely calculation bug)

**Expected with quantization**:
- 4× compression from uint8 storage
- >99% quality preservation
- Minimal decode overhead

**The 34× claim is almost certainly a measurement error, not actual compression.**
