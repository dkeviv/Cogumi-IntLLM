# Why Toy Model Has Memory Reduction but Phi-3.5 Doesn't

## TL;DR

**YES, the toy model DOES use quantization** (uint8 + int8 storage).  
**NO, Phi-3.5 does NOT use quantization** (stores float32).

This is why:
- **Toy model**: 2× compression (13.88 MB → 9.38 MB calculated, 0.41 MB reported is a bug)
- **Phi-3.5**: 0× compression (13,824 MB → 13,824 MB)

---

## The Two Different Implementations

### Implementation 1: Toy Model (`src/qins_codec.py`)

**File**: `src/qins_codec.py`  
**Class**: `QINSLinear`  
**Storage Format**: ✅ **QUANTIZED** (uint8 + int8)

```python
class QINSLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        # QINS storage - QUANTIZED!
        self.register_buffer('stored', torch.zeros(..., dtype=torch.uint8))  # 1 byte
        self.register_buffer('sign', torch.zeros(..., dtype=torch.int8))      # 1 byte
        self.register_buffer('log_min', torch.tensor(0.0))
        self.register_buffer('log_max', torch.tensor(0.0))
```

**Verification**:
```
📦 Stored tensor:
  dtype: torch.uint8           ← QUANTIZED!
  element_size: 1 bytes        ← 1 byte per weight
  min/max: [1, 255]

📦 Sign tensor:
  dtype: torch.int8            ← QUANTIZED!
  element_size: 1 bytes        ← 1 byte per weight
  unique values: [-1, 1]

💾 Memory:
  Original FP32: 262,144 bytes
  QINS storage: 131,072 bytes (65,536 stored + 65,536 sign)
  Compression: 2.00×           ← Real compression!
```

**Encoding method**: Logarithmic encoding
```python
# In QINSCodec.encode():
log_tensor = torch.log(abs_tensor)
normalized = (log_tensor - log_min) / (log_max - log_min)
stored_float = 255.0 - (normalized * 254.0)
stored = stored_float.round().clamp(1, 255).to(torch.uint8)  # ← Quantize to uint8!
```

---

### Implementation 2: Phi-3.5 (`qins_weight_codec.py`)

**File**: `qins_weight_codec.py`  
**Class**: `QINSWeightLinear`  
**Storage Format**: ❌ **NOT QUANTIZED** (float32)

```python
class QINSWeightLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        # QINS storage - NOT quantized!
        self.weight_qins = nn.Parameter(torch.Tensor(out_features, in_features))  # float32!
```

**Why no compression?** Look at the encode function:

```python
def qins_encode(weight: torch.Tensor, alpha: float = 1.0, quantize: bool = False):
    #                                                          ↑
    #                                           Default is FALSE!
    sign = torch.sign(weight)
    abs_weight = weight.abs()
    encoded = sign / (1.0 + alpha * abs_weight)  # Returns float32 [-1, 1]
    
    if quantize:  # ← This branch is NEVER taken by default
        quantized = ((encoded + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
        return quantized
    
    return encoded  # ← Returns float32!
```

**Evidence from logs**:
```
QINS-encoded: torch.Size([3072, 3072]) (36.00 MB FP32 → 36.00 MB QINS)
                                         ^^^^^^^^      ^^^^^^^^
                                         Same size = no compression!
```

For a [3072, 3072] weight matrix:
- FP32: 3072 × 3072 × 4 bytes = 37,748,736 bytes = 36.00 MB
- QINS: 3072 × 3072 × 4 bytes = 37,748,736 bytes = 36.00 MB ← Still float32!

---

## Side-by-Side Comparison

| Feature | Toy Model (`QINSLinear`) | Phi-3.5 (`QINSWeightLinear`) |
|---------|-------------------------|------------------------------|
| **Storage dtype** | uint8 (stored) + int8 (sign) | float32 |
| **Bytes per weight** | 2 bytes | 4 bytes |
| **Encoding method** | Logarithmic (log space) | Rational (projective) |
| **Quantization** | ✅ Enabled by default | ❌ Disabled by default |
| **Compression** | 2× actual | 0× (none) |
| **Memory formula** | stored (1) + sign (1) = 2 bytes | encoded (4) = 4 bytes |

---

## Why This Happened

You implemented **two different Pattern A variants**:

1. **Early toy model test** (`src/qins_codec.py`):
   - Used logarithmic encoding with quantization
   - Stored as uint8 + int8 (2 bytes total)
   - Got 2× compression (modulo the calculation bug)

2. **Later Phi-3.5 test** (`qins_weight_codec.py`):
   - Used rational (projective) encoding without quantization
   - Stored as float32 (4 bytes)
   - Got 0× compression

The Phi-3.5 implementation has quantization **available** but **not enabled** by default:
```python
def qins_encode(weight, alpha=1.0, quantize=False):  # ← quantize=False
```

---

## What You Actually Did (Without Realizing)

### Toy Model Test
```python
# In test_codec_greedy.py:
qins_module = QINSLinear.from_linear(module)  # ← Uses qins_codec.py
```

**This uses `QINSCodec.encode()`** which:
```python
stored = stored_float.round().clamp(1, 255).to(torch.uint8)  # ← Always quantizes!
```

### Phi-3.5 Test
```python
# In test_pattern_a_clean.py:
layer.v_proj = QINSWeightLinear.from_linear(layer.v_proj)  # ← Uses qins_weight_codec.py
```

**This uses `qins_encode()`** which:
```python
def qins_encode(weight, alpha=1.0, quantize=False):  # ← quantize defaults to False
    encoded = sign / (1.0 + alpha * abs_weight)      # ← Returns float32
    return encoded  # ← No quantization!
```

---

## The Memory Difference Explained

### Toy Model (with quantization)
```
Linear layers: 2,359,296 params
  FP32: 2,359,296 × 4 = 9.00 MB
  QINS: 2,359,296 × 2 = 4.50 MB  ← uint8 + int8 storage

Embeddings: 1,280,000 params (not encoded)
  FP32: 1,280,000 × 4 = 4.88 MB
  QINS: 1,280,000 × 4 = 4.88 MB  ← Still FP32

Total:
  FP32: 13.88 MB
  QINS: 9.38 MB
  Compression: 1.48×
```

### Phi-3.5 (without quantization)
```
Converted layers: 128 layers (v_proj, o_proj, gate_proj, up_proj, down_proj)
  FP32: 3,456,000,000 × 4 = 13,824 MB
  QINS: 3,456,000,000 × 4 = 13,824 MB  ← Still float32!

Embeddings & other layers: ~320,000,000 params
  FP32: ~1,280 MB
  QINS: ~1,280 MB

Total:
  FP32: ~15,104 MB
  QINS: ~15,104 MB
  Compression: 1.00× (none)
```

---

## How to Enable Compression in Phi-3.5

**Option 1**: Change default in `qins_weight_codec.py`:
```python
def qins_encode(weight: torch.Tensor, alpha: float = 1.0, quantize: bool = True):
    #                                                          ^^^^^ Change to True
```

**Option 2**: Pass quantize=True when encoding:
```python
class QINSWeightLinear(nn.Module):
    @classmethod
    def from_linear(cls, linear: nn.Linear, alpha: float = 1.0):
        # Encode weights
        weight_qins = qins_encode(linear.weight.data, alpha=alpha, quantize=True)  # ← Add this
```

**Expected result**:
```
Weight memory: FP32: 13,824.00 MB, QINS: 3,456.00 MB, Saved: 10,368.00 MB (75.0%)
Compression: 4×
```

---

## Summary

**You DID use quantization in the toy model** - that's why it shows 2× compression on linear layers.

**You did NOT use quantization in Phi-3.5** - that's why it shows 0× compression.

The difference is:
- **Toy model**: `QINSLinear` (qins_codec.py) - always quantizes to uint8
- **Phi-3.5**: `QINSWeightLinear` (qins_weight_codec.py) - quantization disabled by default

This is an implementation inconsistency between the two Pattern A variants you created.
