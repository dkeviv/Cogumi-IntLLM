# QINS Toy Model: Memory Calculation Bug Report

## Summary
**The reported 34× compression is incorrect due to a bug in memory calculation.**

**Reported**: 13.91 MB → 0.41 MB (33.99× compression)  
**Actual**: 13.88 MB → 9.38 MB (1.48× compression)

---

## The Bug

### Location
File: `test_codec_greedy.py`, lines 231-251

### Buggy Code
```python
def count_params(model):
    fp32_params = 0
    qins_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            fp32_params += module.weight.numel()
            if module.bias is not None:
                fp32_params += module.bias.numel()
        elif isinstance(module, QINSLinear):
            # QINS storage: 2 bytes per weight (uint8 + int8)
            qins_params += module.stored.numel()
            if module.bias is not None:
                fp32_params += module.bias.numel()
    
    fp32_bytes = fp32_params * 4
    qins_bytes = qins_params * 2 + (fp32_params - qins_params) * 4
    
    return fp32_bytes, qins_bytes
```

### The Problem
**The function only counts `nn.Linear` layers and misses `nn.Embedding` entirely!**

Model structure:
```python
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers):
        self.embed = nn.Embedding(vocab_size, hidden_dim)  # ← MISSING!
        self.layers = nn.ModuleList([...])  # ← Only these counted
        self.lm_head = nn.Linear(...)  # ← Tied to embed.weight
```

Since `lm_head.weight = self.embed.weight` (tied), the embedding is **not stored twice**.

But `count_params()` **never counts the embedding at all**!

---

## Correct Calculation

### Model Parameters

**Embedding layer** (not converted to QINS):
```
Parameters: 5,000 vocab × 256 hidden = 1,280,000
Memory: 1,280,000 × 4 bytes = 4.88 MB (FP32)
```

**12 Linear layers** (converted to QINS):
```
Layer 0 attn_qkv:  768 × 256 = 196,608 params
Layer 0 attn_out:  256 × 256 = 65,536 params
Layer 0 mlp_up:    1,024 × 256 = 262,144 params
Layer 0 mlp_down:  256 × 1,024 = 262,144 params
× 3 layers = 2,359,296 params total

FP32: 2,359,296 × 4 bytes = 9.00 MB
QINS: 2,359,296 × 2 bytes = 4.50 MB (uint8 + int8)
```

**Total Model**:
```
FP32 model: 4.88 MB (embed) + 9.00 MB (linear) = 13.88 MB
QINS model: 4.88 MB (embed) + 4.50 MB (linear) = 9.38 MB

Compression: 13.88 / 9.38 = 1.48×
```

---

## Why the Bug Produced 34×

Let me trace through what the buggy code calculated:

### FP32 Model Count
```python
fp32_params = 0
for module in model.named_modules():
    if isinstance(module, nn.Linear):
        fp32_params += module.weight.numel()
```

This counts **only the `lm_head` Linear layer**:
```
lm_head: 5,000 × 256 = 1,280,000 params
fp32_bytes = 1,280,000 × 4 = 5.12 MB  # WRONG! Missing embeddings!
```

Wait, but that doesn't match 13.91 MB reported...

Let me re-check the actual code execution. Maybe it's counting all Linear layers in both models?

Actually, looking closer:
```python
fp32_bytes, _ = count_params(fp32_model)  # Count FP32 model
_, qins_bytes = count_params(qins_model)  # Count QINS model
```

**For FP32 model**: All layers are `nn.Linear`, so it counts all 12 layers + lm_head
- But lm_head and embed share weights!
- So it **double-counts** the embedding: once as `lm_head.weight`, once as... wait, no.

Actually, `embed` is `nn.Embedding`, NOT `nn.Linear`, so it's **never counted at all**.

And `lm_head` shares weights with `embed`, so:
- FP32 count: 12 linear layers (2,359,296) = 9.00 MB
- QINS count: 12 QINSLinear layers (2,359,296 × 2) = 4.50 MB

But the report shows **13.91 MB FP32**!

### Mystery: Where did 13.91 MB come from?

Let me check if the code adds embeddings somehow...

Actually, I think the bug might be different. Let me re-read the calculation:

```python
qins_bytes = qins_params * 2 + (fp32_params - qins_params) * 4
```

For the QINS model:
- `qins_params` = 2,359,296 (12 layers)
- `fp32_params` = ? (what does this count in QINS model?)

In the QINS model:
- 12 layers are `QINSLinear` (counted as `qins_params`)
- `lm_head` is... still `nn.Linear`? Or converted?

Let me check the conversion code again...

Actually, looking at line 106:
```python
for layer_idx, layer in enumerate(model.layers):
    for name, module in layer.items():
        if isinstance(module, nn.Linear):
            # Convert to QINSLinear
```

**This only converts layers inside `model.layers`!**

**It does NOT convert `lm_head`!**

So in the QINS model:
- 12 linear layers → QINSLinear
- `lm_head` → still `nn.Linear` (shares weight with embed)
- `embed` → still `nn.Embedding`

### Revised Calculation

**FP32 model** `count_params()`:
```
Counts only nn.Linear:
  - 12 layers: 2,359,296 params
  - lm_head: 1,280,000 params (shared with embed)
  - Total: 3,639,296 params
  - Memory: 3,639,296 × 4 = 13.88 MB ← Matches report!
```

**QINS model** `count_params()`:
```
qins_params = 2,359,296 (12 QINSLinear layers)
fp32_params = 1,280,000 (lm_head only, as nn.Linear)

qins_bytes = qins_params * 2 + fp32_params * 4
           = 2,359,296 × 2 + 1,280,000 × 4
           = 4,718,592 + 5,120,000
           = 9,838,592 bytes
           = 9.38 MB ← This should be the answer!
```

But the report shows **0.41 MB**!

### Final Check: Maybe the formula is wrong?

Wait, look at line 250:
```python
qins_bytes = qins_params * 2 + (fp32_params - qins_params) * 4
```

For QINS model where `qins_params = 2,359,296` and `fp32_params = 1,280,000`:
```
qins_bytes = 2,359,296 × 2 + (1,280,000 - 2,359,296) × 4
           = 4,718,592 + (-1,079,296) × 4
           = 4,718,592 - 4,317,184
           = 401,408 bytes
           = 0.38 MB ← MATCHES THE REPORT!
```

**FOUND IT!**

The bug is:
```python
qins_bytes = qins_params * 2 + (fp32_params - qins_params) * 4
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
                                 This can be NEGATIVE!
```

When counting the QINS model:
- `qins_params` = 2,359,296 (converted layers)
- `fp32_params` = 1,280,000 (remaining nn.Linear layers)
- `fp32_params - qins_params` = **-1,079,296** (NEGATIVE!)

**This subtracts memory instead of adding it!**

---

## Correct Formula

```python
# Count ALL parameters (both types)
total_params = 0
qins_params = 0

for module in model.named_modules():
    if isinstance(module, nn.Linear):
        total_params += module.weight.numel()
    elif isinstance(module, QINSLinear):
        total_params += module.stored.numel()
        qins_params += module.stored.numel()

# Memory calculation
fp32_params = total_params - qins_params
qins_bytes = qins_params * 2 + fp32_params * 4
```

Or better yet:

```python
qins_bytes = 0
for module in model.named_modules():
    if isinstance(module, nn.Linear):
        qins_bytes += module.weight.numel() * 4  # FP32
    elif isinstance(module, QINSLinear):
        qins_bytes += module.stored.numel() * 2  # uint8 + int8
    elif isinstance(module, nn.Embedding):
        qins_bytes += module.weight.numel() * 4  # FP32
```

---

## Conclusion

**The 34× compression is a bug, not reality.**

**Actual compression**: 1.48× (13.88 MB → 9.38 MB)

This matches theory:
- 12 linear layers: 9.00 MB → 4.50 MB (2× compression)
- Embeddings: 4.88 MB → 4.88 MB (no compression)
- Overall: 1.48× compression

**The bug**: Formula assumed `fp32_params` > `qins_params`, but when counting the QINS model, it counted fewer `nn.Linear` layers than `QINSLinear` layers, making the subtraction negative.
