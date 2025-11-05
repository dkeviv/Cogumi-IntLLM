# IMMEDIATE NEXT STEPS - Week 1 Checklist

**Phase 1: Pattern A Production-Ready (KV Cache Compression)**

---

## 🎯 This Week's Goals

1. ✅ Implement bit-packed KV cache storage
2. ✅ Validate compression works (>99% match)
3. ✅ Measure memory savings (target: 2-4×)
4. ✅ Integrate into test_pattern_a_clean.py

---

## 📝 Task List

### Day 1-2: Core Implementation

**File: `qins_kv_cache.py`**

```python
class BitPackedKVCache:
    """
    Bit-packed KV cache storage.
    
    Strategy:
    - Quantize K/V to 6-bit or 8-bit
    - Pack into pages (16-64 KB)
    - Decompress on-the-fly during attention
    
    Memory savings: 2-4× vs FP32
    Latency overhead: Target <5%
    """
    
    def __init__(self, num_bits=6, page_size_kb=32):
        self.num_bits = num_bits
        self.page_size = page_size_kb * 1024
        
    def compress_kv(self, k, v):
        """Quantize and bit-pack K, V tensors."""
        # TODO: Implement
        pass
    
    def decompress_kv(self, k_packed, v_packed):
        """Unpack and dequantize K, V tensors."""
        # TODO: Implement
        pass
```

**File: `test_kv_compression.py`**

```python
def test_kv_roundtrip():
    """Test KV compression -> decompression accuracy."""
    
    # Generate fake KV cache
    k = torch.randn(1, 32, 128, 96)  # [batch, heads, seq, dim]
    v = torch.randn(1, 32, 128, 96)
    
    # Compress
    cache = BitPackedKVCache(num_bits=6)
    k_packed, v_packed = cache.compress_kv(k, v)
    
    # Decompress
    k_restored, v_restored = cache.decompress_kv(k_packed, v_packed)
    
    # Measure error
    k_error = (k - k_restored).abs().mean() / k.abs().mean()
    v_error = (v - v_restored).abs().mean() / v.abs().mean()
    
    print(f"K relative error: {k_error:.4%}")
    print(f"V relative error: {v_error:.4%}")
    
    assert k_error < 0.01  # <1% error
    assert v_error < 0.01
```

### Day 3-4: Integration & Testing

**Modify: `test_pattern_a_clean.py`**

Add KV cache compression option:

```python
# After converting weights to QINS
print("\n" + "="*60)
print("TESTING KV CACHE COMPRESSION")
print("="*60)

# Create bit-packed cache
kv_cache = BitPackedKVCache(num_bits=6, page_size_kb=32)

# Generate with compressed KV cache
outputs_qins_kv = model.generate(
    **inputs,
    max_new_tokens=num_tokens,
    do_sample=False,
    use_cache=True,
    kv_cache_compress=kv_cache  # NEW PARAMETER
)

# Compare tokens
tokens_baseline = outputs_baseline[0].tolist()
tokens_qins_kv = outputs_qins_kv[0].tolist()

match_rate = sum(a == b for a, b in zip(tokens_baseline, tokens_qins_kv)) / len(tokens_baseline)
print(f"Match rate with KV compression: {match_rate:.1%}")
```

### Day 5: Benchmarking

**File: `benchmark_kv_memory.py`**

```python
def benchmark_kv_memory():
    """
    Compare memory usage:
    1. FP32 baseline (no compression)
    2. QINS weights + FP32 KV
    3. QINS weights + compressed KV (6-bit)
    """
    
    # Load model
    model = load_phi35()
    
    # Generate long context (512 tokens)
    prompt = "..." * 100  # Long prompt
    
    # Test 1: FP32 baseline
    mem_baseline = measure_memory(
        lambda: model.generate(prompt, max_new_tokens=512)
    )
    
    # Test 2: QINS weights, FP32 KV
    model_qins = convert_to_qins(model)
    mem_qins = measure_memory(
        lambda: model_qins.generate(prompt, max_new_tokens=512)
    )
    
    # Test 3: QINS weights + 6-bit KV
    kv_cache = BitPackedKVCache(num_bits=6)
    mem_qins_kv = measure_memory(
        lambda: model_qins.generate(
            prompt, 
            max_new_tokens=512,
            kv_cache_compress=kv_cache
        )
    )
    
    print("\n" + "="*60)
    print("MEMORY USAGE COMPARISON")
    print("="*60)
    print(f"FP32 baseline:        {mem_baseline:.2f} GB")
    print(f"QINS weights only:    {mem_qins:.2f} GB ({mem_baseline/mem_qins:.1f}×)")
    print(f"QINS + 6-bit KV:      {mem_qins_kv:.2f} GB ({mem_baseline/mem_qins_kv:.1f}×)")
    print(f"\nKV cache savings: {(mem_qins - mem_qins_kv)/mem_qins:.1%}")
```

---

## 📊 Expected Results

### Compression Ratios

| Configuration | Memory (GB) | vs Baseline | vs QINS-only |
|---------------|-------------|-------------|--------------|
| FP32 baseline | ~8.0 | 1.0× | - |
| QINS weights | ~2.0 | 4.0× | 1.0× |
| QINS + 6-bit KV | ~1.5 | 5.3× | 1.33× |
| QINS + 8-bit KV | ~1.7 | 4.7× | 1.18× |

### Quality Metrics

| Metric | Target | Acceptable Range |
|--------|--------|------------------|
| Token match rate | >99% | 95-100% |
| Perplexity increase | <2% | 0-5% |
| Relative error (K) | <1% | <2% |
| Relative error (V) | <1% | <2% |

### Performance

| Operation | Time (ms) | vs Baseline |
|-----------|-----------|-------------|
| KV compression | <1 | - |
| KV decompression | <5 | +3-5% |
| Full generation | ~100 | +2-3% |

---

## ✅ Success Criteria

**Must achieve to pass to Phase 2:**

- ✅ KV cache compresses to 6-bit or 8-bit
- ✅ Token match rate >99% on 100-token generation
- ✅ Memory savings >25% vs QINS-weights-only
- ✅ Latency overhead <5%
- ✅ No numerical instability (no NaN/Inf)
- ✅ Code is clean, documented, tested

---

## 🚫 What NOT to Do This Week

- ❌ **Don't implement Pattern B yet** (wait for Phase 2)
- ❌ **Don't add entropy coding** (Phase 3 will decide based on data)
- ❌ **Don't optimize for speed yet** (Phase 4 is for kernels)
- ❌ **Don't compress weights differently** (Pattern A is proven)

**Focus**: Simple, working KV compression. That's it.

---

## 🔧 Development Workflow

### Setup
```bash
cd /Users/vivekdurairaj/Projects/Cogumi-IntLLM
source venv/bin/activate
```

### Create files
```bash
touch qins_kv_cache.py
touch test_kv_compression.py
touch benchmark_kv_memory.py
```

### Test-driven development
```bash
# 1. Write test first
python test_kv_compression.py  # Should fail (not implemented)

# 2. Implement qins_kv_cache.py
# ...

# 3. Test again
python test_kv_compression.py  # Should pass

# 4. Integrate
python test_pattern_a_clean.py  # With KV compression enabled

# 5. Benchmark
python benchmark_kv_memory.py  # Measure savings
```

---

## 📚 Reference

### 6-bit Quantization Math

```python
# Quantize: FP32 [-∞, +∞] → uint6 [0, 63]
def quantize_6bit(x):
    x_min, x_max = x.min(), x.max()
    x_norm = (x - x_min) / (x_max - x_min)  # [0, 1]
    x_quant = (x_norm * 63).round().clamp(0, 63).to(torch.uint8)
    return x_quant, x_min, x_max

# Dequantize: uint6 [0, 63] → FP32 [-∞, +∞]
def dequantize_6bit(x_quant, x_min, x_max):
    x_norm = x_quant.float() / 63.0  # [0, 1]
    x = x_norm * (x_max - x_min) + x_min
    return x
```

### Bit-Packing (6-bit)

```python
# Pack 4 × 6-bit values into 3 bytes (24 bits)
def pack_6bit(values):
    # values: [N] uint8 with values in [0, 63]
    # returns: [N*6/8] uint8 packed
    
    # Example: [a, b, c, d] with 6 bits each
    # Packed: [aaaaaa|bb] [bbbb|cccc] [cc|dddddd]
    
    packed = []
    for i in range(0, len(values), 4):
        a, b, c, d = values[i:i+4]
        
        byte0 = (a << 2) | (b >> 4)           # aaaaaa|bb
        byte1 = ((b & 0xF) << 4) | (c >> 2)   # bbbb|cccc  
        byte2 = ((c & 0x3) << 6) | d          # cc|dddddd
        
        packed.extend([byte0, byte1, byte2])
    
    return torch.tensor(packed, dtype=torch.uint8)
```

### Page-Based Storage

```python
# Store KV cache in pages
class PagedKVCache:
    def __init__(self, page_size_kb=32):
        self.page_size = page_size_kb * 1024  # bytes
        self.pages = []
    
    def allocate_page(self, k, v):
        """Store compressed K, V in a page."""
        k_quant, k_min, k_max = quantize_6bit(k)
        v_quant, v_min, v_max = quantize_6bit(v)
        
        k_packed = pack_6bit(k_quant.flatten())
        v_packed = pack_6bit(v_quant.flatten())
        
        page = {
            'k_packed': k_packed,
            'v_packed': v_packed,
            'k_range': (k_min, k_max),
            'v_range': (v_min, v_max),
            'shape': k.shape
        }
        
        self.pages.append(page)
        return len(self.pages) - 1  # page_id
    
    def fetch_page(self, page_id):
        """Decompress and return K, V from page."""
        page = self.pages[page_id]
        
        k_quant = unpack_6bit(page['k_packed'], page['shape'].numel())
        v_quant = unpack_6bit(page['v_packed'], page['shape'].numel())
        
        k = dequantize_6bit(k_quant, *page['k_range']).reshape(page['shape'])
        v = dequantize_6bit(v_quant, *page['v_range']).reshape(page['shape'])
        
        return k, v
```

---

## 🐛 Debugging Tips

### If token match is low (<95%):
- Check quantization range (outliers causing overflow?)
- Try 8-bit instead of 6-bit
- Check K vs V error separately (which is worse?)

### If memory savings are low (<2×):
- Verify packed storage is actually used
- Check page allocation (are pages too small?)
- Profile memory with `torch.cuda.memory_summary()`

### If latency is high (>10% overhead):
- Profile with `torch.profiler`
- Check if decompression is in critical path
- Consider caching decompressed pages

---

## 📞 Next Week Preview

Once KV compression works:
- 🔥 Start Phase 2: Jacobian transport (V-path)
- 📊 Collect calibration data (1000 samples)
- 🧪 Implement `qins_jacobian_transport.py`

But first: **Get KV compression working this week!**

---

**Start now**: Create `qins_kv_cache.py` skeleton! 🚀
