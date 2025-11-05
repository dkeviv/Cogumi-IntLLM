# QINS Execution Plan: A → B → C

**Clear milestones, measurable outcomes, pragmatic staging**

---

## Phase 1: Pattern A Production-Ready ✅ NOW

### Objective
Ship working compression with minimal risk, maximum practicality

### Implementation: Bit-Packed Storage

**What to compress:**
- ✅ **KV cache values** (largest runtime memory hog)
- ✅ **Optional: Large activations** (if they spill to memory)
- ❌ **NOT weights yet** (Pattern B will handle this better)

**Compression strategy:**
```python
# Simple, fast, cache-friendly
Encoding:   6-bit or 8-bit quantization (no entropy coding yet)
Packing:    Bit-pack into pages
Page size:  16-64 KB (CPU cache line friendly)
Latency:    <1ms decompression per page
```

**Why this approach:**
- ✅ **Proven**: Standard industry practice (vLLM, TGI use similar)
- ✅ **Fast**: No Huffman overhead, hardware-friendly
- ✅ **Safe**: Easy to validate, easy to disable
- ✅ **Immediate ROI**: KV cache is the real memory bottleneck in inference

**Deliverables:**
```
✅ qins_kv_cache.py          - Bit-packed KV storage
✅ test_kv_compression.py    - Validation (match rate >99%)
✅ benchmark_kv_memory.py    - Memory savings measurement
✅ Production config         - Feature flag to enable/disable
```

**Success criteria:**
- KV cache compression: 2-4× (6-bit: 2.67×, 8-bit with sparsity: 4×)
- Decode latency: <5% overhead
- Token match: >99% (allow tiny numerical drift)
- Memory savings: 50-75% on long context tasks

**Timeline:** 1-2 weeks

---

## Phase 2: Pattern B Prototype (V-Path First) 🔥 NEXT

### Objective
Prove Jacobian transport works on simplest path before full rollout

### Why V-Path First?

**Attention architecture:**
```
Q, K, V = q_proj(x), k_proj(x), v_proj(x)
attn_weights = softmax(Q @ K^T / sqrt(d))
output = attn_weights @ V
```

**V-path is safest:**
- ✅ **Linear dependency**: V gets multiplied by attention weights (linear!)
- ✅ **No softmax**: Doesn't interact with nonlinear operations
- ✅ **Easy to isolate**: Can test without touching Q/K complexity
- ✅ **Meaningful**: Still handles ~33% of attention compute

**Implementation:**
```python
# Step 1: Convert v_proj weights to QINS-native
W_v_qins = jacobian_transport(W_v, x_sample, alpha=1.0)

# Step 2: Forward pass with QINS V-path
x_qins = qins_encode(x)
V_qins = qins_matmul(x_qins, W_v_qins)  # Compute in QINS!
V = qins_decode(V_qins)                  # Decode once

# Step 3: Rest of attention in FP32 (Q, K unchanged)
attn_weights = softmax(Q @ K.T / sqrt(d))
output = attn_weights @ V  # V came from QINS path
```

**Validation strategy:**
1. **Single layer test**: Convert only layer[0].self_attn.v_proj
2. **Generate 100 tokens**: Compare with FP32 baseline
3. **Measure divergence**: Token match rate, perplexity, numerical error
4. **Success threshold**: >95% token match, <2% perplexity increase

**Keep bit-packed KV in place:**
- Don't remove Phase 1 compression
- V-path transport is **orthogonal** to KV storage
- Measure combined benefit

**Deliverables:**
```
✅ qins_jacobian_transport.py    - Weight transport implementation
✅ qins_native_compute.py         - QINS matmul operations  
✅ test_v_path_single_layer.py   - V-path only validation
✅ test_v_path_full_model.py     - All V-paths converted
✅ benchmark_decode_reduction.py - Count decode ops (128 → ~64)
```

**Success criteria:**
- Single layer: Token match >95%, error <1%
- Full model: Token match >90%, perplexity <5% increase
- Decode ops: Reduced by ~50% (v_proj decoded once per layer)
- Numerical stability: No NaN/Inf, weights bounded

**Timeline:** 3-4 weeks

---

## Phase 3: Post-B Tuning & Entropy Analysis 📊 AFTER B WORKS

### Objective
Optimize compression with real distribution data from working Pattern B

### Why wait until after B?

**Pattern B changes the distributions:**
```python
# Pattern A: Weight distribution is standard Gaussian-ish
W_fp32 ~ N(0, 0.1²)  →  entropy(qins_encode(W_fp32)) = ?

# Pattern B: Transported weights have different distribution
W_qins_native = jacobian_transport(W_fp32)  →  entropy(W_qins_native) = ??
# ^^ We don't know this until B is implemented!
```

**Similarly for activations:**
```python
# FP32 activations: known distribution
x_fp32 ~ distribution_1  →  entropy ≈ X bits

# QINS activations after B: NEW distribution  
x_qins = qins_encode(x_fp32)  →  entropy ≈ ?? bits
# ^^ Measure this AFTER B is running
```

### Entropy Measurement Plan

**Step 1: Instrument Pattern B to collect statistics**
```python
class QINSStatsCollector:
    """Collect entropy stats during inference."""
    
    def collect_weight_entropy(self, W_qins_native):
        """Measure entropy of transported weights."""
        # Histogram of quantized values
        hist = torch.histc(W_qins_native, bins=256)
        entropy = -sum(p * log(p) for p in hist if p > 0)
        return entropy
    
    def collect_activation_entropy(self, x_qins):
        """Measure entropy of QINS activations."""
        # Per-layer statistics
        hist = torch.histc(x_qins, bins=256)
        entropy = -sum(p * log(p) for p in hist if p > 0)
        return entropy
```

**Step 2: Run inference on validation set (1000+ samples)**
```bash
python collect_entropy_stats.py \
  --model phi35-pattern-b \
  --dataset wikitext-103 \
  --samples 1000 \
  --output entropy_analysis.json
```

**Step 3: Analyze results**
```
Expected findings:
- Some layers: high entropy (don't compress, waste of time)
- Some layers: low entropy (great candidates for Huffman/RLE)
- KV cache: usually low entropy (already handling this)
- Cold weights: very low entropy (good Huffman candidates)
```

### Per-Tensor Entropy Coding Decision

**Add compression where net-beneficial:**

| Component | Typical Entropy | Strategy |
|-----------|----------------|----------|
| **Hot weights** (frequently used) | Medium-High | NO entropy coding (latency matters) |
| **Cold weights** (infrequent layers) | Low-Medium | Huffman compress (load time okay) |
| **KV cache pages** | Low (locality) | RLE + Huffman (big win) |
| **QINS activations** | Unknown yet | Measure first! |
| **Transported weights** | Unknown yet | Measure first! |

**Implementation:**
```python
# Per-tensor decision
if layer.access_frequency < threshold:  # Cold path
    W_compressed = huffman_compress(W_qins)
    compression_type = "huffman"
else:  # Hot path
    W_compressed = W_qins  # No compression
    compression_type = "none"

# Per-channel option for mixed strategy
if layer.has_high_variance_channels:
    W_compressed = per_channel_huffman(W_qins)
```

**Deliverables:**
```
✅ collect_entropy_stats.py       - Instrumentation
✅ entropy_analysis.json          - Distribution data
✅ compression_policy.py          - Hot/cold decision logic
✅ per_tensor_huffman.py          - Selective compression
✅ benchmark_compression_impact.py - Latency vs compression trade-off
```

**Success criteria:**
- Identify 30-50% of weights suitable for Huffman (cold path)
- Achieve 6-10× compression on cold weights
- Keep hot path latency <2% overhead
- Overall: 5-8× compression (combined techniques)

**Timeline:** 2-3 weeks

---

## Phase 4: Fused Decompress→QINS-MAC Kernels 🚀 OPTIMIZATION

### Objective
Eliminate overhead with fused operations for hotspots

### Hotspot Analysis

**Expected bottlenecks after Phase 3:**
```
Profiling will likely show:
1. KV page decompression (if entropy coded)
2. QINS encode/decode conversions
3. Quantization/dequantization steps
4. Memory bandwidth (loading compressed data)
```

**Fused kernel strategy:**
```python
# BEFORE: Three separate operations
x_decompressed = decompress(x_compressed)  # Op 1
x_qins = qins_encode(x_decompressed)       # Op 2  
output = qins_matmul(x_qins, W_qins)       # Op 3

# AFTER: Single fused kernel
output = fused_decompress_qins_mac(
    x_compressed,  # Input
    W_qins,        # Weights
    decompress_params,  # Huffman table, etc.
    alpha=1.0      # QINS parameter
)
# ^^ One kernel launch, minimal memory traffic
```

**CUDA kernel design:**
```cuda
__global__ void fused_decompress_qins_mac(
    uint8_t* x_compressed,      // Compressed input
    uint8_t* W_qins,            // QINS weights
    float*   output,            // Result
    int*     huffman_table,     // Decompression LUT
    float    alpha              // QINS alpha
) {
    // Thread-local decompression
    float x_local = decompress_huffman(x_compressed, huffman_table);
    
    // QINS encode inline
    float x_qins = qins_encode_inline(x_local, alpha);
    
    // MAC operation
    float w_qins = W_qins[thread_idx];
    atomicAdd(&output[output_idx], x_qins * w_qins);
}
```

**Deliverables:**
```
✅ qins_cuda_kernels.cu           - Fused CUDA implementations
✅ test_kernel_correctness.py     - Numerical validation
✅ benchmark_kernel_speedup.py    - Latency improvement
✅ profile_guided_fusion.py       - Auto-select fusion candidates
```

**Success criteria:**
- Identify 5-10 critical hotspots (top 80% of compute time)
- Fused kernels: 2-3× speedup over separate ops
- Numerical accuracy: <0.1% error vs reference
- Overall inference: 1.5-2× speedup

**Timeline:** 4-6 weeks

---

## Phase 5: Extend B to MLP Layers 🔥 SCALE UP

### Objective
Apply proven V-path strategy to MLP (larger weight matrices)

### Why MLP after attention?

**Attention V-path taught us:**
- ✅ Jacobian transport works
- ✅ Numerical stability is manageable
- ✅ Calibration data requirements
- ✅ Per-layer error accumulation patterns

**MLP is next logical step:**
```python
# Phi-3.5 MLP structure
x_gate = gate_proj(x)   # [hidden, intermediate]  ← QINS candidate
x_up = up_proj(x)       # [hidden, intermediate]  ← QINS candidate  
x_act = silu(x_gate) * x_up
output = down_proj(x_act)  # [intermediate, hidden]  ← QINS candidate

# 3 large matmuls per MLP block
# 32 layers × 3 = 96 MLP matmuls (vs 32 V-paths)
# ^^ Much bigger compute win if this works
```

**Staged rollout:**
```
Week 1-2: down_proj only (like V-path, output of MLP)
Week 3-4: gate_proj + up_proj (parallel paths)
Week 5-6: Full MLP validation (all 3 together)
```

**New challenges in MLP:**
- ⚠️ **SiLU nonlinearity**: Happens between QINS ops
- ⚠️ **Larger matrices**: 2-4× more weights than attention
- ⚠️ **Error accumulation**: 3 QINS ops in sequence
- ⚠️ **Calibration data**: Need representative activation stats

**Mitigation strategies:**
```python
# Handle SiLU carefully
x_gate_qins = qins_matmul(x_qins, W_gate_qins)
x_gate_fp32 = qins_decode(x_gate_qins)  # Decode before SiLU
x_act = silu(x_gate_fp32)  # Nonlinearity in FP32
x_act_qins = qins_encode(x_act)  # Re-encode after

# OR: Approximate SiLU in QINS domain (risky, test carefully)
```

**Deliverables:**
```
✅ test_mlp_down_proj.py          - Single MLP layer (down_proj)
✅ test_mlp_gate_up.py            - Parallel paths (gate+up)
✅ test_mlp_full.py               - Complete MLP in QINS
✅ validate_mlp_all_layers.py     - All 32 MLPs converted
✅ benchmark_mlp_speedup.py       - Decode reduction measurement
```

**Success criteria:**
- Single MLP: Token match >95%, perplexity <3% increase
- All MLPs: Token match >85%, perplexity <10% increase
- Decode ops: Reduced from ~128 to ~10-20 (only layer outputs)
- Combined attention+MLP: >90% of model in QINS compute

**Timeline:** 4-6 weeks

---

## Phase 6: Final Compression Pass 📊 OPTIMIZE

### Objective
Revisit all compression decisions with complete Pattern B distributions

### Re-measure Everything

**Now we have full data:**
- ✅ V-path transported weights distribution
- ✅ MLP transported weights distribution  
- ✅ QINS activation distributions (all layers)
- ✅ Error accumulation patterns
- ✅ Access frequency patterns (hot/cold)

**Comprehensive entropy analysis:**
```bash
python final_entropy_analysis.py \
  --model phi35-pattern-b-full \
  --include-weights \
  --include-activations \
  --include-kv-cache \
  --samples 10000 \
  --output final_compression_plan.json
```

**Optimization decisions:**

| Component | Compress? | Method | Expected Gain |
|-----------|-----------|--------|---------------|
| Hot attention weights | ❌ No | Raw bit-pack | 4× |
| Cold attention weights | ✅ Yes | Huffman | 8-10× |
| Hot MLP weights | ❌ No | Raw bit-pack | 4× |
| Cold MLP weights | ✅ Yes | Huffman | 8-10× |
| KV cache | ✅ Yes | RLE + Huffman | 6-8× |
| QINS activations | 🤔 Maybe | Per-layer decision | 2-4× |
| Transported weights | 🤔 Maybe | Measure first | TBD |

**Deliverables:**
```
✅ final_compression_plan.json    - Per-tensor compression policy
✅ adaptive_compression.py        - Runtime policy engine
✅ compression_benchmark_suite.py - Full system measurement
✅ production_config.yaml         - Tuned hyperparameters
```

**Success criteria:**
- Overall compression: 6-10× (combined all techniques)
- Inference latency: <10% overhead vs FP32 baseline
- Quality: Perplexity within 5% of FP32
- Memory: Run Phi-3.5 in <2GB (vs ~8GB FP32)

**Timeline:** 2-3 weeks

---

## Summary Timeline

| Phase | Focus | Duration | Cumulative |
|-------|-------|----------|------------|
| **1. Pattern A Production** | KV bit-packing | 1-2 weeks | 2 weeks |
| **2. Pattern B V-Path** | Jacobian transport prototype | 3-4 weeks | 6 weeks |
| **3. Entropy Analysis** | Measure distributions | 2-3 weeks | 9 weeks |
| **4. Fused Kernels** | Optimize hotspots | 4-6 weeks | 15 weeks |
| **5. MLP Extension** | Scale to full model | 4-6 weeks | 21 weeks |
| **6. Final Compression** | Tune everything | 2-3 weeks | 24 weeks |

**Total: ~6 months to full Pattern B production**

---

## Validation Gates

**Must pass before moving to next phase:**

### Gate 1 → 2: Pattern A works
- ✅ KV compression >2× with <5% latency overhead
- ✅ Token generation quality unchanged

### Gate 2 → 3: V-Path validated  
- ✅ Single layer >95% token match
- ✅ Full model >90% token match
- ✅ No numerical instability (no NaN/Inf)

### Gate 3 → 4: Entropy data collected
- ✅ 1000+ samples processed
- ✅ Clear hot/cold separation identified
- ✅ Compression policy defined

### Gate 4 → 5: Kernels provide speedup
- ✅ 2× speedup on critical path
- ✅ <0.1% numerical error
- ✅ Portable across GPUs

### Gate 5 → 6: MLP works
- ✅ All MLPs in QINS domain
- ✅ >85% token match on full model
- ✅ Combined attention+MLP stable

### Gate 6 → Production: Final validation
- ✅ 6-10× compression achieved
- ✅ <10% latency overhead
- ✅ <5% perplexity increase
- ✅ Stable over 10,000+ token runs

---

## Risk Mitigation

### Risk 1: Jacobian transport numerically unstable
**Mitigation:**
- Use FP64 for transport computation
- Per-channel transport with separate calibration
- Clamp intermediate values aggressively
- **Fallback**: Keep Pattern A (still useful)

### Risk 2: MLP nonlinearity breaks QINS
**Mitigation:**  
- Decode before SiLU, re-encode after
- Test approximate SiLU in QINS domain
- Hybrid: Some MLP layers in FP32
- **Fallback**: QINS only for attention

### Risk 3: Entropy coding overhead too high
**Mitigation:**
- Only compress cold paths (load-time only)
- Fused decompress kernels for hot paths
- Per-page decision (compress large pages only)
- **Fallback**: Bit-pack only (still 4× win)

### Risk 4: Quality degradation unacceptable
**Mitigation:**
- Higher bit-width (10-bit, 12-bit options)
- Per-layer precision tuning
- Hybrid FP32/QINS (critical layers in FP32)
- **Fallback**: Pattern A only (safe 4× compression)

---

## Success Metrics

### Phase 1 (Pattern A)
- ✅ Ship date: 2 weeks
- ✅ Compression: 2-4×
- ✅ Quality: No degradation

### Phase 2 (Pattern B V-Path)
- ✅ Ship date: 6 weeks  
- ✅ Decode ops: 128 → 64
- ✅ Quality: >90% match

### Phase 5 (Full Pattern B)
- ✅ Ship date: 21 weeks
- ✅ Decode ops: 128 → 10-20
- ✅ Quality: >85% match

### Phase 6 (Production)
- ✅ Ship date: 24 weeks
- ✅ Compression: 6-10×
- ✅ Latency: <10% overhead
- ✅ Quality: <5% perplexity increase

### Ultimate Goal
- 🚀 Phi-3.5 (3.8B) runs in <2GB RAM
- 🚀 Inference on edge devices (phones, RPi)
- 🚀 Proof of QINS compute viability
- 🚀 Foundation for Pattern C (hardware)

---

## Decision Points

### After Phase 2: Continue to MLP?
**If V-path quality is good (>95% match):**
- ✅ Proceed to Phase 5 (MLP extension)

**If V-path quality is marginal (90-95% match):**
- ⚠️ Pause, optimize V-path first
- Consider: Higher precision, better calibration

**If V-path quality is poor (<90% match):**
- ❌ Stop Pattern B
- ✅ Ship Pattern A only (still valuable!)
- ⏸️ Wait for better numerical methods

### After Phase 5: Worth pursuing Pattern C?
**If MLP works (>85% match):**
- ✅ Pattern B is viable
- ✅ Start Pattern C planning (hardware/CUDA)

**If MLP doesn't work:**
- ❌ Pattern B limited to attention only
- ⏸️ Ship attention-only QINS
- 🤔 Re-evaluate ROI

---

## Next Immediate Actions

**This week:**
1. ✅ Create `qins_kv_cache.py` (bit-packed KV storage)
2. ✅ Write `test_kv_compression.py` (validation harness)
3. ✅ Benchmark KV memory savings

**Next week:**
1. ✅ Integrate KV compression into inference loop
2. ✅ Validate on 100+ token generation
3. ✅ Measure latency impact

**Week 3:**
1. ✅ Ship Pattern A feature flag (beta)
2. 🔥 Start `qins_jacobian_transport.py` (Pattern B foundation)
3. 🔥 Collect calibration data for V-path

---

## Resources Needed

**Compute:**
- Development: M4 Mac (24GB) ✅ Have
- Testing: GPU instance for kernel dev (later phases)
- Benchmarking: Long-running inference jobs

**Data:**
- Calibration: 1000-10000 samples (WikiText, C4)
- Validation: Diverse prompts (quality testing)
- Profiling: Representative workloads (latency testing)

**Tools:**
- ✅ PyTorch (have)
- ✅ Transformers (have)  
- 🔧 CUDA toolkit (Phase 4)
- 🔧 Profiler (nvprof, PyTorch profiler)

---

## Conclusion

**This is a crisp, staged plan:**
- ✅ **Phase 1**: Ship value fast (KV compression)
- 🔥 **Phase 2**: Validate core idea (V-path transport)
- 📊 **Phase 3-6**: Scale and optimize based on data

**Each phase has:**
- Clear deliverables
- Measurable success criteria
- Risk mitigation strategies
- Go/no-go decision gates

**If things don't work:**
- Pattern A is still valuable (4× compression)
- Pattern B has fallbacks (attention-only mode)
- Pattern C is optional (hardware-dependent)

**Let's execute!** 🚀

---

**Start now**: `qins_kv_cache.py` (Phase 1, Week 1)
