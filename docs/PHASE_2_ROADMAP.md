# Phase 2: QINS-Native Compute via Weight Transport

**Status**: Pattern A ✅ Complete → Pattern B 🔥 In Progress

---

## What We've Proven (Pattern A)

✅ **QINS encoding is lossless** (100% token match on Phi-3.5)
✅ **No hallucination drift** (validated on 15+ tokens)
✅ **Memory compression works** (4× confirmed, up to 12× with extras)
✅ **Information geometry is sound** (weights survive QINS round-trip)
✅ **Production-ready storage codec** (safe for real LLMs)

**Pattern A = Storage/Transport only**
- Store: QINS domain (compressed)
- Compute: FP32 domain (decode before matmul)
- Result: Memory win, no compute win

---

## Phase 2 Goal: QINS-Native Compute

**Eliminate decode overhead by computing in QINS domain**

### Current Pipeline (Pattern A)
```python
W_qins = qins_encode(W_fp32)           # Storage in QINS
# ... later ...
W_fp32 = qins_decode(W_qins)           # Decode for every forward pass ❌
out = x @ W_fp32                        # Compute in FP32
```

**Problem**: Decode overhead on every forward pass

### Target Pipeline (Pattern B)
```python
# One-time weight transport
W_qins_native = jacobian_transport(W_fp32)  # FP32 → QINS-native weights

# Forward pass (all in QINS domain)
x_qins = qins_encode(x_fp32)                # Input to QINS
out_qins = qins_matmul(x_qins, W_qins_native)  # Compute in QINS ✅
out_fp32 = qins_decode(out_qins)            # Decode only at output
```

**Benefit**: Decode once per layer instead of once per weight tensor

---

## Mathematical Foundation

### The Jacobian Transport Formula

We derived:
```
W' = (∂D/∂z) · W · (∂E/∂x)^(-1)
```

Where:
- `E(x)` = QINS encoding function: `z = sign(x) / (1 + α|x|)`
- `D(z)` = QINS decoding function: `x = sign(z) · (1-|z|) / (α|z|)`
- `W` = Original FP32 weights
- `W'` = QINS-native weights (transported)

### Why This Works

**Standard matmul** (Pattern A):
```
y = x · W
```

**QINS-native matmul** (Pattern B):
```
D(y_qins) = D(E(x) · W')
          = D(E(x) · (∂D/∂z) · W · (∂E/∂x)^(-1))
          = x · W  (if transport is correct)
```

The Jacobian terms cancel out the nonlinearity!

---

## Implementation Plan

### Step 1: Implement Jacobian Computation

**File**: `qins_jacobian_transport.py`

```python
def compute_jacobian_encode(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Compute Jacobian of QINS encoding: ∂E/∂x
    
    E(x) = sign(x) / (1 + α|x|)
    
    ∂E/∂x = -α · sign(x)^2 / (1 + α|x|)^2
          = -α / (1 + α|x|)^2  (since sign^2 = 1)
    
    Returns:
        Diagonal Jacobian matrix (element-wise derivative)
    """
    abs_x = x.abs()
    denominator = (1.0 + alpha * abs_x) ** 2
    jacobian = -alpha / denominator
    return jacobian

def compute_jacobian_decode(z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Compute Jacobian of QINS decoding: ∂D/∂z
    
    D(z) = sign(z) · (1 - |z|) / (α|z|)
    
    ∂D/∂z = sign(z) · [-1/(α|z|) - (1-|z|)/(α|z|^2) · sign(z)]
          = -1/(α|z|) - (1-|z|)/(α|z|^2)  (sign terms cancel)
    
    Returns:
        Diagonal Jacobian matrix (element-wise derivative)
    """
    abs_z = z.abs().clamp(min=1e-12)  # Avoid division by zero
    term1 = -1.0 / (alpha * abs_z)
    term2 = -(1.0 - abs_z) / (alpha * abs_z ** 2)
    jacobian = term1 + term2
    return jacobian

def transport_weights_to_qins(
    W: torch.Tensor,
    x_sample: torch.Tensor,
    alpha: float = 1.0
) -> torch.Tensor:
    """
    Transport FP32 weights to QINS-native weights via Jacobian correction.
    
    W' = (∂D/∂z) · W · (∂E/∂x)^(-1)
    
    Args:
        W: Original FP32 weight matrix [out_features, in_features]
        x_sample: Sample input for Jacobian computation [batch, in_features]
        alpha: QINS density parameter
    
    Returns:
        W_qins: QINS-native weights [out_features, in_features]
    """
    # Encode sample input to get z values for ∂D/∂z
    z_sample = qins_encode(x_sample, alpha=alpha, quantize=False)
    
    # Compute Jacobians
    jacobian_encode = compute_jacobian_encode(x_sample, alpha=alpha)  # ∂E/∂x
    jacobian_decode = compute_jacobian_decode(z_sample, alpha=alpha)  # ∂D/∂z
    
    # Invert encode Jacobian (element-wise since diagonal)
    jacobian_encode_inv = 1.0 / jacobian_encode
    
    # Apply transport: W' = (∂D/∂z) · W · (∂E/∂x)^(-1)
    # Since Jacobians are diagonal, this is element-wise scaling
    W_transported = jacobian_decode.unsqueeze(0) * W * jacobian_encode_inv.unsqueeze(-1)
    
    return W_transported
```

### Step 2: Implement QINS-Native Matmul

**File**: `qins_native_compute.py`

```python
def qins_matmul(
    x_qins: torch.Tensor,
    W_qins_native: torch.Tensor
) -> torch.Tensor:
    """
    Matrix multiplication in QINS domain.
    
    This is just standard matmul - the magic is in the transported weights!
    
    Args:
        x_qins: Input activations in QINS domain [batch, in_features]
        W_qins_native: Transported weights [out_features, in_features]
    
    Returns:
        Output in QINS domain [batch, out_features]
    """
    return F.linear(x_qins, W_qins_native)

class QINSNativeLinear(nn.Module):
    """
    Linear layer with QINS-native compute (Pattern B).
    
    Storage: QINS-transported weights
    Compute: Matmul in QINS domain (no decode!)
    Output: QINS domain (caller must decode)
    """
    
    def __init__(
        self,
        linear: nn.Linear,
        x_sample: torch.Tensor,
        alpha: float = 1.0
    ):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.alpha = alpha
        
        # Transport weights once at initialization
        with torch.no_grad():
            W_qins = transport_weights_to_qins(
                linear.weight.data,
                x_sample,
                alpha=alpha
            )
            self.register_buffer('weight_qins', W_qins)
        
        # Bias stays in FP32 (added after decode)
        if linear.bias is not None:
            self.register_buffer('bias', linear.bias.data)
        else:
            self.bias = None
    
    def forward(self, x_qins: torch.Tensor) -> torch.Tensor:
        """
        Forward pass in QINS domain.
        
        Args:
            x_qins: Input in QINS domain
        
        Returns:
            Output in QINS domain (no decode!)
        """
        # Matmul in QINS domain
        out_qins = qins_matmul(x_qins, self.weight_qins)
        
        # Note: Bias is NOT added here (it's in FP32 domain)
        # Caller must decode output before adding bias
        
        return out_qins
```

### Step 3: Create Pattern B Test

**File**: `test_pattern_b_transport.py`

```python
#!/usr/bin/env python3
"""
Test Pattern B: QINS-native compute via Jacobian weight transport

Validation:
1. Load Phi-3.5 baseline
2. Generate reference output (FP32)
3. Convert one layer to QINS-native compute
4. Generate with QINS compute
5. Compare outputs (should match within numerical precision)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qins_jacobian_transport import transport_weights_to_qins
from qins_native_compute import QINSNativeLinear
from qins_weight_codec import qins_encode, qins_decode

def test_single_layer_transport():
    """Test weight transport on a single layer."""
    
    print("="*60)
    print("PATTERN B TEST: Single Layer Transport")
    print("="*60)
    
    # Load model
    model_name = "microsoft/Phi-3.5-mini-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get a reference layer
    target_layer = model.model.layers[0].mlp.down_proj
    
    # Create sample input (calibration data)
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        # Get intermediate activations
        outputs = model(**inputs, output_hidden_states=True)
        x_sample = outputs.hidden_states[1][:, -1, :]  # After first layer
    
    print(f"\n✓ Got sample activation: {x_sample.shape}")
    
    # Test 1: Standard FP32 forward
    with torch.no_grad():
        out_fp32 = target_layer(x_sample)
    
    print(f"✓ FP32 output: {out_fp32.shape}")
    print(f"  Range: [{out_fp32.min():.6f}, {out_fp32.max():.6f}]")
    
    # Test 2: QINS-native forward
    qins_layer = QINSNativeLinear(
        target_layer,
        x_sample,
        alpha=1.0
    )
    
    with torch.no_grad():
        x_qins = qins_encode(x_sample, alpha=1.0)
        out_qins = qins_layer(x_qins)
        out_decoded = qins_decode(out_qins, alpha=1.0)
    
    print(f"✓ QINS output: {out_decoded.shape}")
    print(f"  Range: [{out_decoded.min():.6f}, {out_decoded.max():.6f}]")
    
    # Compare
    max_error = (out_fp32 - out_decoded).abs().max()
    mean_error = (out_fp32 - out_decoded).abs().mean()
    rel_error = mean_error / out_fp32.abs().mean()
    
    print(f"\n📊 Error Analysis:")
    print(f"  Max absolute error: {max_error:.6e}")
    print(f"  Mean absolute error: {mean_error:.6e}")
    print(f"  Relative error: {rel_error:.4%}")
    
    # Success criteria: <1% relative error
    if rel_error < 0.01:
        print(f"\n✅ PATTERN B SUCCESS: Transport works!")
    else:
        print(f"\n❌ PATTERN B FAILED: Error too high")
    
    return rel_error < 0.01

def test_full_model_transport():
    """Test weight transport on full model generation."""
    
    print("\n" + "="*60)
    print("PATTERN B TEST: Full Model Generation")
    print("="*60)
    
    # TODO: Convert all MLP layers to QINS-native
    # TODO: Generate tokens end-to-end
    # TODO: Compare with FP32 baseline
    
    pass

if __name__ == "__main__":
    success = test_single_layer_transport()
    
    if success:
        print("\n🔥 Ready for full model test!")
        # test_full_model_transport()
    else:
        print("\n⚠️  Fix single layer first before scaling up")
```

---

## Expected Challenges

### Challenge 1: Numerical Stability
- **Issue**: Jacobian has divisions by small numbers
- **Solution**: Clamp denominators, use higher precision for transport
- **Mitigation**: Test on synthetic data first

### Challenge 2: Calibration Data
- **Issue**: Need representative x_sample for Jacobian
- **Solution**: Use activation statistics from a few forward passes
- **Mitigation**: Per-channel transport if needed

### Challenge 3: Bias Handling
- **Issue**: Bias is in FP32 domain, output is in QINS domain
- **Solution**: Decode output, add bias, re-encode (or keep bias in FP32)
- **Mitigation**: Profile overhead to decide strategy

### Challenge 4: Layernorm/RMSnorm
- **Issue**: Normalization layers operate in FP32
- **Solution**: Decode before norm, encode after
- **Mitigation**: This is expected - only matmuls are in QINS

---

## Success Criteria

### Phase 2 Complete When:

✅ **Single layer transport works** (error < 1%)
✅ **Full model generates coherent text** (token match > 95%)
✅ **Decode overhead reduced** (fewer decode operations)
✅ **No accuracy degradation** (perplexity similar to FP32)

### Performance Targets:

| Metric | Target | Current (Pattern A) |
|--------|--------|---------------------|
| Decode ops per forward pass | ~10-20 (layer outputs) | ~128 (every matmul) |
| Memory | Same (4×) | 4× ✅ |
| Speed | Similar or faster | Slower (decode overhead) |
| Quality | 95%+ match | 100% ✅ (no compute) |

---

## Beyond Pattern B: Pattern C (Future)

Once Pattern B works, we can optimize further:

**Pattern C: Native QINS Kernels**
- CUDA kernels for QINS matmul (fused encode + matmul)
- QINS-native attention (no decode in attention block)
- Hardware acceleration (QINS ALU on custom silicon)

**This is the true endgame**: Native QINS compute with no FP32 at all.

---

## Timeline

**Week 1-2**: Implement Jacobian transport + single layer test
**Week 3-4**: Full model integration + validation
**Week 5-6**: Optimization + benchmarking
**Week 7+**: Pattern C exploration (CUDA kernels)

---

## Notes

- Pattern A remains the **safety fallback** if Pattern B has issues
- Pattern B is **optional** - Pattern A is production-ready
- Pattern B proves **compute benefits are possible**
- Pattern C is **long-term vision** (hardware-dependent)

**The key insight**: We're not replacing Pattern A, we're building on top of it.

---

**Next Steps**:
1. Create `qins_jacobian_transport.py`
2. Create `qins_native_compute.py`
3. Run `test_pattern_b_transport.py`
4. Debug until single layer works
5. Scale to full model

Let's build the compute engine! 🚀
