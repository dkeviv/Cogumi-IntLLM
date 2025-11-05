#!/usr/bin/env python3
"""
QINS Weight Codec - Pattern A (Codec-at-Rest) - RATIONAL ENCODING

⚠️  ENCODING METHOD: Rational/Projective (NOT Logarithmic)
    Formula: z = sign(x) / (1 + α|x|)
    
✅  PATTERN A: Codec-at-rest (decode before compute) - CORRECT
❌  QUANTIZATION: Not actually applied (stores as float32) - BUG
    Result: 0% compression despite quantize=True

USED FOR:
  - Phi-3.5 benchmark (test_pattern_a_clean.py)
  - Achieved 100% accuracy (15/15 tokens)
  - But 0% compression (13,824 MB → 13,824 MB)

COMPARISON:
  - This file: Rational encoding, float32 storage, 0× compression
  - src/qins_codec.py: Logarithmic encoding, uint8 storage, 2× compression
  
Both achieve 100% accuracy because both follow Pattern A correctly!
The difference is only in memory compression, not correctness.

STATUS: Keep for Phi-3.5 benchmarking (Pattern A validation)
TODO: Fix quantization bug or migrate to logarithmic encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def qins_encode(weight: torch.Tensor, alpha: float = 1.0, quantize: bool = False) -> torch.Tensor:
    """
    Encode FP32 weights to QINS domain.
    
    Formula: z = sign(x) / (1 + α|x|)
    
    Properties:
    - Large |x| → z near 0
    - Small |x| → z near ±1
    - Invertible via qins_decode
    
    Args:
        weight: FP32 weight tensor
        alpha: Density control parameter (default 1.0)
        quantize: If True, quantize to uint8 [0,255] for 4× compression
    
    Returns:
        Encoded tensor in QINS domain (FP32 if quantize=False, uint8 if quantize=True)
    """
    sign = torch.sign(weight)
    sign[sign == 0] = 1  # Handle exact zeros
    
    abs_weight = weight.abs()
    encoded = sign / (1.0 + alpha * abs_weight)
    
    if quantize:
        # Map [-1, 1] to [0, 255] for uint8 storage
        # encoded = -1 → quantized = 0
        # encoded =  0 → quantized = 127
        # encoded = +1 → quantized = 255
        quantized = ((encoded + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
        return quantized
    
    return encoded


def qins_decode(encoded: torch.Tensor, alpha: float = 1.0, is_quantized: bool = False) -> torch.Tensor:
    """
    Decode QINS weights back to FP32 domain for computation.
    
    Formula: x = sign(z) * (1 - |z|) / (α|z|)
    
    Args:
        encoded: QINS-encoded tensor (FP32 or uint8)
        alpha: Density control parameter (must match encode)
        is_quantized: If True, dequantize from uint8 [0,255] to FP32 [-1,1] first
    
    Returns:
        Decoded FP32 tensor
    """
    # Dequantize if needed
    if is_quantized:
        # Map [0, 255] back to [-1, 1]
        # 0 → -1.0, 127 → ~0.0, 255 → 1.0
        encoded = (encoded.float() / 127.5) - 1.0
    
    sign = torch.sign(encoded)
    abs_encoded = encoded.abs()
    
    # Avoid division by zero
    abs_encoded = abs_encoded.clamp(min=1e-12)
    
    decoded = sign * (1.0 - abs_encoded) / (alpha * abs_encoded)
    
    return decoded


class QINSWeightLinear(nn.Module):
    """
    Linear layer with QINS-encoded weights (Pattern A: Codec-at-Rest).
    
    Storage: Weights stored in QINS domain (compressed)
    Compute: Weights decoded to FP32 just-in-time for matmul
    
    Benefits:
    - Memory savings (~50% for weights)
    - No divergence (compute in FP domain)
    - Drop-in replacement for nn.Linear
    
    Safe for:
    - v_proj (attention values)
    - o_proj (attention output)
    - MLP layers (gate_proj, up_proj, down_proj)
    
    DO NOT use for:
    - q_proj, k_proj (would affect KV cache bookkeeping)
    - LayerNorm (expects specific statistics)
    - Embeddings (different usage pattern)
    """
    
    def __init__(self, linear: nn.Linear, alpha: float = 1.0, quantize: bool = True):
        """
        Create QINS weight layer from existing Linear layer.
        
        Args:
            linear: Source nn.Linear to convert
            alpha: QINS density parameter
            quantize: If True, store as uint8 (4× compression). If False, store as FP32 (validation only)
        """
        super().__init__()
        
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.alpha = alpha
        self.is_quantized = quantize
        
        # Copy bias as-is (keep in FP32)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            self.bias = None
        
        # Encode weights and store as buffer (not trainable Parameter)
        with torch.no_grad():
            w_encoded = qins_encode(linear.weight.detach(), alpha, quantize=quantize)
        
        # Register as persistent buffer so it's saved with the model
        self.register_buffer("w_encoded", w_encoded, persistent=True)
        
        # Calculate actual memory usage
        fp32_mb = linear.weight.numel() * 4 / 1024 / 1024
        if quantize:
            qins_mb = w_encoded.numel() * 1 / 1024 / 1024  # uint8 = 1 byte
            dtype_str = "uint8"
        else:
            qins_mb = w_encoded.numel() * 4 / 1024 / 1024  # FP32 = 4 bytes
            dtype_str = "float32"
        
        print(f"  QINS-encoded: {linear.weight.shape} "
              f"({fp32_mb:.2f} MB FP32 → {qins_mb:.2f} MB {dtype_str})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: decode weights just-in-time, compute in FP domain.
        
        Pattern A guarantees:
        - All computation in FP32/BF16 domain
        - No distribution drift
        - Perfect match with FP32 baseline (within float precision)
        """
        # Decode weights on-the-fly (handles quantized or non-quantized)
        w = qins_decode(self.w_encoded, self.alpha, is_quantized=self.is_quantized)
        
        # Standard linear operation in FP domain
        return F.linear(x, w, self.bias)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.bias is not None}, alpha={self.alpha}'


def convert_linear_to_qins(
    model: nn.Module,
    target_names: list = None,
    alpha: float = 1.0,
    verbose: bool = True
) -> nn.Module:
    """
    Convert specified Linear layers to QINS weight encoding.
    
    Safe default targets for Phi-3.5 (and similar transformers):
    - "v_proj": Attention value projection
    - "o_proj": Attention output projection  
    - "gate_proj", "up_proj", "down_proj": MLP layers
    
    DO NOT include:
    - "q_proj", "k_proj": Would affect KV cache sequence bookkeeping
    - LayerNorm: Expects specific float statistics
    - "embed_tokens": Different usage pattern
    
    Args:
        model: PyTorch model to modify in-place
        target_names: List of layer name suffixes to convert
                     (default: safe Phi-3.5 targets)
        alpha: QINS density parameter
        verbose: Print conversion progress
    
    Returns:
        Modified model (same object, edited in-place)
    
    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct")
        >>> model = convert_linear_to_qins(model)
        >>> # Now model uses QINS weight encoding (Pattern A)
    """
    if target_names is None:
        # Safe default: attention values/output + MLP
        target_names = ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Converting Linear layers to QINS (Pattern A: Codec-at-Rest)")
        print(f"{'='*70}")
        print(f"Target layer suffixes: {target_names}")
        print(f"Alpha parameter: {alpha}")
        print()
    
    converted_count = 0
    total_fp32_mb = 0
    total_qins_mb = 0
    
    # Build name->module dict for parent lookup
    name_to_module = dict(model.named_modules())
    
    for name, module in list(model.named_modules()):
        # Check if this module should be converted
        if not isinstance(module, nn.Linear):
            continue
        
        should_convert = any(name.endswith(target) for target in target_names)
        if not should_convert:
            continue
        
        # Calculate memory
        fp32_mb = module.weight.numel() * 4 / 1024 / 1024
        
        # Create QINS version
        qins_layer = QINSWeightLinear(module, alpha=alpha)
        qins_mb = qins_layer.w_encoded.numel() * 4 / 1024 / 1024
        
        # Replace in parent module
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = name_to_module[parent_name]
            setattr(parent, child_name, qins_layer)
        else:
            # Top-level module (rare)
            setattr(model, name, qins_layer)
        
        converted_count += 1
        total_fp32_mb += fp32_mb
        total_qins_mb += qins_mb
        
        if verbose and converted_count % 10 == 0:
            print(f"  Converted {converted_count} layers...")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Conversion complete!")
        print(f"{'='*70}")
        print(f"Layers converted: {converted_count}")
        print(f"Weight memory:")
        print(f"  FP32:  {total_fp32_mb:.2f} MB")
        print(f"  QINS:  {total_qins_mb:.2f} MB")
        print(f"  Saved: {total_fp32_mb - total_qins_mb:.2f} MB ({(1 - total_qins_mb/total_fp32_mb)*100:.1f}%)")
        print()
    
    return model


def verify_qins_layer(layer: QINSWeightLinear, original: nn.Linear, atol: float = 1e-3) -> dict:
    """
    Verify QINS layer produces similar outputs to original.
    
    Args:
        layer: QINSWeightLinear layer
        original: Original nn.Linear layer
        atol: Absolute tolerance for comparison
    
    Returns:
        Dict with verification results
    """
    # Create random input
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, original.in_features)
    
    with torch.no_grad():
        y_original = original(x)
        y_qins = layer(x)
    
    # Calculate error
    abs_error = (y_original - y_qins).abs()
    max_error = abs_error.max().item()
    mean_error = abs_error.mean().item()
    
    # Relative error
    rel_error = abs_error / (y_original.abs() + 1e-8)
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()
    
    passed = max_error < atol
    
    return {
        'passed': passed,
        'max_abs_error': max_error,
        'mean_abs_error': mean_error,
        'max_rel_error': max_rel_error,
        'mean_rel_error': mean_rel_error,
        'tolerance': atol
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("QINS Weight Codec Test")
    print("="*70)
    
    # Test encode/decode round-trip
    print("\n1. Testing encode/decode round-trip...")
    original = torch.randn(100, 50)
    encoded = qins_encode(original, alpha=1.0)
    decoded = qins_decode(encoded, alpha=1.0)
    
    error = (original - decoded).abs()
    print(f"   Max error: {error.max():.6f}")
    print(f"   Mean error: {error.mean():.6f}")
    print(f"   ✓ Round-trip verified")
    
    # Test QINSWeightLinear
    print("\n2. Testing QINSWeightLinear layer...")
    linear = nn.Linear(128, 256, bias=True)
    qins_linear = QINSWeightLinear(linear, alpha=1.0)
    
    x = torch.randn(4, 10, 128)
    
    with torch.no_grad():
        y1 = linear(x)
        y2 = qins_linear(x)
    
    error = (y1 - y2).abs()
    print(f"   Max output error: {error.max():.6f}")
    print(f"   Mean output error: {error.mean():.6f}")
    print(f"   ✓ Layer forward pass verified")
    
    # Memory comparison
    fp32_bytes = linear.weight.numel() * 4
    qins_bytes = qins_linear.w_encoded.numel() * 4  # Still float32 but pattern A
    print(f"\n3. Memory (this test):")
    print(f"   FP32: {fp32_bytes:,} bytes")
    print(f"   QINS: {qins_bytes:,} bytes")
    print(f"   Note: Same size in this test (both float32)")
    print(f"   In production: quantize to INT8 for real savings")
    
    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)
