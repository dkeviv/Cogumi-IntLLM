"""
QINS Scaling Calibration for Long-Rollout Stability

Adds three components to stabilize distribution statistics:
1. Global logit scale α - matches output logit variance
2. Per-layer feature scaling - calibrates hidden state distributions
3. Selective precision - Q/K in FP16, V/MLP in QINS

No retraining required - all scales computed analytically from FP32 reference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np

class CalibratedProjectiveLinear(nn.Module):
    """
    ProjectiveLinear with distribution calibration
    
    Adds per-channel scaling to match FP32 output statistics:
    y = QINS(x) * scale
    
    Scale computed from FP32 reference (one-time calibration)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        scale: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # QINS storage (2 bytes per weight)
        self.register_buffer('stored', torch.zeros(out_features, in_features, dtype=torch.uint8))
        self.register_buffer('sign', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('log_min', torch.tensor(0.0))
        self.register_buffer('log_max', torch.tensor(0.0))
        
        # Calibration scale (per output channel)
        if scale is not None:
            self.register_buffer('scale', scale)
        else:
            self.register_buffer('scale', torch.ones(out_features))
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Cache for fast decoding
        self._weight_cache = None
        self._log_min_scalar = 0.0
        self._log_max_scalar = 0.0
    
    def _reconstruct_weights(self) -> torch.Tensor:
        """Reconstruct FP32 weights from QINS encoding"""
        # Check cache
        if self._weight_cache is not None:
            return self._weight_cache
        
        # Use pre-computed scalars
        log_min = self._log_min_scalar
        log_max = self._log_max_scalar
        
        # Reverse inverse mapping
        normalized = (255.0 - self.stored.float()) / 254.0
        
        # Map to log space
        log_weight = log_min + normalized * (log_max - log_min)
        
        # Exponentiate and apply sign
        weight = self.sign.float() * torch.exp(log_weight)
        
        # Cache for next forward pass
        self._weight_cache = weight
        
        return weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with calibration
        
        y = (W_qins @ x + b) * scale
        """
        weight = self._reconstruct_weights()
        output = F.linear(x, weight, self.bias)
        
        # Apply per-channel calibration scale
        # Broadcasting: (batch, seq, out_features) * (out_features,)
        output = output * self.scale
        
        return output
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        calibration_input: Optional[torch.Tensor] = None,
        num_calibration_samples: int = 100
    ) -> 'CalibratedProjectiveLinear':
        """
        Convert nn.Linear to CalibratedProjectiveLinear with scaling
        
        Args:
            linear: FP32 linear layer
            calibration_input: Optional input for calibration (batch, seq, in_features)
                             If None, uses random Gaussian samples
            num_calibration_samples: Number of samples for calibration
        
        Returns:
            Calibrated QINS layer with optimal scaling
        """
        in_features = linear.in_features
        out_features = linear.out_features
        has_bias = linear.bias is not None
        
        # Convert weights to QINS
        weight = linear.weight.data
        sign = torch.sign(weight).to(torch.int8)
        sign[sign == 0] = 1
        
        abs_weight = torch.abs(weight).clamp(min=1e-8)
        log_weight = torch.log(abs_weight)
        
        non_zero_mask = torch.abs(weight) > 1e-8
        if non_zero_mask.sum() == 0:
            log_min = torch.tensor(-10.0)
            log_max = torch.tensor(0.0)
            stored = torch.ones_like(weight, dtype=torch.uint8) * 128
        else:
            log_min = log_weight[non_zero_mask].min()
            log_max = log_weight[non_zero_mask].max()
            
            normalized = (log_weight - log_min) / (log_max - log_min + 1e-8)
            stored_float = 255 - (normalized * 254)
            stored = stored_float.round().clamp(1, 255).to(torch.uint8)
        
        # Compute calibration scale
        if calibration_input is None:
            # Use random Gaussian samples (typical activations)
            calibration_input = torch.randn(num_calibration_samples, in_features) * 0.02
        
        with torch.no_grad():
            # FP32 output statistics
            fp32_output = F.linear(calibration_input, weight, None)
            fp32_mean = fp32_output.mean(dim=0)
            fp32_std = fp32_output.std(dim=0)
            
            # QINS output statistics (before scaling)
            # Reconstruct weights
            normalized_recon = (255.0 - stored.float()) / 254.0
            log_weight_recon = log_min + normalized_recon * (log_max - log_min)
            weight_recon = sign.float() * torch.exp(log_weight_recon)
            
            qins_output = F.linear(calibration_input, weight_recon, None)
            qins_mean = qins_output.mean(dim=0)
            qins_std = qins_output.std(dim=0)
            
            # Compute per-channel scale to match FP32 statistics
            # scale = fp32_std / (qins_std + eps)
            scale = fp32_std / (qins_std + 1e-8)
            
            # Clamp scale to reasonable range [0.5, 2.0]
            scale = scale.clamp(0.5, 2.0)
        
        # Create calibrated layer
        layer = cls(in_features, out_features, has_bias, scale=scale)
        
        # Copy QINS parameters
        layer.stored.copy_(stored)
        layer.sign.copy_(sign)
        layer.log_min.fill_(log_min.item())
        layer.log_max.fill_(log_max.item())
        
        # Pre-compute scalars for fast decoding
        layer._log_min_scalar = log_min.item()
        layer._log_max_scalar = log_max.item()
        
        # Copy bias
        if has_bias:
            layer.bias.data.copy_(linear.bias.data)
        
        return layer
    
    def clear_cache(self):
        """Clear weight cache (call when modifying stored/sign)"""
        self._weight_cache = None

class LogitScaler(nn.Module):
    """
    Global logit scaling layer
    
    Matches output logit variance to FP32 reference:
    logits_scaled = logits * alpha
    
    Alpha computed from calibration data
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.register_buffer('alpha', torch.tensor(alpha))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits to match FP32 variance"""
        return logits * self.alpha
    
    @classmethod
    def calibrate(
        cls,
        fp32_logits: torch.Tensor,
        qins_logits: torch.Tensor
    ) -> 'LogitScaler':
        """
        Compute optimal alpha from FP32 vs QINS logits
        
        Args:
            fp32_logits: Reference logits (N, vocab_size)
            qins_logits: QINS logits before scaling (N, vocab_size)
        
        Returns:
            LogitScaler with calibrated alpha
        """
        with torch.no_grad():
            fp32_std = fp32_logits.std()
            qins_std = qins_logits.std()
            
            alpha = fp32_std / (qins_std + 1e-8)
            
            # Clamp to reasonable range
            alpha = alpha.clamp(0.8, 1.2)
        
        return cls(alpha.item())

def convert_model_with_calibration(
    model: nn.Module,
    calibration_data: torch.Tensor,
    selective_qk: bool = True,
    verbose: bool = True
) -> nn.Module:
    """
    Convert model to calibrated QINS with selective precision
    
    Strategy:
    - Q/K projections: Keep FP16 (most drift-sensitive)
    - V projections: Use calibrated QINS
    - MLP projections: Use calibrated QINS
    - Output/logit projection: Use calibrated QINS + LogitScaler
    
    Args:
        model: FP32 model to convert
        calibration_data: Input data for calibration (batch, seq, hidden)
        selective_qk: If True, keep Q/K in FP16, convert V/MLP to QINS
        verbose: Print conversion progress
    
    Returns:
        Converted model with calibration
    """
    if verbose:
        print("=" * 80)
        print("Converting model with calibration")
        print("=" * 80)
    
    converted_count = 0
    skipped_count = 0
    
    # First pass: collect calibration data per layer
    layer_activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(input, tuple):
                layer_activations[name] = input[0].detach()
            else:
                layer_activations[name] = input.detach()
        return hook
    
    # Register hooks
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handle = module.register_forward_hook(hook_fn(name))
            handles.append(handle)
    
    # Run calibration forward pass
    if verbose:
        print(f"\nRunning calibration with {calibration_data.shape[0]} samples...")
    
    with torch.no_grad():
        _ = model(calibration_data)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    if verbose:
        print(f"✅ Collected activations for {len(layer_activations)} layers")
    
    # Second pass: convert layers with calibration
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Decide whether to convert based on layer name
            skip_layer = False
            
            if selective_qk:
                # Keep Q and K projections in FP16
                if any(x in name.lower() for x in ['q_proj', 'k_proj', 'query', 'key']):
                    skip_layer = True
                    if verbose:
                        print(f"  [SKIP] {name} - keeping Q/K in FP16")
            
            if skip_layer:
                skipped_count += 1
                continue
            
            # Get calibration data for this layer
            if name in layer_activations:
                calib_input = layer_activations[name]
                
                # Convert with calibration
                calibrated_layer = CalibratedProjectiveLinear.from_linear(
                    module,
                    calibration_input=calib_input,
                    num_calibration_samples=min(100, calib_input.shape[0])
                )
                
                # Replace in model
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                
                setattr(parent, child_name, calibrated_layer)
                converted_count += 1
                
                if verbose and converted_count % 10 == 0:
                    print(f"  Converted {converted_count} layers...")
    
    if verbose:
        print(f"\n✅ Conversion complete:")
        print(f"   Converted: {converted_count} layers")
        print(f"   Skipped (Q/K): {skipped_count} layers")
        print(f"   Strategy: {'Selective (Q/K FP16)' if selective_qk else 'Full QINS'}")
    
    return model

# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("Testing Calibrated QINS")
    print("=" * 80)
    
    # Create test layer
    in_dim = 512
    out_dim = 512
    batch_size = 50
    
    fp32_layer = nn.Linear(in_dim, out_dim)
    nn.init.normal_(fp32_layer.weight, mean=0, std=0.02)
    
    # Calibration data
    calib_input = torch.randn(batch_size, in_dim) * 0.02
    
    print(f"\n1. Converting with calibration...")
    qins_layer = CalibratedProjectiveLinear.from_linear(
        fp32_layer,
        calibration_input=calib_input
    )
    
    print(f"   ✅ Scale range: [{qins_layer.scale.min():.3f}, {qins_layer.scale.max():.3f}]")
    print(f"   ✅ Scale mean: {qins_layer.scale.mean():.3f}")
    
    # Test on new data
    test_input = torch.randn(100, in_dim) * 0.02
    
    with torch.no_grad():
        fp32_output = fp32_layer(test_input)
        qins_output = qins_layer(test_input)
    
    print(f"\n2. Testing output statistics...")
    print(f"   FP32: mean={fp32_output.mean():.6f}, std={fp32_output.std():.6f}")
    print(f"   QINS: mean={qins_output.mean():.6f}, std={qins_output.std():.6f}")
    
    error = (fp32_output - qins_output).abs().mean()
    print(f"   Error: {error:.6f}")
    
    # Check if calibration improves stability
    print(f"\n3. Checking distribution stability...")
    fp32_std = fp32_output.std(dim=0)
    qins_std = qins_output.std(dim=0)
    
    std_ratio = (qins_std / fp32_std).mean()
    print(f"   Std ratio (QINS/FP32): {std_ratio:.3f}")
    print(f"   {'✅ GOOD' if 0.95 < std_ratio < 1.05 else '⚠️  NEEDS TUNING'}")
    
    print("\n" + "=" * 80)
    print("✅ Calibration test complete!")
    print("=" * 80)
