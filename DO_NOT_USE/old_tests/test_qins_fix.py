#!/usr/bin/env python3
"""
Quick test to verify QINS logarithmic encoding
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'src')
from projective_layer import convert_to_qins, reconstruct_from_qins

if __name__ == "__main__":
    # Test conversion
    print("=" * 70)
    print("QINS Conversion Test")
    print("=" * 70)
    
    original_weight = torch.randn(100, 50) * 0.1  # Typical NN weights
    
    print("\n📊 Original weight stats:")
    print(f"  Min: {original_weight.min():.6f}")
    print(f"  Max: {original_weight.max():.6f}")
    print(f"  Mean: {original_weight.mean():.6f}")
    print(f"  Std: {original_weight.std():.6f}")
    
    # Convert
    print("\n🔄 Converting to QINS...")
    stored, sign, log_min, log_max = convert_to_qins(original_weight)
    
    print(f"\n📦 Stored value stats:")
    print(f"  Min: {stored.min()}")
    print(f"  Max: {stored.max()}")
    print(f"  Mean: {stored.float().mean():.1f}")
    print(f"  Unique values: {stored.unique().numel()}")
    print(f"  Log range: [{log_min:.4f}, {log_max:.4f}]")
    
    print(f"\n🔖 Sign stats:")
    print(f"  Positive weights: {(sign == 1).sum().item()}")
    print(f"  Negative weights: {(sign == -1).sum().item()}")
    
    # Reconstruct first (needed for verification)
    print("\n🔄 Reconstructing from QINS...")
    reconstructed_weight = reconstruct_from_qins(stored, sign, log_min, log_max)
    
    # Check inverse relationship with signs
    print(f"\n✅ Verifying inverse relationship (with signs):")
    
    # Find indices
    abs_weights = original_weight.abs()
    max_idx = abs_weights.argmax()
    min_idx = abs_weights.argmin()
    
    # Get a medium value
    sorted_abs = abs_weights.flatten().sort()[0]
    median_idx = (abs_weights.flatten() == sorted_abs[len(sorted_abs)//2]).nonzero()[0][0]
    
    def show_weight_conversion(idx, label):
        orig_w = original_weight.flatten()[idx].item()
        stored_val = stored.flatten()[idx].item()
        sign_val = sign.flatten()[idx].item()
        reconstructed_w = reconstructed_weight.flatten()[idx].item()
        
        # Check if signs match
        orig_sign = '+1' if orig_w >= 0 else '-1'
        stored_sign = '+1' if sign_val > 0 else '-1'
        recon_sign = '+1' if reconstructed_w >= 0 else '-1'
        sign_preserved = (orig_sign == stored_sign == recon_sign)
        sign_check = "✓" if sign_preserved else "✗"
        
        print(f"  {label}:")
        print(f"    Original weight: {orig_w:+.6f}")
        print(f"    Sign stored:     {stored_sign}")
        print(f"    Stored value:    {stored_val:3d}  (magnitude encoding)")
        print(f"    Reconstructed:   {reconstructed_w:+.6f}  {sign_check} (sign {'preserved' if sign_preserved else 'CHANGED'})")
        print(f"    Error:           {abs(orig_w - reconstructed_w):.6f}")
    
    print(f"  (Inverse: Large |weight| → Small stored, Small |weight| → Large stored)\n")
    show_weight_conversion(max_idx, "Largest |weight| (should → stored ≈ 1)")
    show_weight_conversion(median_idx, "Medium |weight|")
    show_weight_conversion(min_idx, "Smallest |weight| (should → stored ≈ 255)")
    
    print(f"\n📊 Reconstructed weight stats:")
    print(f"  Min: {reconstructed_weight.min():.6f}")
    print(f"  Max: {reconstructed_weight.max():.6f}")
    print(f"  Mean: {reconstructed_weight.mean():.6f}")
    print(f"  Std: {reconstructed_weight.std():.6f}")
    
    # Error analysis
    abs_error = (original_weight - reconstructed_weight).abs()
    rel_error = abs_error / (original_weight.abs() + 1e-8)
    
    print(f"\n📉 Conversion error:")
    print(f"  Mean absolute error: {abs_error.mean():.6f}")
    print(f"  Max absolute error: {abs_error.max():.6f}")
    print(f"  Mean relative error: {rel_error.mean():.4%}")
    print(f"  Max relative error: {rel_error.max():.4%}")
    
    # Verify signs are preserved
    sign_match = ((original_weight >= 0) == (reconstructed_weight >= 0))
    print(f"\n✓ Signs preserved: {sign_match.sum().item()} / {sign_match.numel()} ({100*sign_match.float().mean():.1f}%)")
    
    # Memory savings
    fp32_bytes = original_weight.numel() * 4
    qins_bytes = stored.numel() * 1 + sign.numel() * 1 + 8  # +8 for log_min/log_max
    
    print(f"\n💾 Memory usage:")
    print(f"  FP32: {fp32_bytes:,} bytes")
    print(f"  QINS: {qins_bytes:,} bytes (stored: {stored.numel():,} + sign: {sign.numel():,} + metadata: 8)")
    print(f"  Compression: {fp32_bytes/qins_bytes:.2f}×")
    
    print("\n" + "=" * 70)
    print("✓ Test complete!")
    print("=" * 70)
