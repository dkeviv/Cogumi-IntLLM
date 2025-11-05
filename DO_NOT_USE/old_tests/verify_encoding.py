#!/usr/bin/env python3
"""
Verify QINS encoding uses logarithmic + inverse relationship.
"""

import torch
import numpy as np

def verify_qins_encoding(model_path='models/phi35-qins-codec.pt'):
    """Verify the QINS encoding is correct."""
    
    print("=" * 70)
    print("QINS Encoding Verification")
    print("=" * 70)
    print()
    
    # Load model
    print(f"Loading model from {model_path}...")
    model_data = torch.load(model_path, map_location='cpu')
    print(f"✓ Loaded {len(model_data)} tensors")
    print()
    
    # Get first layer's QINS data
    layer_name = 'model.layers.0.self_attn.o_proj'
    stored = model_data[f'{layer_name}.stored']
    sign = model_data[f'{layer_name}.sign']
    log_min = model_data[f'{layer_name}.log_min']
    log_max = model_data[f'{layer_name}.log_max']
    
    print(f"Analyzing layer: {layer_name}")
    print(f"  Shape: {stored.shape}")
    print(f"  Log range: [{log_min:.6f}, {log_max:.6f}]")
    print(f"  Stored values: min={stored.min()}, max={stored.max()}")
    print(f"  Stored unique values: {stored.unique().numel()}")
    print()
    
    # Reconstruct weights
    print("Reconstructing weights...")
    normalized = (255.0 - stored.float()) / 254.0
    log_weight = log_min + normalized * (log_max - log_min)
    abs_weight = torch.exp(log_weight)
    reconstructed = sign.float() * abs_weight
    print("✓ Reconstruction complete")
    print()
    
    # Verify inverse relationship
    print("=" * 70)
    print("INVERSE RELATIONSHIP VERIFICATION")
    print("(Large magnitude should map to small stored value)")
    print("=" * 70)
    print()
    
    flat_stored = stored.flatten()
    flat_reconstructed = reconstructed.flatten()
    
    # Find examples at different stored values
    for stored_val in [1, 64, 128, 192, 255]:
        indices = (flat_stored == stored_val).nonzero(as_tuple=True)[0]
        if len(indices) > 0:
            sample_idx = indices[0]
            magnitude = abs(flat_reconstructed[sample_idx].item())
            weight = flat_reconstructed[sample_idx].item()
            print(f"  stored={stored_val:3d} → magnitude={magnitude:.6f}, weight={weight:+.6f}")
    
    print()
    
    # Statistical verification
    print("=" * 70)
    print("STATISTICAL VERIFICATION")
    print("=" * 70)
    print()
    
    print(f"Reconstructed weight statistics:")
    print(f"  Min: {reconstructed.min():.6f}")
    print(f"  Max: {reconstructed.max():.6f}")
    print(f"  Mean: {reconstructed.mean():.6f}")
    print(f"  Std: {reconstructed.std():.6f}")
    print()
    
    # Check correlation: stored should be NEGATIVELY correlated with magnitude
    abs_recon = reconstructed.abs().flatten()
    correlation = torch.corrcoef(torch.stack([flat_stored.float(), abs_recon]))[0, 1]
    
    print(f"Correlation (stored vs |magnitude|): {correlation:.4f}")
    if correlation < -0.5:
        print("✅ INVERSE relationship CONFIRMED (strong negative correlation)")
        result = "PASS"
    elif correlation > 0.5:
        print("❌ ERROR: POSITIVE correlation detected (wrong encoding!)")
        result = "FAIL"
    else:
        print("⚠️  Weak correlation (unexpected)")
        result = "WARN"
    print()
    
    # Verify encoding formula
    print("=" * 70)
    print("ENCODING FORMULA VERIFICATION")
    print("=" * 70)
    print()
    
    print("Expected formula: stored = 255 - (normalized * 254)")
    print("  where normalized = (log|w| - log_min) / (log_max - log_min)")
    print()
    
    # Test several weights
    test_indices = [0, 100, 1000, 5000]
    all_match = True
    
    for idx in test_indices:
        if idx >= len(flat_reconstructed):
            continue
            
        test_weight = flat_reconstructed[idx].item()
        test_stored = flat_stored[idx].item()
        test_log = np.log(abs(test_weight))
        test_normalized = (test_log - log_min.item()) / (log_max.item() - log_min.item())
        test_stored_calc = 255 - (test_normalized * 254)
        
        match = abs(test_stored - test_stored_calc) < 1.0
        match_str = "✓" if match else "✗"
        
        print(f"  {match_str} weight={test_weight:+.6f} → stored={test_stored} (expected={test_stored_calc:.1f})")
        
        if not match:
            all_match = False
    
    print()
    if all_match:
        print("✅ Logarithmic + inverse encoding formula CONFIRMED")
    else:
        print("❌ Encoding formula mismatch detected!")
        result = "FAIL"
    
    print()
    print("=" * 70)
    print(f"OVERALL RESULT: {result}")
    print("=" * 70)
    
    return result == "PASS"


if __name__ == "__main__":
    import sys
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/phi35-qins-codec.pt'
    
    success = verify_qins_encoding(model_path)
    sys.exit(0 if success else 1)
