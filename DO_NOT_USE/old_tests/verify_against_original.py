#!/usr/bin/env python3
"""
Verify QINS conversion against original FP32 model.
Checks BOTH encoding accuracy AND sign preservation.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM

def verify_conversion():
    print('=' * 70)
    print('QINS vs Original FP32 Verification')
    print('=' * 70)
    print()
    
    # Load original FP32 model
    print('📥 Loading original FP32 model...')
    original_model = AutoModelForCausalLM.from_pretrained(
        'microsoft/Phi-3.5-mini-instruct',
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    print('✓ Original model loaded')
    print()
    
    # Load QINS model
    print('📥 Loading QINS model...')
    qins_data = torch.load('models/phi35-qins-codec.pt', map_location='cpu')
    print('✓ QINS model loaded')
    print()
    
    # Test first layer
    layer_name = 'model.layers.0.self_attn.o_proj'
    print(f'Testing layer: {layer_name}')
    print('=' * 70)
    print()
    
    # Get original weights
    original_weight = original_model.model.layers[0].self_attn.o_proj.weight.data
    
    # Get QINS components
    stored = qins_data[f'{layer_name}.stored']
    sign = qins_data[f'{layer_name}.sign']
    log_min = qins_data[f'{layer_name}.log_min']
    log_max = qins_data[f'{layer_name}.log_max']
    
    # Reconstruct from QINS
    normalized = (255.0 - stored.float()) / 254.0
    log_weight = log_min + normalized * (log_max - log_min)
    abs_weight = torch.exp(log_weight)
    reconstructed = sign.float() * abs_weight
    
    print('1️⃣  SIGN PRESERVATION CHECK')
    print('-' * 70)
    
    # Compare signs
    original_signs = torch.sign(original_weight)
    qins_signs = sign.float()
    
    sign_matches = (original_signs == qins_signs).sum().item()
    total = original_signs.numel()
    sign_match_rate = sign_matches / total
    
    print(f'Sign match: {sign_matches:,} / {total:,} ({sign_match_rate * 100:.4f}%)')
    
    if sign_match_rate == 1.0:
        print('✅ ALL SIGNS PRESERVED PERFECTLY!')
    else:
        print(f'❌ {total - sign_matches:,} sign mismatches!')
        
        # Show examples of mismatches
        mismatch_mask = original_signs != qins_signs
        mismatch_indices = mismatch_mask.nonzero(as_tuple=True)
        
        print('\nFirst 5 mismatches:')
        for idx in range(min(5, len(mismatch_indices[0]))):
            i, j = mismatch_indices[0][idx].item(), mismatch_indices[1][idx].item()
            orig = original_weight[i, j].item()
            stored_sign = sign[i, j].item()
            print(f'  [{i},{j}] Original: {orig:+.6f} → QINS sign: {stored_sign:+2d}')
    
    print()
    print('2️⃣  MAGNITUDE ENCODING CHECK')
    print('-' * 70)
    
    # Compare magnitudes
    abs_error = (original_weight - reconstructed).abs()
    rel_error = abs_error / (original_weight.abs() + 1e-8)
    
    print(f'Absolute error:')
    print(f'  Mean: {abs_error.mean():.6f}')
    print(f'  Max:  {abs_error.max():.6f}')
    print(f'  Std:  {abs_error.std():.6f}')
    print()
    
    print(f'Relative error:')
    print(f'  Mean: {rel_error.mean():.4%}')
    print(f'  Max:  {rel_error.max():.4%}')
    print(f'  Median: {rel_error.median():.4%}')
    
    if rel_error.mean() < 0.02:  # <2% error
        print('✅ Encoding accuracy is EXCELLENT')
    elif rel_error.mean() < 0.05:  # <5% error
        print('✅ Encoding accuracy is GOOD')
    else:
        print('⚠️  Encoding accuracy needs improvement')
    
    print()
    print('3️⃣  SPOT CHECK: Sample Weights')
    print('-' * 70)
    
    # Check specific positions
    test_positions = [(0, 0), (0, 100), (10, 50), (100, 200), (500, 1000)]
    
    print('Position | Original      | Reconstructed | Error')
    print('-' * 70)
    
    for i, j in test_positions:
        orig = original_weight[i, j].item()
        recon = reconstructed[i, j].item()
        error = abs(orig - recon)
        rel_err = error / (abs(orig) + 1e-8)
        
        sign_match = '✓' if (orig > 0) == (recon > 0) else '✗'
        
        print(f'[{i:3d},{j:4d}] {sign_match} {orig:+.6f}   {recon:+.6f}   {rel_err:.2%}')
    
    print()
    print('4️⃣  INVERSE RELATIONSHIP CHECK')
    print('-' * 70)
    
    # Verify inverse: large magnitude → small stored value
    flat_stored = stored.flatten()
    flat_abs_orig = original_weight.abs().flatten()
    
    correlation = torch.corrcoef(torch.stack([flat_stored.float(), flat_abs_orig]))[0, 1]
    
    print(f'Correlation (stored vs |magnitude|): {correlation:.4f}')
    
    if correlation < -0.5:
        print('✅ INVERSE relationship confirmed')
    else:
        print('❌ Inverse relationship NOT found')
    
    print()
    
    # Show examples
    print('Examples:')
    
    # Find largest, medium, smallest magnitude weights
    sorted_indices = torch.argsort(flat_abs_orig, descending=True)
    
    for desc, idx in [('Largest', sorted_indices[0]), 
                       ('Medium', sorted_indices[len(sorted_indices)//2]),
                       ('Smallest', sorted_indices[-1])]:
        mag = flat_abs_orig[idx].item()
        stored_val = flat_stored[idx].item()
        print(f'  {desc:8s} magnitude={mag:.6f} → stored={stored_val}')
    
    print()
    print('=' * 70)
    print('OVERALL VERDICT')
    print('=' * 70)
    
    if sign_match_rate == 1.0 and rel_error.mean() < 0.02 and correlation < -0.5:
        print('✅ CONVERSION PERFECT!')
        print('   - Signs: 100% preserved')
        print('   - Accuracy: <2% error')
        print('   - Encoding: Inverse logarithmic confirmed')
        return True
    else:
        print('⚠️  CONVERSION HAS ISSUES')
        if sign_match_rate < 1.0:
            print(f'   ❌ Signs: {(1-sign_match_rate)*100:.2f}% mismatch')
        if rel_error.mean() >= 0.02:
            print(f'   ❌ Accuracy: {rel_error.mean():.2%} mean error')
        if correlation >= -0.5:
            print(f'   ❌ Encoding: Correlation {correlation:.4f} (wrong)')
        return False

if __name__ == '__main__':
    import sys
    success = verify_conversion()
    sys.exit(0 if success else 1)
