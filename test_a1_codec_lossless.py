#!/usr/bin/env python3
"""
Phase A - Step A1: Prove Codec is Lossless
===========================================

Goal: Ensure encode→decode round-trips with minimal error
Success Criteria: 
  - Cosine similarity ≥ 0.999999
  - Max absolute error ≤ 1e-6
  
Scope: QINS encode/decode functions only
Validation: Round-trip stats on various tensor distributions
"""

import torch
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.qins_codec import QINSCodec


def test_round_trip(tensor: torch.Tensor, name: str) -> dict:
    """
    Test encode→decode round-trip accuracy.
    
    Args:
        tensor: Input tensor to test
        name: Description of tensor distribution
        
    Returns:
        Dictionary with accuracy metrics
    """
    # Encode
    stored, sign, log_min, log_max = QINSCodec.encode(tensor)
    
    # Decode
    reconstructed = QINSCodec.decode(stored, sign, log_min, log_max)
    
    # Calculate metrics
    abs_error = (tensor - reconstructed).abs()
    rel_error = abs_error / (tensor.abs() + 1e-10)
    
    # Cosine similarity
    tensor_flat = tensor.flatten()
    recon_flat = reconstructed.flatten()
    cosine_sim = torch.nn.functional.cosine_similarity(
        tensor_flat.unsqueeze(0),
        recon_flat.unsqueeze(0)
    ).item()
    
    results = {
        'name': name,
        'shape': tuple(tensor.shape),
        'original_min': tensor.min().item(),
        'original_max': tensor.max().item(),
        'original_mean': tensor.mean().item(),
        'original_std': tensor.std().item(),
        'max_abs_error': abs_error.max().item(),
        'mean_abs_error': abs_error.mean().item(),
        'max_rel_error': rel_error.max().item(),
        'mean_rel_error': rel_error.mean().item(),
        'cosine_similarity': cosine_sim,
        'stored_min': stored.min().item(),
        'stored_max': stored.max().item(),
        'pass_cosine': cosine_sim >= 0.999999,
        'pass_max_error': abs_error.max().item() <= 1e-6
    }
    
    return results


def print_results(results: dict):
    """Pretty print test results."""
    print(f"\n{'='*70}")
    print(f"Test: {results['name']}")
    print(f"{'='*70}")
    print(f"Shape: {results['shape']}")
    print(f"\nOriginal tensor stats:")
    print(f"  Min:  {results['original_min']:12.6e}")
    print(f"  Max:  {results['original_max']:12.6e}")
    print(f"  Mean: {results['original_mean']:12.6e}")
    print(f"  Std:  {results['original_std']:12.6e}")
    
    print(f"\nStored values (uint8):")
    print(f"  Min:  {results['stored_min']}")
    print(f"  Max:  {results['stored_max']}")
    
    print(f"\nRound-trip accuracy:")
    print(f"  Cosine similarity:    {results['cosine_similarity']:.12f}")
    print(f"  Max absolute error:   {results['max_abs_error']:.6e}")
    print(f"  Mean absolute error:  {results['mean_abs_error']:.6e}")
    print(f"  Max relative error:   {results['max_rel_error']:.6e}")
    print(f"  Mean relative error:  {results['mean_rel_error']:.6e}")
    
    print(f"\nSuccess criteria:")
    cosine_check = "✅ PASS" if results['pass_cosine'] else "❌ FAIL"
    error_check = "✅ PASS" if results['pass_max_error'] else "❌ FAIL"
    print(f"  Cosine ≥ 0.999999:    {cosine_check} ({results['cosine_similarity']:.12f})")
    print(f"  Max error ≤ 1e-6:     {error_check} ({results['max_abs_error']:.6e})")
    
    overall = "✅ PASS" if (results['pass_cosine'] and results['pass_max_error']) else "❌ FAIL"
    print(f"\nOverall: {overall}")


def main():
    """Run A1 validation tests."""
    print("="*70)
    print("Phase A - Step A1: Codec Lossless Validation")
    print("="*70)
    print("\nTesting QINS encode→decode round-trip accuracy")
    print("Implementation: src/qins_codec.py (logarithmic encoding)")
    
    all_results = []
    
    # Test 1: Standard normal distribution (typical NN weights)
    print("\n" + "="*70)
    print("Test 1: Standard Normal Distribution")
    print("="*70)
    tensor = torch.randn(256, 256) * 0.02  # Typical weight init scale
    results = test_round_trip(tensor, "Standard Normal (σ=0.02)")
    print_results(results)
    all_results.append(results)
    
    # Test 2: Uniform distribution
    print("\n" + "="*70)
    print("Test 2: Uniform Distribution")
    print("="*70)
    tensor = torch.rand(128, 512) * 0.1 - 0.05  # [-0.05, 0.05]
    results = test_round_trip(tensor, "Uniform [-0.05, 0.05]")
    print_results(results)
    all_results.append(results)
    
    # Test 3: Wide range (mix of large and small values)
    print("\n" + "="*70)
    print("Test 3: Wide Dynamic Range")
    print("="*70)
    tensor = torch.randn(64, 1024) * 0.5  # Wider range
    results = test_round_trip(tensor, "Wide Range (σ=0.5)")
    print_results(results)
    all_results.append(results)
    
    # Test 4: Small values (numerical stability test)
    print("\n" + "="*70)
    print("Test 4: Small Values (Numerical Stability)")
    print("="*70)
    tensor = torch.randn(512, 128) * 0.001  # Very small
    results = test_round_trip(tensor, "Small Values (σ=0.001)")
    print_results(results)
    all_results.append(results)
    
    # Test 5: Attention-like distribution (concentrated around 0)
    print("\n" + "="*70)
    print("Test 5: Attention-like Distribution")
    print("="*70)
    tensor = torch.randn(1024, 64) * 0.01  # Attention weights typical scale
    results = test_round_trip(tensor, "Attention-like (σ=0.01)")
    print_results(results)
    all_results.append(results)
    
    # Test 6: MLP weights (larger magnitude)
    print("\n" + "="*70)
    print("Test 6: MLP-like Distribution")
    print("="*70)
    tensor = torch.randn(2048, 512) * 0.1  # MLP typical scale
    results = test_round_trip(tensor, "MLP-like (σ=0.1)")
    print_results(results)
    all_results.append(results)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY - Phase A Step A1")
    print("="*70)
    
    all_pass = all(r['pass_cosine'] and r['pass_max_error'] for r in all_results)
    
    print(f"\nTests run: {len(all_results)}")
    print(f"Tests passed: {sum(r['pass_cosine'] and r['pass_max_error'] for r in all_results)}")
    
    print("\nPer-test results:")
    for r in all_results:
        status = "✅" if (r['pass_cosine'] and r['pass_max_error']) else "❌"
        print(f"  {status} {r['name']:40s} (cosine: {r['cosine_similarity']:.12f}, max_err: {r['max_abs_error']:.6e})")
    
    print("\n" + "="*70)
    if all_pass:
        print("✅ A1 VALIDATION PASSED")
        print("="*70)
        print("\nCodec is LOSSLESS within tolerance:")
        print("  ✓ All tests: cosine similarity ≥ 0.999999")
        print("  ✓ All tests: max absolute error ≤ 1e-6")
        print("\nReady to proceed to A2 (Greedy Parity on Phi-3.5)")
    else:
        print("❌ A1 VALIDATION FAILED")
        print("="*70)
        print("\nSome tests did not meet criteria.")
        print("Action required: Review encoding parameters or implementation")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
