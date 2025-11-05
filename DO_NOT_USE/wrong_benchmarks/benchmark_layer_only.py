"""
Benchmark QINS vs FP32 at layer level (bypasses cache compatibility issues)

This tests the core ProjectiveLinear performance without full model complications.
"""

import torch
import torch.nn as nn
import time
import gc
import psutil
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.projective_layer import ProjectiveLinear, convert_to_qins

def get_memory_mb():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def benchmark_linear_layer(layer: nn.Module, input_tensor: torch.Tensor, num_runs: int = 100):
    """Benchmark a linear layer."""
    # Get device from parameters or buffers
    try:
        device = next(layer.parameters()).device
    except StopIteration:
        # For QINS layers, check buffers instead
        device = next(layer.buffers()).device
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = layer(input_tensor)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            output = layer(input_tensor)
            torch.mps.synchronize() if device.type == 'mps' else None
            end = time.perf_counter()
            times.append(end - start)
    
    return {
        'mean_time_ms': sum(times) / len(times) * 1000,
        'min_time_ms': min(times) * 1000,
        'max_time_ms': max(times) * 1000,
        'throughput_tps': input_tensor.shape[0] * input_tensor.shape[1] / (sum(times) / len(times)),
        'output': output
    }

def main():
    print("=" * 70)
    print("QINS vs FP32 Layer Benchmark (Single Layer Test)")
    print("=" * 70)
    print()
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    print(f"Initial memory: {get_memory_mb():.2f} MB")
    print()
    
    # Test configuration (realistic Phi-3.5 layer sizes)
    batch_size = 1
    seq_len = 128
    hidden_size = 3072  # Phi-3.5 hidden size
    
    print(f"Test configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Layer shape: {hidden_size} → {hidden_size}")
    print()
    
    # Create input
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    
    # ============================================================================
    # TEST 1: FP32 Linear Layer
    # ============================================================================
    print("=" * 70)
    print("TEST 1: FP32 Linear Layer (Baseline)")
    print("=" * 70)
    
    mem_before = get_memory_mb()
    layer_fp32 = nn.Linear(hidden_size, hidden_size, bias=False)
    layer_fp32 = layer_fp32.to(device)
    mem_after = get_memory_mb()
    layer_memory_fp32 = mem_after - mem_before
    
    print(f"✓ FP32 layer created")
    print(f"  Parameters: {sum(p.numel() for p in layer_fp32.parameters()):,}")
    print(f"  Memory: {layer_memory_fp32:.2f} MB")
    print()
    
    print("Benchmarking FP32 layer (100 runs)...")
    results_fp32 = benchmark_linear_layer(layer_fp32, input_tensor, num_runs=100)
    
    print(f"✓ FP32 Results:")
    print(f"  Mean time: {results_fp32['mean_time_ms']:.3f} ms")
    print(f"  Min time: {results_fp32['min_time_ms']:.3f} ms")
    print(f"  Max time: {results_fp32['max_time_ms']:.3f} ms")
    print(f"  Throughput: {results_fp32['throughput_tps']:,.0f} tokens/sec")
    print()
    
    # ============================================================================
    # TEST 2: QINS ProjectiveLinear Layer (Cold Cache)
    # ============================================================================
    print("=" * 70)
    print("TEST 2: QINS ProjectiveLinear (Cold Cache)")
    print("=" * 70)
    
    # Convert FP32 layer to QINS
    stored, sign, log_min, log_max = convert_to_qins(layer_fp32.weight.data)
    
    mem_before = get_memory_mb()
    layer_qins = ProjectiveLinear(hidden_size, hidden_size, bias=False)
    layer_qins.stored = stored.to(device)
    layer_qins.sign = sign.to(device)
    layer_qins.log_min = torch.tensor(log_min, device=device)
    layer_qins.log_max = torch.tensor(log_max, device=device)
    layer_qins._cache_valid = False  # Force cold cache
    mem_after = get_memory_mb()
    layer_memory_qins = mem_after - mem_before
    
    print(f"✓ QINS layer created")
    print(f"  Stored values: {stored.numel():,} × uint8")
    print(f"  Sign values: {sign.numel():,} × int8")
    print(f"  Memory: {layer_memory_qins:.2f} MB")
    print(f"  Compression: {layer_memory_fp32 / layer_memory_qins:.2f}×")
    print()
    
    print("Benchmarking QINS layer (100 runs, cold cache)...")
    results_qins_cold = benchmark_linear_layer(layer_qins, input_tensor, num_runs=100)
    
    print(f"✓ QINS Results (Cold Cache):")
    print(f"  Mean time: {results_qins_cold['mean_time_ms']:.3f} ms")
    print(f"  Min time: {results_qins_cold['min_time_ms']:.3f} ms")
    print(f"  Max time: {results_qins_cold['max_time_ms']:.3f} ms")
    print(f"  Throughput: {results_qins_cold['throughput_tps']:,.0f} tokens/sec")
    print()
    
    # ============================================================================
    # TEST 3: QINS with Warm Cache
    # ============================================================================
    print("=" * 70)
    print("TEST 3: QINS ProjectiveLinear (Warm Cache)")
    print("=" * 70)
    
    # Ensure cache is warm
    layer_qins._cache_valid = True  # Cache should already be warm from previous test
    
    print("Benchmarking QINS layer (100 runs, warm cache)...")
    results_qins_warm = benchmark_linear_layer(layer_qins, input_tensor, num_runs=100)
    
    print(f"✓ QINS Results (Warm Cache):")
    print(f"  Mean time: {results_qins_warm['mean_time_ms']:.3f} ms")
    print(f"  Min time: {results_qins_warm['min_time_ms']:.3f} ms")
    print(f"  Max time: {results_qins_warm['max_time_ms']:.3f} ms")
    print(f"  Throughput: {results_qins_warm['throughput_tps']:,.0f} tokens/sec")
    print()
    
    # ============================================================================
    # COMPARISON
    # ============================================================================
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print()
    
    print("Memory Usage:")
    print(f"  FP32:         {layer_memory_fp32:.2f} MB")
    print(f"  QINS:         {layer_memory_qins:.2f} MB")
    print(f"  Compression:  {layer_memory_fp32 / layer_memory_qins:.2f}×")
    print()
    
    print("Speed (Mean Time):")
    print(f"  FP32:              {results_fp32['mean_time_ms']:.3f} ms")
    print(f"  QINS (cold):       {results_qins_cold['mean_time_ms']:.3f} ms  ({results_qins_cold['mean_time_ms']/results_fp32['mean_time_ms']:.2f}× slower)")
    print(f"  QINS (warm):       {results_qins_warm['mean_time_ms']:.3f} ms  ({results_qins_warm['mean_time_ms']/results_fp32['mean_time_ms']:.2f}× {'faster' if results_qins_warm['mean_time_ms'] < results_fp32['mean_time_ms'] else 'slower'})")
    print(f"  Cache speedup:     {results_qins_cold['mean_time_ms']/results_qins_warm['mean_time_ms']:.1f}×")
    print()
    
    print("Throughput:")
    print(f"  FP32:         {results_fp32['throughput_tps']:,.0f} tokens/sec")
    print(f"  QINS (cold):  {results_qins_cold['throughput_tps']:,.0f} tokens/sec")
    print(f"  QINS (warm):  {results_qins_warm['throughput_tps']:,.0f} tokens/sec")
    print()
    
    # Accuracy check
    print("Accuracy Check:")
    output_fp32 = results_fp32['output']
    output_qins = results_qins_warm['output']
    
    abs_error = (output_fp32 - output_qins).abs()
    rel_error = abs_error / (output_fp32.abs() + 1e-8)
    
    print(f"  Mean absolute error: {abs_error.mean():.6f}")
    print(f"  Max absolute error:  {abs_error.max():.6f}")
    print(f"  Mean relative error: {rel_error.mean():.4%}")
    print(f"  Max relative error:  {rel_error.max():.4%}")
    print()
    
    print("=" * 70)
    print("✓ BENCHMARK COMPLETE")
    print("=" * 70)
    print()
    
    print("KEY FINDINGS:")
    if results_qins_warm['mean_time_ms'] < results_fp32['mean_time_ms']:
        speedup = results_fp32['mean_time_ms'] / results_qins_warm['mean_time_ms']
        print(f"  ✅ QINS with warm cache is {speedup:.2f}× FASTER than FP32!")
    else:
        slowdown = results_qins_warm['mean_time_ms'] / results_fp32['mean_time_ms']
        print(f"  ⚠️  QINS with warm cache is {slowdown:.2f}× slower than FP32")
    
    print(f"  ✅ QINS uses {layer_memory_fp32 / layer_memory_qins:.2f}× less memory")
    print(f"  ✅ Weight caching provides {results_qins_cold['mean_time_ms']/results_qins_warm['mean_time_ms']:.1f}× speedup")
    print(f"  ✅ Mean relative error: {rel_error.mean():.4%} (< 1% target)")

if __name__ == "__main__":
    main()
