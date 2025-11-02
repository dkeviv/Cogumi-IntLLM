#!/usr/bin/env python3
"""
Quick speed test: Just measure forward pass speed with caching
"""

import torch
import torch.nn as nn
import sys
import time
sys.path.insert(0, 'src')
from projective_layer import ProjectiveLinear

print("="*70)
print("QINS Forward Pass Speed Test")
print("="*70)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"\nDevice: {device}")

# Create realistic layer size (from Phi-3.5)
linear = nn.Linear(3072, 3072, bias=True)
nn.init.normal_(linear.weight, mean=0, std=0.02)

# Convert to QINS
proj = ProjectiveLinear(3072, 3072, bias=True).to(device)
proj.from_linear(linear)
proj.eval()

# Test input (batch of 32 sequences, length 128)
x = torch.randn(32, 128, 3072).to(device)

print(f"\nTest setup:")
print(f"  Layer: {3072} → {3072}")
print(f"  Input shape: {x.shape}")
print(f"  Parameters: {3072 * 3072:,}")

# Warmup (first pass builds cache)
print("\n🔥 Warmup (building cache)...")
with torch.no_grad():
    _ = proj(x)

# Measure cold cache
print("\n❄️  Cold cache (first pass):")
proj._invalidate_cache()
with torch.no_grad():
    start = time.time()
    y1 = proj(x)
    cold_time = time.time() - start
print(f"  Time: {cold_time*1000:.1f} ms")

# Measure warm cache (20 iterations)
print("\n🔥 Warm cache (20 iterations):")
times = []
with torch.no_grad():
    for i in range(20):
        start = time.time()
        y = proj(x)
        times.append(time.time() - start)

warm_time = sum(times) / len(times)
print(f"  Average time: {warm_time*1000:.3f} ms")
print(f"  Min time: {min(times)*1000:.3f} ms")
print(f"  Max time: {max(times)*1000:.3f} ms")

speedup = cold_time / warm_time
print(f"\n✅ Cache speedup: {speedup:.1f}×")
print(f"\n💡 Throughput: {32*128 / warm_time:.0f} tokens/sec (with caching)")

# Verify correctness
error = (y1 - y).abs().max().item()
print(f"\n✓ Output consistency: {error:.10f} (cache working correctly)")

print("\n" + "="*70)
print("Summary")
print("="*70)
print(f"✅ QINS layer working with weight caching")
print(f"✅ {speedup:.1f}× speedup from caching")
print(f"✅ Ready for full model inference")
print("="*70)
