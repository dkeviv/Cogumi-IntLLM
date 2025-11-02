#!/usr/bin/env python3
"""
Simple benchmark: FP32 vs QINS forward pass speed
No generation - just measure inference speed
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
import time
import psutil
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from projective_layer import ProjectiveLinear
import torch.nn as nn

def get_memory_mb():
    """Get current process memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def get_model_size_mb(model):
    """Calculate model parameter memory in MB."""
    total_params = 0
    for param in model.parameters():
        total_params += param.numel() * param.element_size()
    for buffer in model.buffers():
        total_params += buffer.numel() * buffer.element_size()
    return total_params / (1024 * 1024)

def convert_model_to_qins(model):
    """Convert model's Linear layers to ProjectiveLinear."""
    count = 0
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                proj = ProjectiveLinear(
                    child.in_features,
                    child.out_features,
                    bias=(child.bias is not None)
                )
                proj.from_linear(child)
                setattr(module, child_name, proj)
                count += 1
                if count % 10 == 0:
                    print(f"  Converted {count} layers...", end='\r')
    print(f"  Converted {count} layers total         ")
    return model

def benchmark_forward_pass(model, tokenizer, device, num_runs=10):
    """Benchmark forward pass speed (no generation)."""
    prompt = "The future of artificial intelligence is bright and full of possibilities that will transform"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Warmup (builds cache for QINS)
    with torch.no_grad():
        for _ in range(3):
            _ = model(**inputs)
    
    # Benchmark
    times = []
    for run in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        elapsed = time.time() - start_time
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    tokens_per_sec = inputs['input_ids'].shape[1] / avg_time
    
    return {
        'avg_time': avg_time,
        'tokens_per_sec': tokens_per_sec,
        'runs': num_runs
    }

print("="*70)
print("QINS vs FP32 Forward Pass Benchmark - Phi-3.5-mini")
print("="*70)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"\nDevice: {device}")
print(f"Initial memory: {get_memory_mb():.2f} MB")

# Load tokenizer
print("\n📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
tokenizer.pad_token = tokenizer.eos_token

# ============================================================================
# TEST 1: FP32 Model
# ============================================================================
print("\n" + "="*70)
print("TEST 1: FP32 Model (Baseline)")
print("="*70)

gc.collect()
if device == "mps":
    torch.mps.empty_cache()
time.sleep(2)
mem_before_fp32 = get_memory_mb()

print("Loading FP32 model...")
start_load = time.time()
model_fp32 = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    torch_dtype=torch.float32,
    trust_remote_code=True
)
model_fp32 = model_fp32.to(device)
model_fp32.eval()
load_time_fp32 = time.time() - start_load

time.sleep(1)
mem_after_fp32 = get_memory_mb()
model_size_fp32 = get_model_size_mb(model_fp32)

print(f"\n✓ FP32 Model loaded in {load_time_fp32:.2f}s")
print(f"  Model size: {model_size_fp32:.2f} MB")
print(f"  Memory used: {mem_after_fp32 - mem_before_fp32:.2f} MB")

print(f"\nBenchmarking FP32 forward pass...")
bench_fp32 = benchmark_forward_pass(model_fp32, tokenizer, device, num_runs=10)

print(f"\nFP32 Performance:")
print(f"  Average time: {bench_fp32['avg_time']*1000:.1f} ms/forward")
print(f"  Throughput: {bench_fp32['tokens_per_sec']:.1f} tokens/sec")

# Save baseline
baseline_time = bench_fp32['avg_time']
baseline_speed = bench_fp32['tokens_per_sec']
baseline_memory = mem_after_fp32 - mem_before_fp32

# Clean up
del model_fp32
gc.collect()
if device == "mps":
    torch.mps.empty_cache()
time.sleep(3)

# ============================================================================
# TEST 2: QINS Model
# ============================================================================
print("\n" + "="*70)
print("TEST 2: QINS Model (INT8 Quantized)")
print("="*70)

gc.collect()
if device == "mps":
    torch.mps.empty_cache()
time.sleep(2)
mem_before_qins = get_memory_mb()

print("Loading FP32 model...")
start_load = time.time()
model_qins = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    torch_dtype=torch.float32,
    trust_remote_code=True
)

print("\nConverting to QINS...")
model_qins = convert_model_to_qins(model_qins)
load_time_qins = time.time() - start_load

model_qins = model_qins.to(device)
model_qins.eval()

time.sleep(1)
mem_after_qins = get_memory_mb()
model_size_qins = get_model_size_mb(model_qins)

print(f"\n✓ QINS Model ready in {load_time_qins:.2f}s")
print(f"  Model size: {model_size_qins:.2f} MB")
print(f"  Memory used: {mem_after_qins - mem_before_qins:.2f} MB")

print(f"\nBenchmarking QINS forward pass...")
bench_qins = benchmark_forward_pass(model_qins, tokenizer, device, num_runs=10)

print(f"\nQINS Performance:")
print(f"  Average time: {bench_qins['avg_time']*1000:.1f} ms/forward")
print(f"  Throughput: {bench_qins['tokens_per_sec']:.1f} tokens/sec")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*70)
print("COMPARISON RESULTS")
print("="*70)

memory_reduction = baseline_memory / (mem_after_qins - mem_before_qins)
model_size_reduction = model_size_fp32 / model_size_qins
speed_ratio = bench_qins['tokens_per_sec'] / baseline_speed

print(f"\n📊 Memory:")
print(f"  FP32:  {baseline_memory:.2f} MB")
print(f"  QINS:  {mem_after_qins - mem_before_qins:.2f} MB")
print(f"  Reduction: {memory_reduction:.2f}×")

print(f"\n⚡ Speed:")
print(f"  FP32:  {baseline_speed:.1f} tokens/sec")
print(f"  QINS:  {bench_qins['tokens_per_sec']:.1f} tokens/sec")
if speed_ratio >= 1.0:
    print(f"  QINS is {speed_ratio:.2f}× FASTER ✅")
else:
    print(f"  QINS is {1/speed_ratio:.2f}× slower ({(1-speed_ratio)*100:.1f}% slowdown)")

print(f"\n📈 Model Size:")
print(f"  FP32:  {model_size_fp32:.2f} MB")
print(f"  QINS:  {model_size_qins:.2f} MB")
print(f"  Reduction: {model_size_reduction:.2f}×")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✅ QINS provides {model_size_reduction:.2f}× model size reduction")
print(f"✅ QINS uses {memory_reduction:.2f}× less runtime memory")
if speed_ratio >= 0.9:
    print(f"✅ QINS speed is comparable to FP32 ({speed_ratio:.2f}×)")
else:
    print(f"⚠️  QINS is slower than FP32 ({speed_ratio:.2f}×)")
print("="*70)
