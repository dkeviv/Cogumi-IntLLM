#!/usr/bin/env python3
"""
Speed-optimized QINS inference with KV-cache
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from projective_layer import ProjectiveLinear
import torch.nn as nn

def convert_to_qins(model):
    """Convert model to QINS."""
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
    return model, count

print("="*70)
print("Speed Optimization Test: KV-Cache vs No Cache")
print("="*70)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"\nDevice: {device}")

# Load model and tokenizer
print("\n📥 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    torch_dtype=torch.float32,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
tokenizer.pad_token = tokenizer.eos_token

print("Converting to QINS...")
model, num_layers = convert_to_qins(model)
print(f"✓ Converted {num_layers} layers")

model.to(device)
model.eval()

# Test prompt
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
num_tokens = 30

print(f"\nPrompt: '{prompt}'")
print(f"Generating {num_tokens} tokens...\n")

# ============================================================================
# Test 1: WITHOUT KV-cache (current implementation)
# ============================================================================
print("="*70)
print("TEST 1: Without KV-Cache (use_cache=False)")
print("="*70)

input_ids = inputs['input_ids'].clone()
start_time = time.time()

with torch.no_grad():
    for i in range(num_tokens):
        # Recompute everything each time
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits / 0.7, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        input_ids = torch.cat([input_ids, next_token], dim=-1)

elapsed_no_cache = time.time() - start_time
tokens_no_cache = input_ids.shape[1] - inputs['input_ids'].shape[1]
speed_no_cache = tokens_no_cache / elapsed_no_cache

print(f"\n✓ Generated {tokens_no_cache} tokens in {elapsed_no_cache:.2f}s")
print(f"  Speed: {speed_no_cache:.2f} tokens/sec")
print(f"  Latency: {(elapsed_no_cache/tokens_no_cache)*1000:.1f} ms/token")
print(f"\nOutput: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")

# ============================================================================
# Test 2: WITH KV-cache (optimized)
# ============================================================================
print("\n" + "="*70)
print("TEST 2: With KV-Cache (use_cache=True)")
print("="*70)

input_ids = inputs['input_ids'].clone()
past_key_values = None
start_time = time.time()

with torch.no_grad():
    for i in range(num_tokens):
        # Use cached key-values
        outputs = model(
            input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
            past_key_values=past_key_values,
            use_cache=True
        )
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        
        probs = F.softmax(logits / 0.7, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        input_ids = torch.cat([input_ids, next_token], dim=-1)

elapsed_with_cache = time.time() - start_time
tokens_with_cache = input_ids.shape[1] - inputs['input_ids'].shape[1]
speed_with_cache = tokens_with_cache / elapsed_with_cache

print(f"\n✓ Generated {tokens_with_cache} tokens in {elapsed_with_cache:.2f}s")
print(f"  Speed: {speed_with_cache:.2f} tokens/sec")
print(f"  Latency: {(elapsed_with_cache/tokens_with_cache)*1000:.1f} ms/token")
print(f"\nOutput: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")

# ============================================================================
# Comparison
# ============================================================================
print("\n" + "="*70)
print("COMPARISON")
print("="*70)

speedup = speed_with_cache / speed_no_cache
time_saved = elapsed_no_cache - elapsed_with_cache
time_saved_pct = (time_saved / elapsed_no_cache) * 100

print(f"\n📊 Speed:")
print(f"  Without cache: {speed_no_cache:.2f} tokens/sec")
print(f"  With cache:    {speed_with_cache:.2f} tokens/sec")
print(f"  Speedup:       {speedup:.2f}×")

print(f"\n⏱️  Time:")
print(f"  Without cache: {elapsed_no_cache:.2f}s")
print(f"  With cache:    {elapsed_with_cache:.2f}s")
print(f"  Time saved:    {time_saved:.2f}s ({time_saved_pct:.1f}%)")

print(f"\n💡 Recommendation:")
if speedup > 1.5:
    print(f"  ✅ KV-cache provides {speedup:.1f}× speedup - HIGHLY RECOMMENDED")
    print(f"  ✅ Update demo_chat.py to use use_cache=True")
elif speedup > 1.1:
    print(f"  ✓ KV-cache provides {speedup:.1f}× speedup - recommended")
else:
    print(f"  ⚠️  KV-cache provides only {speedup:.1f}× speedup")
    print(f"  May not be worth the added complexity")

print("\n" + "="*70)
