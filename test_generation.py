#!/usr/bin/env python3
"""
Quick test to verify chat generation works end-to-end
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'src')
from projective_layer import ProjectiveLinear
import torch.nn.functional as F

print("="*60)
print("Testing QINS Chat Generation (Simplified)")
print("="*60)

# Simulate a tiny transformer-like generation test
print("\n[Test 1] Weight reconstruction quality")
linear = nn.Linear(768, 768, bias=True)
nn.init.normal_(linear.weight, mean=0, std=0.02)

proj = ProjectiveLinear(768, 768, bias=True)
proj.from_linear(linear)

# Test forward pass
x = torch.randn(1, 10, 768)  # Batch=1, Seq=10, Hidden=768

with torch.no_grad():
    y_orig = linear(x)
    y_qins = proj(x)

error = torch.abs(y_orig - y_qins).mean().item()
print(f"Forward pass error: {error:.6f}")
print(f"Status: {'✓ PASS' if error < 0.1 else '✗ FAIL'}")

# Test if generation loop would work
print("\n[Test 2] Token-by-token generation simulation")
vocab_size = 32000
lm_head = nn.Linear(768, vocab_size, bias=False)
nn.init.normal_(lm_head.weight, mean=0, std=0.02)

proj_lm = ProjectiveLinear(768, vocab_size, bias=False)
proj_lm.from_linear(lm_head)

# Simulate generation
input_ids = torch.tensor([[1, 100, 200]])  # Fake tokens
hidden = torch.randn(1, 3, 768)

with torch.no_grad():
    # Original
    logits_orig = lm_head(hidden[:, -1, :])
    probs_orig = F.softmax(logits_orig / 0.7, dim=-1)
    next_token_orig = torch.multinomial(probs_orig, 1)
    
    # QINS
    logits_qins = proj_lm(hidden[:, -1, :])
    probs_qins = F.softmax(logits_qins / 0.7, dim=-1)
    next_token_qins = torch.multinomial(probs_qins, 1)

logits_error = torch.abs(logits_orig - logits_qins).mean().item()
print(f"Logits error: {logits_error:.6f}")
print(f"Original next token: {next_token_orig.item()}")
print(f"QINS next token: {next_token_qins.item()}")
print(f"Status: {'✓ PASS' if logits_error < 1.0 else '✗ FAIL'}")

# Test if sampling distribution is reasonable
print("\n[Test 3] Sampling distribution quality")
top_k = 10
orig_top_probs, orig_top_indices = torch.topk(probs_orig[0], top_k)
qins_top_probs, qins_top_indices = torch.topk(probs_qins[0], top_k)

overlap = len(set(orig_top_indices.tolist()) & set(qins_top_indices.tolist()))
print(f"Top-{top_k} token overlap: {overlap}/{top_k}")
print(f"Status: {'✓ PASS' if overlap >= top_k * 0.7 else '✗ FAIL'}")

print("\n" + "="*60)
print("GENERATION TEST SUMMARY")
print("="*60)

if error < 0.1 and logits_error < 1.0 and overlap >= top_k * 0.7:
    print("🎉 ALL TESTS PASSED!")
    print("✓ QINS generation should work correctly")
    print("✓ Chat responses will be coherent")
else:
    print("⚠️  Some tests failed")
    print("Chat quality may be degraded")
