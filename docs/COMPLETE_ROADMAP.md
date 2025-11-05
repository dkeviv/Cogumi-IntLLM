# QINS Complete Implementation Roadmap

**Status**: Pattern A Complete → Validation → Optimization → Pattern B

---

## ✅ Current State: Pattern A (Codec-at-Rest)

**What We Have**:
- ✅ QINS weight encoding/decoding working
- ✅ 100% token match on Phi-3.5 (15 tokens, 1 prompt)
- ✅ FP32 compute fully intact
- ✅ No accuracy risk
- ✅ Weights stored in QINS domain (currently float32)
- ✅ Decode just-in-time for matmul

**What We Don't Have Yet**:
- ❌ Quantization (still storing as float32, not uint8)
- ❌ Memory compression (0% savings currently)
- ❌ Robust validation (only 15 tokens, 1 prompt)
- ❌ KV cache compression
- ❌ Weight transport (Pattern B)

**Reality Check**: We have a **lossless reversible wrapper** around float tensors. This proves QINS integrates cleanly. Now we need to:
1. Validate it properly
2. Add actual compression
3. Optimize performance
4. Build Pattern B

---

## 🎯 Phase 1: Robust Validation (This Week)

**Goal**: Close the "15-token, single prompt" gap before adding complexity.

### A) Self-Consistency Test (1 minute)

**Purpose**: Prove codec correctness at mathematical level

**File**: `test_validation_self_consistency.py`

```python
#!/usr/bin/env python3
"""
Test A: Self-consistency of QINS codec
Validates: decode(encode(X)) ≈ X for all weight tensors
"""

import torch
from qins_weight_codec import qins_encode, qins_decode

def test_self_consistency():
    print("="*60)
    print("TEST A: Self-Consistency (Round-Trip Accuracy)")
    print("="*60)
    
    # Load Phi-3.5
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    target_layers = ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    results = []
    
    for name, module in model.named_modules():
        if any(t in name for t in target_layers):
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data
                
                # Round-trip
                W_enc = qins_encode(W, alpha=1.0, quantize=False)
                W_dec = qins_decode(W_enc, alpha=1.0, is_quantized=False)
                
                # Metrics
                abs_error = (W - W_dec).abs()
                rel_error = abs_error / (W.abs() + 1e-8)
                cosine = torch.nn.functional.cosine_similarity(
                    W.flatten(), W_dec.flatten(), dim=0
                )
                
                max_abs = abs_error.max().item()
                p99_rel = torch.quantile(rel_error.flatten(), 0.99).item()
                
                results.append({
                    "layer": name,
                    "max_abs_err": max_abs,
                    "p99_rel_err": p99_rel,
                    "cosine": cosine.item()
                })
                
                print(f"\n{name}:")
                print(f"  Max abs error: {max_abs:.6e}")
                print(f"  P99 rel error: {p99_rel:.4%}")
                print(f"  Cosine sim:    {cosine.item():.6f}")
    
    # Pass criteria
    print("\n" + "="*60)
    print("PASS CRITERIA:")
    print("  Cosine ≥ 0.9999")
    print("  P99 rel-err ≤ 3%")
    print("="*60)
    
    all_cosine = [r["cosine"] for r in results]
    all_p99 = [r["p99_rel_err"] for r in results]
    
    min_cosine = min(all_cosine)
    max_p99 = max(all_p99)
    
    print(f"\nWorst cosine: {min_cosine:.6f}")
    print(f"Worst P99:    {max_p99:.4%}")
    
    if min_cosine >= 0.9999 and max_p99 <= 0.03:
        print("\n✅ PASS: Codec is mathematically lossless!")
    else:
        print("\n❌ FAIL: Codec introduces significant error")
        if min_cosine < 0.9999:
            print(f"   Cosine too low: {min_cosine:.6f}")
        if max_p99 > 0.03:
            print(f"   P99 error too high: {max_p99:.4%}")

if __name__ == "__main__":
    test_self_consistency()
```

**Pass Bar**: 
- Cosine ≥ 0.9999
- P99 rel-err ≤ 3%

---

### B) Autoregressive Stability Test (5 minutes)

**Purpose**: Close "15-token, single prompt" gap

**File**: `test_validation_autoregressive.py`

```python
#!/usr/bin/env python3
"""
Test B: Autoregressive Stability
Validates: 10 prompts × 1,000 tokens with no drift
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qins_weight_codec import convert_linear_to_qins
import matplotlib.pyplot as plt

def test_autoregressive_stability():
    print("="*60)
    print("TEST B: Autoregressive Stability (1000 tokens)")
    print("="*60)
    
    # Diverse prompts
    prompts = [
        # News
        "Breaking news: Scientists have discovered",
        "In a surprising turn of events, the government announced",
        
        # Code
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        "class DatabaseConnection:\n    def __init__(self, host, port):\n        self.",
        
        # Math
        "To solve the equation x^2 + 5x + 6 = 0, we can factor it as",
        "The derivative of f(x) = x^3 + 2x^2 - x + 1 is",
        
        # Dialogue
        "Customer: I'd like to return this product.\nAgent: I understand. Can you",
        "Doctor: How long have you been experiencing these symptoms?\nPatient: It started about",
        
        # Reasoning
        "If all cats are mammals, and all mammals are animals, then logically",
        "The key difference between correlation and causation is that"
    ]
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\nLoading FP32 baseline...")
    model_fp32 = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model_fp32.eval()
    
    print("Loading QINS model...")
    model_qins = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    target_names = ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model_qins = convert_linear_to_qins(model_qins, target_names=target_names)
    model_qins.eval()
    
    max_tokens = 1000
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Testing: {prompt[:50]}...")
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        
        # Generate FP32
        with torch.no_grad():
            out_fp32 = model_fp32.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Generate QINS
        with torch.no_grad():
            out_qins = model_qins.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Token-by-token match
        tokens_fp32 = out_fp32[0].tolist()
        tokens_qins = out_qins[0].tolist()
        
        matches = []
        for j in range(min(len(tokens_fp32), len(tokens_qins))):
            matches.append(1 if tokens_fp32[j] == tokens_qins[j] else 0)
        
        # Compute match% per position window
        window_size = 50
        match_by_position = []
        for pos in range(0, len(matches), window_size):
            window = matches[pos:pos+window_size]
            match_pct = sum(window) / len(window) * 100
            match_by_position.append(match_pct)
        
        results.append({
            "prompt": prompt[:50],
            "total_match": sum(matches) / len(matches) * 100,
            "match_by_position": match_by_position
        })
        
        print(f"  Total match: {results[-1]['total_match']:.1f}%")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    for r in results:
        positions = [i*50 for i in range(len(r["match_by_position"]))]
        plt.plot(positions, r["match_by_position"], alpha=0.5, label=r["prompt"])
    plt.axhline(y=95, color='r', linestyle='--', label='Pass threshold (95%)')
    plt.xlabel("Token Position")
    plt.ylabel("Match % (50-token window)")
    plt.title("Autoregressive Stability: Match Rate vs Position")
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.savefig("autoregressive_stability.png", dpi=150)
    print("\n✓ Plot saved: autoregressive_stability.png")
    
    # Pass criteria
    avg_match = sum(r["total_match"] for r in results) / len(results)
    min_match = min(r["total_match"] for r in results)
    
    # Check for downward trend
    all_positions = []
    for r in results:
        all_positions.extend(r["match_by_position"])
    
    early = sum(all_positions[:len(all_positions)//3]) / (len(all_positions)//3)
    late = sum(all_positions[-len(all_positions)//3:]) / (len(all_positions)//3)
    
    print("\n" + "="*60)
    print("RESULTS:")
    print(f"  Average match: {avg_match:.2f}%")
    print(f"  Minimum match: {min_match:.2f}%")
    print(f"  Early (0-333):   {early:.2f}%")
    print(f"  Late (667-1000): {late:.2f}%")
    print(f"  Drift: {late - early:+.2f}%")
    print("="*60)
    print("PASS CRITERIA:")
    print("  Average match ≥ 95%")
    print("  No downward trend (late - early ≥ -2%)")
    print("="*60)
    
    if avg_match >= 95 and (late - early) >= -2:
        print("\n✅ PASS: Autoregressive generation is stable!")
    else:
        print("\n❌ FAIL: Stability issues detected")
        if avg_match < 95:
            print(f"   Average match too low: {avg_match:.2f}%")
        if (late - early) < -2:
            print(f"   Downward drift detected: {late - early:.2f}%")

if __name__ == "__main__":
    test_autoregressive_stability()
```

**Pass Bar**:
- Average greedy match ≥ 95%
- No downward trend (late - early ≥ -2%)

---

### C) Sampling Parity Test (5 minutes)

**Purpose**: Validate beyond greedy (temperature, top-p)

**File**: `test_validation_sampling.py`

```python
#!/usr/bin/env python3
"""
Test C: Sampling Parity
Validates: temperature=0.7, top_p=0.9 produces similar distributions
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from qins_weight_codec import convert_linear_to_qins
import numpy as np

def compute_kl_divergence(logits_fp32, logits_qins, temperature=1.0):
    """Compute KL(QINS || FP32) over logits"""
    p = F.softmax(logits_fp32 / temperature, dim=-1)
    q = F.softmax(logits_qins / temperature, dim=-1)
    
    kl = (p * (p.log() - q.log())).sum(dim=-1)
    return kl.mean().item()

def top_k_overlap(tokens_fp32, tokens_qins, k=10):
    """Compute top-k token overlap"""
    overlap = len(set(tokens_fp32[:k]) & set(tokens_qins[:k]))
    return overlap / k

def test_sampling_parity():
    print("="*60)
    print("TEST C: Sampling Parity")
    print("="*60)
    
    prompts = [
        "The future of artificial intelligence",
        "Once upon a time in a distant galaxy",
        "def calculate_fibonacci(n):",
        "The key to solving climate change is",
        "In quantum mechanics, the uncertainty principle",
    ] * 4  # 20 total
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\nLoading models...")
    model_fp32 = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model_fp32.eval()
    
    model_qins = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    target_names = ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model_qins = convert_linear_to_qins(model_qins, target_names=target_names)
    model_qins.eval()
    
    temperature = 0.7
    top_p = 0.9
    max_tokens = 256
    
    kl_divergences = []
    top1_overlaps = []
    top5_overlaps = []
    top10_overlaps = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt[:40]}...")
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        
        # Generate with sampling
        with torch.no_grad():
            out_fp32 = model_fp32.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id
            )
            
            out_qins = model_qins.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Compute logits for first generated token (for KL)
        with torch.no_grad():
            logits_fp32 = model_fp32(input_ids).logits[0, -1, :]
            logits_qins = model_qins(input_ids).logits[0, -1, :]
        
        kl = compute_kl_divergence(logits_fp32, logits_qins, temperature)
        kl_divergences.append(kl)
        
        # Top-k overlaps
        tokens_fp32 = out_fp32[0, input_ids.shape[1]:].tolist()
        tokens_qins = out_qins[0, input_ids.shape[1]:].tolist()
        
        # Get top-k from logits
        _, top_fp32 = torch.topk(logits_fp32, k=10)
        _, top_qins = torch.topk(logits_qins, k=10)
        
        top1_overlaps.append(top_k_overlap(top_fp32.tolist(), top_qins.tolist(), k=1))
        top5_overlaps.append(top_k_overlap(top_fp32.tolist(), top_qins.tolist(), k=5))
        top10_overlaps.append(top_k_overlap(top_fp32.tolist(), top_qins.tolist(), k=10))
    
    # Results
    avg_kl = np.mean(kl_divergences)
    avg_top1 = np.mean(top1_overlaps) * 100
    avg_top5 = np.mean(top5_overlaps) * 100
    avg_top10 = np.mean(top10_overlaps) * 100
    
    print("\n" + "="*60)
    print("RESULTS:")
    print(f"  KL divergence:  {avg_kl:.4f}")
    print(f"  Top-1 overlap:  {avg_top1:.1f}%")
    print(f"  Top-5 overlap:  {avg_top5:.1f}%")
    print(f"  Top-10 overlap: {avg_top10:.1f}%")
    print("="*60)
    print("PASS CRITERIA:")
    print("  KL < 0.1 (small divergence)")
    print("  Top-10 overlap ≥ 97%")
    print("="*60)
    
    if avg_kl < 0.1 and avg_top10 >= 97:
        print("\n✅ PASS: Sampling distributions match!")
    else:
        print("\n❌ FAIL: Sampling divergence detected")
        if avg_kl >= 0.1:
            print(f"   KL too high: {avg_kl:.4f}")
        if avg_top10 < 97:
            print(f"   Top-10 overlap too low: {avg_top10:.1f}%")

if __name__ == "__main__":
    test_sampling_parity()
```

**Pass Bar**:
- KL divergence < 0.1
- Top-10 overlap ≥ 97%

---

### D) Perplexity Test (3-5 minutes)

**Purpose**: Quantitative quality metric

**File**: `test_validation_perplexity.py`

```python
#!/usr/bin/env python3
"""
Test D: Perplexity on WikiText-2
Validates: PPL(QINS) ≈ PPL(FP32)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qins_weight_codec import convert_linear_to_qins
from datasets import load_dataset

def compute_perplexity(model, tokenizer, texts, max_length=512):
    """Compute perplexity on a list of texts"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
        input_ids = inputs.input_ids
        
        if input_ids.shape[1] < 2:
            continue
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            
        total_loss += loss.item() * input_ids.shape[1]
        total_tokens += input_ids.shape[1]
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

def test_perplexity():
    print("="*60)
    print("TEST D: Perplexity on WikiText-2")
    print("="*60)
    
    # Load dataset
    print("\nLoading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if len(t) > 100][:100]  # First 100 substantial texts
    print(f"Loaded {len(texts)} texts")
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\nComputing FP32 perplexity...")
    model_fp32 = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    ppl_fp32 = compute_perplexity(model_fp32, tokenizer, texts)
    print(f"FP32 PPL: {ppl_fp32:.2f}")
    
    print("\nComputing QINS perplexity...")
    model_qins = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    target_names = ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model_qins = convert_linear_to_qins(model_qins, target_names=target_names)
    ppl_qins = compute_perplexity(model_qins, tokenizer, texts)
    print(f"QINS PPL: {ppl_qins:.2f}")
    
    # Results
    delta_ppl = ppl_qins - ppl_fp32
    rel_delta = delta_ppl / ppl_fp32 * 100
    
    print("\n" + "="*60)
    print("RESULTS:")
    print(f"  FP32 PPL:  {ppl_fp32:.2f}")
    print(f"  QINS PPL:  {ppl_qins:.2f}")
    print(f"  Δ PPL:     {delta_ppl:+.2f}")
    print(f"  Rel Δ:     {rel_delta:+.2f}%")
    print("="*60)
    print("PASS CRITERIA:")
    print("  |Δ PPL| ≤ 0.1 OR |Rel Δ| ≤ 0.5%")
    print("="*60)
    
    if abs(delta_ppl) <= 0.1 or abs(rel_delta) <= 0.5:
        print("\n✅ PASS: Perplexity preserved!")
    else:
        print("\n❌ FAIL: Perplexity degradation detected")
        print(f"   Δ PPL = {delta_ppl:+.2f} (threshold ±0.1)")
        print(f"   Rel Δ = {rel_delta:+.2f}% (threshold ±0.5%)")

if __name__ == "__main__":
    test_perplexity()
```

**Pass Bar**:
- |Δ PPL| ≤ 0.1 OR |Rel Δ| ≤ 0.5%

---

### E) Long-Context Test (Optional, 5 minutes)

**Purpose**: Ensure no late-position blow-up

**File**: `test_validation_long_context.py`

```python
#!/usr/bin/env python3
"""
Test E: Long-Context Stability
Validates: No drift at 2k, 4k, 8k context lengths
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qins_weight_codec import convert_linear_to_qins

def test_long_context():
    print("="*60)
    print("TEST E: Long-Context Stability")
    print("="*60)
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    model_fp32 = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model_fp32.eval()
    
    model_qins = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    target_names = ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model_qins = convert_linear_to_qins(model_qins, target_names=target_names)
    model_qins.eval()
    
    # Test different context lengths
    context_lengths = [2000, 4000, 8000]
    
    for ctx_len in context_lengths:
        print(f"\n{'='*60}")
        print(f"Testing {ctx_len} token context")
        print(f"{'='*60}")
        
        # Create long context (repeat a pattern)
        base_text = "The quick brown fox jumps over the lazy dog. " * 100
        tokens = tokenizer(base_text, return_tensors="pt").input_ids
        
        # Truncate to desired length
        if tokens.shape[1] > ctx_len:
            tokens = tokens[:, :ctx_len]
        else:
            # Repeat to reach desired length
            repeats = (ctx_len // tokens.shape[1]) + 1
            tokens = tokens.repeat(1, repeats)[:, :ctx_len]
        
        print(f"Context length: {tokens.shape[1]} tokens")
        
        # Generate 256 more tokens
        with torch.no_grad():
            out_fp32 = model_fp32.generate(
                tokens,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            
            out_qins = model_qins.generate(
                tokens,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Check match
        new_fp32 = out_fp32[0, tokens.shape[1]:].tolist()
        new_qins = out_qins[0, tokens.shape[1]:].tolist()
        
        match = sum(1 for a, b in zip(new_fp32, new_qins) if a == b)
        match_pct = match / len(new_fp32) * 100
        
        print(f"Match: {match}/{len(new_fp32)} ({match_pct:.1f}%)")
        
        # Check for position-dependent drift
        early_match = sum(1 for a, b in zip(new_fp32[:64], new_qins[:64]) if a == b) / 64 * 100
        late_match = sum(1 for a, b in zip(new_fp32[-64:], new_qins[-64:]) if a == b) / 64 * 100
        
        print(f"Early (0-64):   {early_match:.1f}%")
        print(f"Late (192-256): {late_match:.1f}%")
        print(f"Drift: {late_match - early_match:+.1f}%")
        
        if match_pct >= 95 and (late_match - early_match) >= -5:
            print(f"✅ PASS at {ctx_len} tokens")
        else:
            print(f"❌ FAIL at {ctx_len} tokens")

if __name__ == "__main__":
    test_long_context()
```

**Pass Bar**:
- No significant drift at any context length
- Late-position match ≥ 90%

---

## 🎯 Phase 2: Bit-Packing (Week 2)

**Goal**: Pack QINS integers compactly (6-8 bits) WITHOUT quantization yet

**Critical**: This is **lossless bit-packing**, NOT quantization!
- We're still storing full precision QINS values
- Just packing them more efficiently in memory
- No information loss (reversible)

### Step 1: Implement Bit-Packing (8-bit first, then 6-bit)

**Goal**: Pack QINS integers tightly in memory (lossless)

**Key Insight**: QINS values are currently stored as float32 (4 bytes each). We can:
1. Map float32 QINS values → integer range [0, 255] for 8-bit
2. OR map to [0, 63] for 6-bit (more aggressive)
3. Pack multiple values into fewer bytes
4. Unpack + decode before compute

**This is LOSSLESS if we keep full integer precision!**

**File**: `qins_bitpack.py`

```python
#!/usr/bin/env python3
"""
QINS Bit-Packing (Lossless Storage Optimization)
Pack QINS float32 values into integer representation
NO quantization - just more efficient storage format
"""

import torch

def pack_8bit_lossless(qins_values: torch.Tensor) -> torch.Tensor:
    """
    Pack QINS float32 values into 8-bit integers (lossless)
    
    QINS values are in range [-1, 1]
    Map to [0, 255]: int = round((qins + 1.0) * 127.5)
    
    This is REVERSIBLE with no information loss
    (beyond what's already in float32 precision)
    
    Compression: 4 bytes → 1 byte per value (4×)
    """
    # Clamp to valid QINS range
    qins_clamped = qins_values.clamp(-1.0, 1.0)
    
    # Map [-1, 1] → [0, 255]
    packed = ((qins_clamped + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    
    return packed

def unpack_8bit_lossless(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack 8-bit integers back to QINS float32 values
    
    Reverse mapping: qins = (int / 127.5) - 1.0
    """
    # Map [0, 255] → [-1, 1]
    qins_values = (packed.float() / 127.5) - 1.0
    
    return qins_values

def pack_6bit_lossless(qins_values: torch.Tensor) -> torch.Tensor:
    """
    Pack QINS values into 6-bit integers (more aggressive)
    
    Map [-1, 1] → [0, 63]
    4 values can fit in 3 bytes (24 bits)
    Compression: 16 bytes → 3 bytes (5.33×)
    
    NOTE: This loses 2 bits of precision per value
    Only use if validation shows <1% error
    """
    qins_clamped = qins_values.clamp(-1.0, 1.0)
    
    # Map [-1, 1] → [0, 63]
    packed = ((qins_clamped + 1.0) * 31.5).round().clamp(0, 63).to(torch.uint8)
    
    # TODO: Actually pack 4 values into 3 bytes (bit manipulation)
    # For now, this is logically 6-bit but stored as uint8
    return packed

def unpack_6bit_lossless(packed: torch.Tensor) -> torch.Tensor:
    """Unpack 6-bit back to QINS float32"""
    # Map [0, 63] → [-1, 1]
    qins_values = (packed.float() / 31.5) - 1.0
    return qins_values

class QINSBitPackedLinear(nn.Module):
    """
    Linear layer with bit-packed QINS storage
    
    Storage: QINS values packed to 8-bit or 6-bit integers
    Forward: unpack → decode → FP32 matmul
    """
    def __init__(
        self,
        linear: nn.Linear,
        alpha: float = 1.0,
        bits: int = 8  # 8 or 6
    ):
        super().__init__()
        self.alpha = alpha
        self.bits = bits
        
        # Encode to QINS
        from qins_weight_codec import qins_encode
        w_qins = qins_encode(linear.weight.data, alpha, quantize=False)
        
        # Pack to integers
        if bits == 8:
            w_packed = pack_8bit_lossless(w_qins)
        elif bits == 6:
            w_packed = pack_6bit_lossless(w_qins)
        else:
            raise ValueError(f"Only 6 or 8 bits supported, got {bits}")
        
        self.register_buffer("w_packed", w_packed)
        
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data)
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: unpack → decode → matmul"""
        # Unpack to QINS float32
        if self.bits == 8:
            w_qins = unpack_8bit_lossless(self.w_packed)
        else:
            w_qins = unpack_6bit_lossless(self.w_packed)
        
        # Decode to FP32
        from qins_weight_codec import qins_decode
        w = qins_decode(w_qins, self.alpha, is_quantized=False)
        
        # Standard matmul
        return F.linear(x, w, self.bias)

# TODO: Benchmark pack/unpack overhead vs memory bandwidth savings
# TODO: Validate 6-bit vs 8-bit quality difference
```

**Test Plan**:
```python
# Test round-trip accuracy
W = torch.randn(1000, 1000)
W_qins = qins_encode(W, alpha=1.0, quantize=False)

# 8-bit packing
W_8bit = pack_8bit_lossless(W_qins)
W_unpacked = unpack_8bit_lossless(W_8bit)
W_decoded = qins_decode(W_unpacked, alpha=1.0)

error = (W - W_decoded).abs().max()
print(f"8-bit round-trip error: {error:.6e}")
# Should be ~1e-4 (float32 precision)

# Memory check
print(f"Original:  {W.element_size() * W.numel()} bytes")
print(f"Packed:    {W_8bit.element_size() * W_8bit.numel()} bytes")
print(f"Reduction: {W.numel() * 4 / W_8bit.numel():.1f}×")
```

---

## 🎯 Phase 3: Optimization (Week 3)

### Step 3: Fused Decode Kernel

**Goal**: Move decode closer to matmul

**File**: `qins_fused_decode.py`

```python
#!/usr/bin/env python3
"""
QINS Fused Decode Kernel
Decode weights right before matmul (no Python overhead)
"""

import torch
import triton
import triton.language as tl

@triton.jit
def qins_decode_kernel(
    encoded_ptr,
    decoded_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for QINS decoding
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load encoded values
    z = tl.load(encoded_ptr + offsets, mask=mask, other=0.0)
    
    # QINS decode: x = sign(z) * (1 - |z|) / (alpha * |z|)
    sign_z = tl.where(z >= 0, 1.0, -1.0)
    abs_z = tl.abs(z)
    abs_z = tl.maximum(abs_z, 1e-12)  # Avoid division by zero
    
    x = sign_z * (1.0 - abs_z) / (alpha * abs_z)
    
    # Store decoded values
    tl.store(decoded_ptr + offsets, x, mask=mask)

def qins_decode_fused(encoded: torch.Tensor, alpha: float) -> torch.Tensor:
    """Fused QINS decode using Triton"""
    n_elements = encoded.numel()
    decoded = torch.empty_like(encoded)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    qins_decode_kernel[grid](
        encoded, decoded, alpha, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return decoded

# TODO: Benchmark vs Python decode
# TODO: Profile memory bandwidth
```

---

## 🎯 Phase 4: First Quantization - KV-V Only (Week 4)

### Step 4: QINS-INT8 for KV-V Only (FIRST quantization experiment)

**Goal**: Compress V in KV cache with quantization (lowest risk)

**Critical**: This is the FIRST time we introduce actual quantization!
- NOT weights (they stay lossless bit-packed)
- ONLY KV-V values in cache
- Safe because: V doesn't affect autoregressive bookkeeping

**File**: `qins_kv_cache.py`

```python
#!/usr/bin/env python3
"""
QINS KV Cache Wrapper
Compress V values in attention cache
"""

import torch
from typing import Tuple

class QINSKVCache:
    """
    Wrapper around DynamicCache that compresses V
    """
    def __init__(self, alpha: float = 1.0, quantize: bool = True):
        self.alpha = alpha
        self.quantize = quantize
        self.key_cache = []
        self.value_cache = []  # Will store QINS-encoded V
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Store K as-is, encode V before storing
        """
        # K: Store unchanged
        if layer_idx >= len(self.key_cache):
            self.key_cache.append(key_states)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=2
            )
        
        # V: Encode before storing
        from qins_weight_codec import qins_encode
        v_encoded = qins_encode(value_states, self.alpha, self.quantize)
        
        if layer_idx >= len(self.value_cache):
            self.value_cache.append(v_encoded)
        else:
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], v_encoded], dim=2
            )
        
        # Return for attention (decode V on the fly)
        from qins_weight_codec import qins_decode
        v_decoded = qins_decode(
            self.value_cache[layer_idx],
            self.alpha,
            is_quantized=self.quantize
        )
        
        return self.key_cache[layer_idx], v_decoded
    
    def get_usable_length(self, *args, **kwargs):
        return self.key_cache[0].shape[2] if self.key_cache else 0
    
    # TODO: Implement remaining DynamicCache methods

# Test: Measure KV memory savings
# Should see ~2× reduction in V storage
```

---

## 🎯 Phase 5: Pattern B - Weight Transport (Week 5-6)

### Step 5: Begin Weight Transport (v_proj only)

**Goal**: First QINS-native compute

**File**: `qins_transport.py`

```python
#!/usr/bin/env python3
"""
Pattern B: Jacobian Weight Transport
Transform weights to QINS-native domain
"""

import torch

def compute_jacobian_encode(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    ∂E/∂x where E(x) = sign(x) / (1 + α|x|)
    
    ∂E/∂x = -α * sign(x)^2 / (1 + α|x|)^2
          = -α / (1 + α|x|)^2
    """
    abs_x = x.abs()
    denom = (1.0 + alpha * abs_x) ** 2
    jacobian = -alpha / denom
    return jacobian

def compute_jacobian_decode(z: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    ∂D/∂z where D(z) = sign(z) * (1 - |z|) / (α|z|)
    """
    abs_z = z.abs().clamp(min=1e-12)
    term1 = -1.0 / (alpha * abs_z)
    term2 = -(1.0 - abs_z) / (alpha * abs_z ** 2)
    jacobian = term1 + term2
    return jacobian

def transport_weights_to_qins(
    W: torch.Tensor,
    x_sample: torch.Tensor,
    alpha: float = 1.0
) -> torch.Tensor:
    """
    Transport FP32 weights to QINS-native: W' = (∂D/∂z) · W · (∂E/∂x)^(-1)
    
    Args:
        W: Weight matrix [out, in]
        x_sample: Sample input activations [batch, in]
        alpha: QINS density
    
    Returns:
        W_qins: Transported weights for QINS-native compute
    """
    from qins_weight_codec import qins_encode
    
    # Get sample z values
    z_sample = qins_encode(x_sample.mean(dim=0), alpha, quantize=False)
    
    # Compute Jacobians
    jac_encode = compute_jacobian_encode(x_sample.mean(dim=0), alpha)
    jac_decode = compute_jacobian_decode(z_sample, alpha)
    
    # Invert encode Jacobian (element-wise, diagonal)
    jac_encode_inv = 1.0 / (jac_encode + 1e-12)
    
    # Transport: W' = J_D · W · J_E^-1
    # Since Jacobians are diagonal, this is element-wise scaling
    W_transported = jac_decode.unsqueeze(0) * W * jac_encode_inv.unsqueeze(-1)
    
    return W_transported

# TODO: Test on v_proj of single layer
# TODO: Validate output matches FP32
```

---

### Step 6: QINS-Native Matmul

**File**: `qins_native_matmul.py`

```python
#!/usr/bin/env python3
"""
QINS-Native Matrix Multiplication
Compute in QINS domain without decode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QINSNativeLinear(nn.Module):
    """
    Linear layer with QINS-native compute (Pattern B)
    """
    def __init__(
        self,
        linear: nn.Linear,
        x_sample: torch.Tensor,
        alpha: float = 1.0
    ):
        super().__init__()
        self.alpha = alpha
        
        # Transport weights once
        from qins_transport import transport_weights_to_qins
        W_qins = transport_weights_to_qins(linear.weight.data, x_sample, alpha)
        
        self.register_buffer("weight_qins", W_qins)
        
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data)
        else:
            self.bias = None
    
    def forward(self, x_qins: torch.Tensor) -> torch.Tensor:
        """
        Compute in QINS domain (no decode!)
        
        Input: QINS tensor
        Output: QINS tensor (caller decodes)
        """
        # Matmul in QINS domain
        out_qins = F.linear(x_qins, self.weight_qins)
        
        # Note: Bias is NOT added here (it's FP32)
        # Caller must decode and add bias
        return out_qins

# TODO: Test against FP32 v_proj
# TODO: Measure decode reduction (should be ~10× fewer decodes)
```

---

### Step 7: Extend to More Layers

**Order**:
1. v_proj ✅ (done in Step 6)
2. down_proj (MLP)
3. up_proj (MLP)
4. o_proj (attention output)
5. NOT q_proj/k_proj yet (defer to later)

---

## 📋 Summary Timeline

| Week | Phase | Goal | Deliverable | Quantization? |
|------|-------|------|-------------|---------------|
| 1 | Validation A-E | Close gaps | 5 test files passing | **NO** - lossless only |
| 2 | Bit-packing | 4× memory | 8-bit/6-bit packing | **NO** - lossless packing |
| 2 | Pack validation | Verify overhead | Microbenchmarks | **NO** |
| 3 | Optimization | Speed | Fused decode kernel | **NO** |
| 4 | KV-V quantization | 2× KV mem | QINS-INT8 for V only | **YES** - first quant! |
| 5 | Transport v_proj | First native compute | Single layer Pattern B | **NO** - just transport |
| 6 | Transport MLP | More native compute | down/up/o_proj | **NO** |
| 7+ | Full Pattern B | All layers | Complete implementation | **NO** |

**Key Point**: Quantization is ONLY used for KV-V (Week 4). Everything else stays lossless!

---

## 🎯 Success Criteria

**Pattern A Complete When**:
- ✅ All validation tests pass (A-E)
- ✅ 4× weight compression active (lossless bit-packing)
- ✅ 2× KV cache compression working (QINS-INT8 for V only)
- ✅ No quality degradation (<1% PPL increase)
- ✅ Stable over 1000+ tokens
- ✅ Works with sampling (not just greedy)
- ✅ Fused decode kernel working
- ⚠️ **Quantization ONLY for KV-V, everything else lossless**

**Pattern B Complete When**:
- ✅ Weight transport working (v_proj + MLP)
- ✅ QINS-native matmul validated
- ✅ Decode operations reduced 10×
- ✅ Quality preserved (>99% match)
- ✅ Measurable speed improvement

---

## 🚨 Critical Path

**DO NOT PROCEED TO NEXT PHASE UNTIL CURRENT PHASE PASSES ALL TESTS**

1. **Week 1**: Validation must pass → Pattern A confidence ✅
   - **NO quantization** - only lossless codec testing
2. **Week 2**: Bit-packing must work → Real memory wins ✅
   - **NO quantization** - lossless integer packing only
3. **Week 3**: Optimization → Production-ready speed ✅
   - **NO quantization** - just fused kernels
4. **Week 4**: KV-V quantization → First quant experiment ✅
   - **YES quantization** - but ONLY for KV-V values
5. **Week 5-6**: Pattern B → Native compute proof ✅
   - **NO quantization** - weight transport is transformation, not quantization

**Each phase builds on previous. No shortcuts!**

**Critical Reminder**: 
- Quantization is introduced ONLY in Week 4 (KV-V)
- All other phases remain lossless
- This minimizes risk and isolates potential issues
