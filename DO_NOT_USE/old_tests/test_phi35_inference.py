#!/usr/bin/env python3
"""
Test inference quality of QINS-converted Phi-3.5-mini.
Compares generation between FP32 original and QINS codec model.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from qins_codec import QINSLinear

# Fix DynamicCache compatibility issue with Phi-3.5
if not hasattr(DynamicCache, "get_usable_length"):
    def get_usable_length(self, seq_length: int, layer_idx: int = 0):
        """Return usable length for cache."""
        return int(getattr(self, "seen_tokens", 0) or 0)
    DynamicCache.get_usable_length = get_usable_length
    print("✓ Applied DynamicCache.get_usable_length compatibility fix")

if not hasattr(DynamicCache, "get_max_length"):
    def get_max_length(self):
        """Return max length for cache."""
        return None
    DynamicCache.get_max_length = get_max_length
    print("✓ Applied DynamicCache.get_max_length compatibility fix")

def load_qins_model(model_path, device='cpu'):
    """Load QINS model - reconstruct weights on the fly."""
    print(f"Loading QINS model from {model_path}...")
    
    # Load QINS state dict first
    qins_state = torch.load(model_path, map_location='cpu')
    print(f"✓ Loaded QINS state dict ({len(qins_state)} tensors)")
    
    # Reconstruct FP32 weights from QINS encoding
    print("Reconstructing FP32 weights from QINS encoding...")
    fp32_state = {}
    
    # Find all stored/sign pairs and decode them
    processed_layers = set()
    
    for key in qins_state.keys():
        if key.endswith('.stored'):
            layer_name = key[:-7]  # Remove '.stored'
            
            if layer_name in processed_layers:
                continue
            processed_layers.add(layer_name)
            
            # Get QINS components
            stored = qins_state[f'{layer_name}.stored']
            sign = qins_state[f'{layer_name}.sign']
            log_min = qins_state[f'{layer_name}.log_min']
            log_max = qins_state[f'{layer_name}.log_max']
            
            # Decode to FP32
            normalized = (255.0 - stored.float()) / 254.0
            log_weight = log_min + normalized * (log_max - log_min)
            abs_weight = torch.exp(log_weight)
            weight = sign.float() * abs_weight
            
            # Store as weight
            fp32_state[f'{layer_name}.weight'] = weight
            
            # Copy bias if exists
            bias_key = f'{layer_name}.bias'
            if bias_key in qins_state:
                fp32_state[bias_key] = qins_state[bias_key]
        
        elif not any(key.endswith(suffix) for suffix in ['.sign', '.log_min', '.log_max', '.stored']):
            # Copy non-QINS parameters (embeddings, norms, etc.)
            fp32_state[key] = qins_state[key]
    
    print(f"✓ Reconstructed {len(processed_layers)} QINS layers to FP32")
    
    # Now load model with reconstructed FP32 weights
    print("Loading model architecture with reconstructed weights...")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        'microsoft/Phi-3.5-mini-instruct',
        trust_remote_code=True
    )
    
    # Disable cache during model initialization to avoid cache errors
    config.use_cache = False
    
    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True
    )
    
    # Re-enable cache after loading
    model.config.use_cache = True
    
    # Load the reconstructed FP32 state
    model.load_state_dict(fp32_state, strict=True)
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ QINS model loaded on {device} (decoded to FP32)")
    return model

def greedy_generate(model, tokenizer, prompt, max_tokens=50, device='cpu'):
    """Generate text using greedy decoding."""
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    
    generated_ids = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            generated_ids.append(next_token.item())
            
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    
    return generated_ids

def compare_logits(model_fp32, model_qins, tokenizer, prompt, device='cpu'):
    """Compare logits between FP32 and QINS models."""
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    
    with torch.no_grad():
        # FP32 forward
        outputs_fp32 = model_fp32(input_ids)
        logits_fp32 = outputs_fp32.logits[:, -1, :]
        
        # QINS forward
        outputs_qins = model_qins(input_ids)
        logits_qins = outputs_qins.logits[:, -1, :]
    
    return logits_fp32, logits_qins

def calculate_metrics(logits_fp32, logits_qins):
    """Calculate comparison metrics between logits."""
    # Absolute difference
    abs_diff = (logits_fp32 - logits_qins).abs()
    
    # Relative difference
    rel_diff = abs_diff / (logits_fp32.abs() + 1e-8)
    
    # Top-k overlap
    k = 10
    top_fp32 = logits_fp32.topk(k).indices
    top_qins = logits_qins.topk(k).indices
    overlap = len(set(top_fp32[0].tolist()) & set(top_qins[0].tolist()))
    
    # Argmax match
    argmax_match = (logits_fp32.argmax() == logits_qins.argmax()).item()
    
    # KL divergence
    probs_fp32 = F.softmax(logits_fp32, dim=-1)
    probs_qins = F.softmax(logits_qins, dim=-1)
    kl_div = F.kl_div(
        probs_qins.log(),
        probs_fp32,
        reduction='batchmean'
    ).item()
    
    return {
        'abs_diff_mean': abs_diff.mean().item(),
        'abs_diff_max': abs_diff.max().item(),
        'rel_diff_mean': rel_diff.mean().item(),
        'rel_diff_max': rel_diff.max().item(),
        'top10_overlap': overlap,
        'argmax_match': argmax_match,
        'kl_divergence': kl_div
    }

def main():
    print('=' * 80)
    print('PHI-3.5-MINI INFERENCE QUALITY TEST')
    print('FP32 Original vs QINS Codec-at-Rest')
    print('=' * 80)
    print()
    
    device = 'cpu'  # Use CPU for now to avoid memory issues
    
    # ========================================================================
    # PART 1: LOAD MODELS
    # ========================================================================
    print('📥 PART 1: LOADING MODELS')
    print('-' * 80)
    print()
    
    print('Loading FP32 original model...')
    model_fp32 = AutoModelForCausalLM.from_pretrained(
        'microsoft/Phi-3.5-mini-instruct',
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model_fp32 = model_fp32.to(device)
    model_fp32.eval()
    print('✓ FP32 model loaded')
    print()
    
    print('Loading QINS codec model...')
    model_qins = load_qins_model('models/phi35-qins-codec.pt', device=device)
    print()
    
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3.5-mini-instruct')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print('✓ Tokenizer loaded')
    print()
    
    # ========================================================================
    # PART 2: LOGITS COMPARISON
    # ========================================================================
    print('=' * 80)
    print('📊 PART 2: LOGITS COMPARISON')
    print('-' * 80)
    print()
    
    test_prompts = [
        "The capital of France is",
        "To be or not to be,",
        "In the beginning,",
        "Once upon a time",
        "The answer is"
    ]
    
    all_metrics = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"Test {i}/5: '{prompt}'")
        
        logits_fp32, logits_qins = compare_logits(
            model_fp32, model_qins, tokenizer, prompt, device
        )
        
        metrics = calculate_metrics(logits_fp32, logits_qins)
        all_metrics.append(metrics)
        
        print(f"  Abs diff: mean={metrics['abs_diff_mean']:.6f}, max={metrics['abs_diff_max']:.6f}")
        print(f"  Rel diff: mean={metrics['rel_diff_mean']:.4%}, max={metrics['rel_diff_max']:.4%}")
        print(f"  Top-10 overlap: {metrics['top10_overlap']}/10")
        print(f"  Argmax match: {'✓' if metrics['argmax_match'] else '✗'}")
        print(f"  KL divergence: {metrics['kl_divergence']:.6f}")
        print()
    
    # Average metrics
    print('Average across all prompts:')
    avg_metrics = {
        key: sum(m[key] for m in all_metrics) / len(all_metrics)
        for key in all_metrics[0].keys()
    }
    
    print(f"  Abs diff: mean={avg_metrics['abs_diff_mean']:.6f}")
    print(f"  Rel diff: mean={avg_metrics['rel_diff_mean']:.4%}")
    print(f"  Top-10 overlap: {avg_metrics['top10_overlap']:.1f}/10")
    print(f"  Argmax match rate: {avg_metrics['argmax_match']*100:.1f}%")
    print(f"  KL divergence: {avg_metrics['kl_divergence']:.6f}")
    print()
    
    # ========================================================================
    # PART 3: GREEDY GENERATION TEST
    # ========================================================================
    print('=' * 80)
    print('📊 PART 3: GREEDY GENERATION TEST')
    print('-' * 80)
    print()
    
    test_prompt = "The capital of France is"
    max_tokens = 20
    
    print(f"Prompt: '{test_prompt}'")
    print(f"Max tokens: {max_tokens}")
    print()
    
    print("Generating with FP32 model...")
    start = time.time()
    tokens_fp32 = greedy_generate(model_fp32, tokenizer, test_prompt, max_tokens, device)
    time_fp32 = time.time() - start
    text_fp32 = tokenizer.decode(tokens_fp32)
    print(f"✓ Generated {len(tokens_fp32)} tokens in {time_fp32:.2f}s")
    print(f"  Text: '{text_fp32}'")
    print()
    
    print("Generating with QINS model...")
    start = time.time()
    tokens_qins = greedy_generate(model_qins, tokenizer, test_prompt, max_tokens, device)
    time_qins = time.time() - start
    text_qins = tokenizer.decode(tokens_qins)
    print(f"✓ Generated {len(tokens_qins)} tokens in {time_qins:.2f}s")
    print(f"  Text: '{text_qins}'")
    print()
    
    # Token-by-token comparison
    match_count = sum(1 for t1, t2 in zip(tokens_fp32, tokens_qins) if t1 == t2)
    match_rate = match_count / max(len(tokens_fp32), len(tokens_qins))
    
    print("Comparison:")
    print(f"  Tokens generated: FP32={len(tokens_fp32)}, QINS={len(tokens_qins)}")
    print(f"  Match rate: {match_count}/{max(len(tokens_fp32), len(tokens_qins))} ({match_rate*100:.1f}%)")
    print(f"  Speed: FP32={time_fp32:.2f}s, QINS={time_qins:.2f}s ({time_qins/time_fp32:.2f}×)")
    print()
    
    if match_rate < 1.0:
        print("Token differences:")
        for i, (t1, t2) in enumerate(zip(tokens_fp32, tokens_qins)):
            if t1 != t2:
                print(f"  Position {i}: FP32={tokenizer.decode([t1])} (id={t1}), "
                      f"QINS={tokenizer.decode([t2])} (id={t2})")
    
    print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print('=' * 80)
    print('📋 SUMMARY')
    print('=' * 80)
    print()
    
    print('Logits comparison:')
    print(f"  Mean absolute error: {avg_metrics['abs_diff_mean']:.6f}")
    print(f"  Mean relative error: {avg_metrics['rel_diff_mean']:.4%}")
    print(f"  Top-10 overlap: {avg_metrics['top10_overlap']:.1f}/10")
    print(f"  Argmax match: {avg_metrics['argmax_match']*100:.0f}%")
    print(f"  KL divergence: {avg_metrics['kl_divergence']:.6f}")
    print()
    
    print('Generation quality:')
    print(f"  Token match rate: {match_rate*100:.1f}%")
    print(f"  Speed overhead: {time_qins/time_fp32:.2f}× ({(time_qins/time_fp32-1)*100:+.1f}%)")
    print()
    
    # Quality assessment
    if (avg_metrics['argmax_match'] >= 0.95 and 
        match_rate >= 0.95 and
        avg_metrics['kl_divergence'] < 0.01):
        print('✅ INFERENCE QUALITY: EXCELLENT')
        print('   QINS codec maintains high fidelity to FP32 original.')
        success = True
    elif (avg_metrics['argmax_match'] >= 0.80 and 
          match_rate >= 0.80):
        print('✅ INFERENCE QUALITY: GOOD')
        print('   Minor differences from FP32, but acceptable.')
        success = True
    else:
        print('⚠️  INFERENCE QUALITY: NEEDS REVIEW')
        print('   Significant differences from FP32 detected.')
        success = False
    
    return success

if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
