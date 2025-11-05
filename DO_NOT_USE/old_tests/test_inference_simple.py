#!/usr/bin/env python3
"""
Simple inference quality test using working benchmark approach.
Just runs a quick generation test with FP32 vs QINS decoded models.
"""

# Copy the working DynamicCache shim from benchmark
from transformers.cache_utils import DynamicCache
import torch

# Initialize seen_tokens attribute if it doesn't exist
if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = 0

if not hasattr(DynamicCache, "_max_cache_length"):
    DynamicCache._max_cache_length = None

if not hasattr(DynamicCache, "set_max_length"):
    def set_max_length(self, max_length: int):
        self._max_cache_length = int(max_length)
    DynamicCache.set_max_length = set_max_length

if not hasattr(DynamicCache, "get_usable_length"):
    def get_usable_length(self, seq_length: int, layer_idx: int = 0):
        return int(getattr(self, "seen_tokens", 0) or 0)
    DynamicCache.get_usable_length = get_usable_length

if not hasattr(DynamicCache, "get_max_length"):
    def get_max_length(self):
        return getattr(self, "_max_cache_length", None) or float("inf")
    DynamicCache.get_max_length = get_max_length

# Wrap update to track seen_tokens
if hasattr(DynamicCache, "update"):
    _original_update = DynamicCache.update
    def wrapped_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        result = _original_update(self, key_states, value_states, layer_idx, cache_kwargs)
        # Track seen_tokens from actual cache length
        if hasattr(self, "key_cache") and len(self.key_cache) > layer_idx:
            self.seen_tokens = self.key_cache[layer_idx].shape[-2]
        return result
    DynamicCache.update = wrapped_update

# Proper crop implementation
if not hasattr(DynamicCache, "crop"):
    def crop(self, max_length: int):
        """Crop cache to max_length by slicing the actual KV tensors."""
        self._max_cache_length = int(max_length)
        
        if not hasattr(self, "key_cache") or not self.key_cache:
            return
        
        # Crop each layer's cache
        for i in range(len(self.key_cache)):
            if self.key_cache[i] is not None:
                k = self.key_cache[i]
                v = self.value_cache[i]
                
                # k/v shape: [batch, heads, seq, dim]
                seq_len = k.shape[-2]
                
                if seq_len > max_length:
                    # Keep the most recent max_length tokens
                    self.key_cache[i] = k[..., seq_len - max_length:, :]
                    self.value_cache[i] = v[..., seq_len - max_length:, :]
        
        # Update tracker
        if hasattr(self, "seen_tokens"):
            self.seen_tokens = min(self.seen_tokens, max_length)
    
    DynamicCache.crop = crop

print("✓ Applied DynamicCache.get_usable_length compatibility fix")
print("✓ Applied DynamicCache.get_max_length compatibility fix")
print("✓ Applied DynamicCache.set_max_length compatibility fix")
print("✓ Applied DynamicCache.crop compatibility fix")

from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def load_and_decode_qins(model_path):
    """Load QINS model and decode all weights to FP32."""
    print(f"Loading QINS state dict...")
    qins_state = torch.load(model_path, map_location='cpu')
    
    print("Decoding QINS weights to FP32...")
    fp32_state = {}
    count = 0
    
    for key in list(qins_state.keys()):
        if key.endswith('.stored'):
            layer_name = key[:-7]
            
            stored = qins_state[f'{layer_name}.stored']
            sign = qins_state[f'{layer_name}.sign']
            log_min = qins_state[f'{layer_name}.log_min']
            log_max = qins_state[f'{layer_name}.log_max']
            
            # Decode
            normalized = (255.0 - stored.float()) / 254.0
            log_weight = log_min + normalized * (log_max - log_min)
            abs_weight = torch.exp(log_weight)
            weight = sign.float() * abs_weight
            
            fp32_state[f'{layer_name}.weight'] = weight
            
            if f'{layer_name}.bias' in qins_state:
                fp32_state[f'{layer_name}.bias'] = qins_state[f'{layer_name}.bias']
            
            count += 1
            if count % 10 == 0:
                print(f"  Decoded {count} layers...", end='\r')
        
        elif not any(key.endswith(s) for s in ['.sign', '.log_min', '.log_max', '.stored']):
            fp32_state[key] = qins_state[key]
    
    print(f"  Decoded {count} QINS layers total")
    return fp32_state

def test_generation(model, tokenizer, prompt, max_tokens=20):
    """Test generation with a simple prompt."""
    # Simple tokenization without padding - matches benchmark approach
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print(f"\nPrompt: '{prompt}'")
    
    with torch.no_grad():
        start = time.time()
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=max_tokens,
            do_sample=False,  # Greedy
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        elapsed = time.time() - start
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tokens = outputs[0].tolist()
    
    print(f"Generated: '{text}'")
    print(f"Tokens: {len(tokens)} in {elapsed:.2f}s ({len(tokens)/elapsed:.1f} tok/s)")
    
    return tokens, text

def main():
    print('=' * 80)
    print('SIMPLE INFERENCE QUALITY TEST')
    print('=' * 80)
    print()
    
    device = 'cpu'
    prompt = "The capital of France is"
    max_tokens = 15
    
    # Load tokenizer once
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3.5-mini-instruct')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Important for causal models
    print("✓ Tokenizer loaded")
    print()
    
    # Test 1: FP32 Original
    print('=' * 80)
    print('TEST 1: FP32 ORIGINAL MODEL')
    print('=' * 80)
    print()
    
    print("Loading FP32 model...")
    model_fp32 = AutoModelForCausalLM.from_pretrained(
        'microsoft/Phi-3.5-mini-instruct',
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model_fp32.eval()
    model_fp32.config.use_cache = True
    
    # Initialize cache with max length for Phi-3.5 compatibility
    try:
        max_ctx = getattr(model_fp32.config, "sliding_window", None) \
                  or getattr(model_fp32.config, "max_position_embeddings", None) \
                  or 4096
        cache = DynamicCache()
        cache.set_max_length(int(max_ctx))
        print(f"  Cache max length set to: {max_ctx}")
    except Exception as e:
        print(f"  Warning: Could not set cache max length: {e}")
    
    print("✓ FP32 model loaded")
    
    tokens_fp32, text_fp32 = test_generation(model_fp32, tokenizer, prompt, max_tokens)
    
    del model_fp32
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print()
    
    # Test 2: QINS Decoded
    print('=' * 80)
    print('TEST 2: QINS MODEL (DECODED TO FP32)')
    print('=' * 80)
    print()
    
    # Load QINS and decode
    fp32_state = load_and_decode_qins('models/phi35-qins-codec.pt')
    
    print("\nLoading model architecture...")
    model_qins = AutoModelForCausalLM.from_pretrained(
        'microsoft/Phi-3.5-mini-instruct',
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    print("Loading decoded weights...")
    model_qins.load_state_dict(fp32_state, strict=True)
    model_qins.eval()
    model_qins.config.use_cache = True
    
    # Initialize cache with max length for Phi-3.5 compatibility
    try:
        max_ctx = getattr(model_qins.config, "sliding_window", None) \
                  or getattr(model_qins.config, "max_position_embeddings", None) \
                  or 4096
        cache = DynamicCache()
        cache.set_max_length(int(max_ctx))
        print(f"  Cache max length set to: {max_ctx}")
    except Exception as e:
        print(f"  Warning: Could not set cache max length: {e}")
    
    print("✓ QINS model loaded and decoded")
    
    tokens_qins, text_qins = test_generation(model_qins, tokenizer, prompt, max_tokens)
    
    # Compare
    print()
    print('=' * 80)
    print('COMPARISON')
    print('=' * 80)
    print()
    
    print(f"FP32:  {text_fp32}")
    print(f"QINS:  {text_qins}")
    print()
    
    # Token match
    min_len = min(len(tokens_fp32), len(tokens_qins))
    matches = sum(1 for i in range(min_len) if tokens_fp32[i] == tokens_qins[i])
    match_rate = matches / min_len if min_len > 0 else 0
    
    print(f"Token match: {matches}/{min_len} ({match_rate*100:.1f}%)")
    
    if matches < min_len:
        print("\nFirst difference at position:")
        for i in range(min_len):
            if tokens_fp32[i] != tokens_qins[i]:
                print(f"  Position {i}:")
                print(f"    FP32: {tokenizer.decode([tokens_fp32[i]])} (id={tokens_fp32[i]})")
                print(f"    QINS: {tokenizer.decode([tokens_qins[i]])} (id={tokens_qins[i]})")
                break
    
    print()
    
    if match_rate >= 0.95:
        print("✅ EXCELLENT: >95% token match")
        return True
    elif match_rate >= 0.80:
        print("✅ GOOD: >80% token match")
        return True
    else:
        print("⚠️  Token mismatch detected")
        return False

if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
