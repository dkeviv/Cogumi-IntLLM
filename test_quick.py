#!/usr/bin/env python3
"""
Ultra-simple test with GPT-2 (much faster to load)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from projective_layer import ProjectiveLinear
import torch.nn as nn

def convert_model_to_projective(model):
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
    print(f"Converted {count} layers")
    return model

print("="*60)
print("Quick QINS Test with GPT-2 (faster!)")
print("="*60)

# Load GPT-2 (much smaller and faster)
print("\n📥 Loading GPT-2...")
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
print("✓ Loaded")

# Convert
print("\n🔄 Converting to QINS...")
model = convert_model_to_projective(model)
print("✓ Converted")

# Test
print("\n📝 Generating text...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print(f"\nPrompt: {prompt}")
print("Output: ", end='')

with torch.no_grad():
    for _ in range(20):
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits / 0.7, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        token_text = tokenizer.decode(next_token[0])
        print(token_text, end='', flush=True)
        
        inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token], dim=-1)
        inputs['attention_mask'] = torch.cat([
            inputs['attention_mask'],
            torch.ones((1, 1), dtype=torch.long, device=device)
        ], dim=-1)

print("\n\n✅ QINS generation working!")
print("="*60)
