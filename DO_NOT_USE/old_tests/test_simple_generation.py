#!/usr/bin/env python3
"""
Simple test: Load QINS model and generate text
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

sys.path.insert(0, 'src')
from projective_layer import ProjectiveLinear
import torch.nn as nn

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

print("="*70)
print("Simple QINS Generation Test")
print("="*70)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"\nDevice: {device}")

# Load tokenizer
print("\n📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
tokenizer.pad_token = tokenizer.eos_token

# Load model
print("\n📥 Loading FP32 model...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    torch_dtype=torch.float32,
    trust_remote_code=True
)

print("\n🔄 Converting to QINS...")
model = convert_model_to_qins(model)

print("\n📤 Moving to device...")
model = model.to(device)
model.eval()

print("\n✅ Model ready!")

# Test generation
print("\n" + "="*70)
print("Testing Generation")
print("="*70)

prompt = "The future of artificial intelligence is"
print(f"\nPrompt: '{prompt}'")
print("Generating response...")

inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_ids = inputs['input_ids']

# Simple autoregressive generation
generated_text = prompt
with torch.no_grad():
    for i in range(30):  # Generate 30 tokens
        # Forward pass (disable cache to avoid compatibility issues)
        outputs = model(input_ids, use_cache=False)
        logits = outputs.logits[:, -1, :]
        
        # Sample next token
        probs = F.softmax(logits / 0.7, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        # Decode and print
        token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
        print(token_text, end='', flush=True)
        generated_text += token_text
        
        # Append for next iteration
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # Stop if EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

print("\n\n" + "="*70)
print("✅ Generation Complete!")
print("="*70)
print(f"\nFull response:\n{generated_text}")
