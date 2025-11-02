#!/usr/bin/env python3
"""
Quick single-prompt test for QINS chat
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from projective_layer import ProjectiveLinear
import torch.nn as nn

def convert_model_to_projective(model, scale=256, verbose=True):
    """Convert model's Linear layers to ProjectiveLinear."""
    converted_count = 0
    
    def convert_layer(module, name=""):
        nonlocal converted_count
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            
            if isinstance(child, nn.Linear):
                proj_layer = ProjectiveLinear(
                    child.in_features,
                    child.out_features,
                    bias=(child.bias is not None),
                    scale=scale
                )
                proj_layer.from_linear(child)
                setattr(module, child_name, proj_layer)
                converted_count += 1
                
                if verbose and converted_count % 10 == 0:
                    print(f"  Converted {converted_count} layers...")
            else:
                convert_layer(child, full_name)
    
    if verbose:
        print("\n🔄 Converting to QINS...")
    
    convert_layer(model)
    
    if verbose:
        print(f"✓ Converted {converted_count} layers!\n")
    
    return model

print("="*60)
print("QINS Single-Prompt Test")
print("="*60)

# Load
print("\n📥 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    torch_dtype=torch.float32,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
print("✓ Model loaded")

# Convert
model = convert_model_to_projective(model, verbose=True)

# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()
print(f"✓ On device: {device}\n")

# Generate
prompt = "What is 2+2?"
chat_prompt = f"<|system|>\nYou are a helpful AI assistant.<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

print("="*60)
print(f"👤 User: {prompt}")
print("🤖 Assistant: ", end='', flush=True)

input_ids = tokenizer(chat_prompt, return_tensors="pt").input_ids.to(device)

response_text = ""
with torch.no_grad():
    for i in range(50):  # Generate 50 tokens
        outputs = model(input_ids, use_cache=False)
        logits = outputs.logits[:, -1, :]
        
        # Sample with temperature
        logits = logits / 0.7
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        # Append token
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # Decode full sequence and extract response
        full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if "<|assistant|>" in full_text:
            new_response = full_text.split("<|assistant|>")[-1].strip()
        else:
            new_response = full_text
        
        # Print new text
        if len(new_response) > len(response_text):
            new_part = new_response[len(response_text):]
            print(new_part, end='', flush=True)
            response_text = new_response

print("\n\n" + "="*60)
print("✅ Test complete!")
print(f"Generated {len(input_ids[0])} tokens total")
print("="*60)
