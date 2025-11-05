#!/usr/bin/env python3
"""
Absolute minimal test - just load Phi-3.5 and generate one token.
No QINS, no custom code, just pure HuggingFace.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    torch_dtype=torch.float32,
    trust_remote_code=True
)
model.eval()

print("Testing generation...")
prompt = "Hello"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated: {text}")
print("✓ SUCCESS!")
