#!/usr/bin/env python3
"""
Terminal-based chat test for QINS model
Quick way to verify chat generation works without Gradio
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import with proper path handling
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
                # Create ProjectiveLinear replacement
                proj_layer = ProjectiveLinear(
                    child.in_features,
                    child.out_features,
                    bias=(child.bias is not None),
                    scale=scale
                )
                
                # Convert weights
                proj_layer.from_linear(child)
                
                # Replace in module
                setattr(module, child_name, proj_layer)
                converted_count += 1
                
                if verbose:
                    print(f"  ✓ Converted: {full_name} [{child.in_features} → {child.out_features}]")
            else:
                # Recurse
                convert_layer(child, full_name)
    
    if verbose:
        print("="*60)
        print("Converting model to Projective INT8")
        print("="*60)
    
    convert_layer(model)
    
    if verbose:
        print(f"\n✓ Converted {converted_count} Linear layers")
        print("="*60)
    
    return model

print("="*60)
print("QINS Terminal Chat Test")
print("="*60)

# Load model
print("\n📥 Loading Phi-3.5-mini from HuggingFace...")
print("(This will take a few minutes on first run)")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    torch_dtype=torch.float32,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✓ Model loaded!")

# Convert to QINS
print("\n🔄 Converting to QINS...")
model = convert_model_to_projective(model, verbose=False)
print("✓ Conversion complete!")

# Move to device
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
model.eval()
print(f"✓ Model on device: {device}")

# Chat function
def generate_response(prompt, max_tokens=100, temperature=0.7, top_p=0.9):
    """Generate a single response."""
    # Format with Phi-3.5 chat template
    chat_prompt = f"<|system|>\nYou are a helpful AI assistant.<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    
    # Tokenize
    input_ids = tokenizer(chat_prompt, return_tensors="pt").input_ids.to(device)
    
    # Generate
    response_text = ""
    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(input_ids, use_cache=False)
            logits = outputs.logits[:, -1, :]
            
            # Temperature
            logits = logits / temperature
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[0, indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Decode and print
            # Decode the full sequence so far to get proper spacing
            current_response = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            # Extract just the assistant's response (after the last <|assistant|>)
            if "<|assistant|>" in current_response:
                new_response = current_response.split("<|assistant|>")[-1].strip()
            else:
                new_response = current_response
            
            # Print only the new part
            if len(new_response) > len(response_text):
                new_text = new_response[len(response_text):]
                print(new_text, end='', flush=True)
                response_text = new_response
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    
    print()  # Newline
    return response_text

# Interactive chat
print("\n" + "="*60)
print("Chat is ready! Type your messages below.")
print("Commands: 'quit' to exit, 'clear' to reset")
print("="*60)

history = []

while True:
    try:
        user_input = input("\n👤 You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            history = []
            print("\n🗑️  Chat history cleared!")
            continue
        
        print("🤖 Assistant: ", end='', flush=True)
        response = generate_response(user_input)
        history.append((user_input, response))
        
    except KeyboardInterrupt:
        print("\n\n👋 Chat interrupted. Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
