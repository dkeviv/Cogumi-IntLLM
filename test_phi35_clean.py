"""
Clean test of Phi-3.5 with NO shims - verifying base functionality
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 70)
print("CLEAN PHI-3.5 TEST (No Shims)")
print("=" * 70)

model_id = "microsoft/Phi-3.5-mini-instruct"

print("\n1. Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(model_id)
tok.padding_side = "right"           # important for causal models
tok.truncation_side = "right"
print(f"✅ Tokenizer loaded")
print(f"   Padding side: {tok.padding_side}")
print(f"   Pad token: {tok.pad_token}")
print(f"   EOS token: {tok.eos_token}")

print("\n2. Loading model...")
# Detect best device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
model = model.to(device)
model.eval()
model.config.use_cache = True
print(f"✅ Model loaded")
print(f"   Device: {model.device}")
print(f"   Use cache: {model.config.use_cache}")

print("\n3. Testing forward pass with cache...")
prompt = "The future of artificial intelligence is"
inputs = tok(prompt, return_tensors="pt").to(model.device)

print(f"   Input IDs shape: {inputs['input_ids'].shape}")
print(f"   Attention mask shape: {inputs['attention_mask'].shape}")
print(f"   Padding side: {tok.padding_side}")

with torch.no_grad():
    # forward (no generate) with cache enabled
    out = model(**inputs, use_cache=True)
    print(f"✅ Forward pass successful")
    print(f"   Output shape: {out.logits.shape}")
    if out.past_key_values:
        print(f"   Cache layers: {len(out.past_key_values)}")

print("\n4. Testing generation...")
gen = model.generate(**inputs, max_new_tokens=16, use_cache=True)
result = tok.decode(gen[0], skip_special_tokens=True)
print(f"✅ Generation successful")
print(f"   Result: {result}")

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED - Phi-3.5 base functionality works!")
print("=" * 70)
