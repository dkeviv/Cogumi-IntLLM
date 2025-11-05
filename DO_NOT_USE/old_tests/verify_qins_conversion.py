#!/usr/bin/env python3
"""
Verify that weights were actually converted to QINS.
Check if the model has QINSWeightLinear layers instead of nn.Linear.
"""

import torch
from transformers import AutoModelForCausalLM
from qins_weight_codec import convert_linear_to_qins, QINSWeightLinear
import torch.nn as nn

print("="*70)
print("VERIFYING QINS WEIGHT CONVERSION")
print("="*70)

# Load and convert model
print("\n1. Loading Phi-3.5...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    torch_dtype=torch.float32,
    trust_remote_code=True
)

print("\n2. Checking original model layers...")
original_linear_count = 0
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        original_linear_count += 1
        if original_linear_count <= 3:
            print(f"   Found nn.Linear: {name} {module.weight.shape}")

print(f"   Total nn.Linear layers: {original_linear_count}")

print("\n3. Converting to QINS...")
model = convert_linear_to_qins(
    model,
    target_names=["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    alpha=1.0,
    verbose=False  # Suppress conversion output
)

print("\n4. Checking converted model layers...")
qins_count = 0
linear_count = 0
for name, module in model.named_modules():
    if isinstance(module, QINSWeightLinear):
        qins_count += 1
        if qins_count <= 3:
            print(f"   ✅ Found QINSWeightLinear: {name}")
            print(f"      - Encoded weights: {module.w_encoded.shape}")
            print(f"      - Storage dtype: {module.w_encoded.dtype}")
            print(f"      - Alpha: {module.alpha}")
    elif isinstance(module, nn.Linear):
        linear_count += 1
        if linear_count <= 3:
            print(f"   Found nn.Linear (not converted): {name} {module.weight.shape}")

print(f"\n5. Summary:")
print(f"   QINSWeightLinear layers: {qins_count}")
print(f"   Remaining nn.Linear layers: {linear_count}")
print(f"   Original nn.Linear layers: {original_linear_count}")

if qins_count > 0:
    print("\n✅ VERIFIED: Weights were converted to QINS!")
    print(f"   - {qins_count} layers now use QINSWeightLinear")
    print(f"   - Weights stored in QINS domain (w_encoded)")
    print(f"   - Decoded on-the-fly during forward pass")
else:
    print("\n❌ ERROR: No QINS layers found!")

# Test encode/decode cycle
print("\n6. Testing encode/decode on a sample weight...")
sample_layer = None
for module in model.modules():
    if isinstance(module, QINSWeightLinear):
        sample_layer = module
        break

if sample_layer:
    from qins_weight_codec import qins_decode
    
    # Decode the encoded weights
    decoded_weights = qins_decode(sample_layer.w_encoded, sample_layer.alpha)
    
    print(f"   Encoded weights shape: {sample_layer.w_encoded.shape}")
    print(f"   Encoded weights range: [{sample_layer.w_encoded.min():.6f}, {sample_layer.w_encoded.max():.6f}]")
    print(f"   Decoded weights range: [{decoded_weights.min():.6f}, {decoded_weights.max():.6f}]")
    print(f"   Decoded weights dtype: {decoded_weights.dtype}")
    
    # Check that encoded values are in expected QINS range
    if sample_layer.w_encoded.abs().max() <= 1.0:
        print("\n   ✅ Encoded values in expected QINS range [-1, 1]")
    else:
        print(f"\n   ⚠️  Encoded values outside expected range: max={sample_layer.w_encoded.abs().max()}")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
