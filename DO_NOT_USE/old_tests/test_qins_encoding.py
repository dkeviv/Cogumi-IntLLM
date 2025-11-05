"""
Debug: Check if QINS encoding/decoding is correct
Verify the inverse relationship and reconstruction accuracy
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.projective_layer import convert_to_qins, reconstruct_from_qins

print("=" * 80)
print("QINS ENCODING/DECODING VERIFICATION")
print("=" * 80)

# Test 1: Verify inverse relationship
print("\n1. Testing Inverse Relationship")
print("-" * 80)

# Create test weights with known magnitudes
test_weights = torch.tensor([
    -0.5,      # Large magnitude (negative)
    0.5,       # Large magnitude (positive) 
    -0.001,    # Small magnitude (negative)
    0.001,     # Small magnitude (positive)
    -0.1,      # Medium magnitude
    0.1,       # Medium magnitude
])

stored, sign, log_min, log_max = convert_to_qins(test_weights)

print(f"\nOriginal weights: {test_weights.tolist()}")
print(f"Stored values:    {stored.tolist()}")
print(f"Signs:            {sign.tolist()}")
print(f"Log range:        [{log_min:.6f}, {log_max:.6f}]")

print(f"\n✓ Checking inverse relationship:")
for i, (w, s) in enumerate(zip(test_weights, stored)):
    print(f"  Weight: {w.item():+.6f} (|w|={abs(w.item()):.6f}) → stored={s.item():3d}")

# Verify: largest magnitude should have smallest stored value
max_mag_idx = test_weights.abs().argmax()
min_mag_idx = test_weights.abs().argmin()

print(f"\n  Largest |w| = {test_weights[max_mag_idx].abs().item():.6f} → stored = {stored[max_mag_idx].item()}")
print(f"  Smallest |w| = {test_weights[min_mag_idx].abs().item():.6f} → stored = {stored[min_mag_idx].item()}")

if stored[max_mag_idx] < stored[min_mag_idx]:
    print(f"  ✅ CORRECT: Large magnitude → small stored value")
else:
    print(f"  ❌ WRONG: Inverse relationship broken!")

# Test 2: Verify reconstruction accuracy
print("\n2. Testing Reconstruction Accuracy")
print("-" * 80)

reconstructed = reconstruct_from_qins(stored, sign, log_min, log_max)

print(f"\nOriginal:       {test_weights.tolist()}")
print(f"Reconstructed:  {reconstructed.tolist()}")

abs_error = (test_weights - reconstructed).abs()
rel_error = abs_error / (test_weights.abs() + 1e-8)

print(f"\nAbsolute errors: {abs_error.tolist()}")
print(f"Relative errors: {[f'{e*100:.2f}%' for e in rel_error.tolist()]}")

print(f"\nMean absolute error: {abs_error.mean():.8f}")
print(f"Max absolute error:  {abs_error.max():.8f}")

if abs_error.mean() < 0.001:
    print(f"✅ GOOD: Reconstruction accurate")
else:
    print(f"⚠️  WARNING: High reconstruction error")

# Test 3: Check sign preservation
print("\n3. Testing Sign Preservation")
print("-" * 80)

original_signs = torch.sign(test_weights)
reconstructed_signs = torch.sign(reconstructed)

sign_match = (original_signs == reconstructed_signs).all()

print(f"Original signs:      {original_signs.tolist()}")
print(f"Reconstructed signs: {reconstructed_signs.tolist()}")
print(f"Match: {sign_match}")

if sign_match:
    print(f"✅ PERFECT: All signs preserved")
else:
    print(f"❌ ERROR: Sign mismatch!")

# Test 4: Layer-level test
print("\n4. Testing Full Layer Conversion")
print("-" * 80)

from src.projective_layer import ProjectiveLinear

in_dim = 64
out_dim = 32

fp32_layer = nn.Linear(in_dim, out_dim)
nn.init.normal_(fp32_layer.weight, mean=0, std=0.02)

qins_layer = ProjectiveLinear(in_dim, out_dim)
qins_layer.from_linear(fp32_layer)

# Test on random input
test_input = torch.randn(10, in_dim) * 0.02

with torch.no_grad():
    fp32_output = fp32_layer(test_input)
    qins_output = qins_layer(test_input)

output_error = (fp32_output - qins_output).abs().mean()

print(f"Input shape:  {test_input.shape}")
print(f"Output shape: {fp32_output.shape}")
print(f"Output error: {output_error:.8f}")

if output_error < 0.001:
    print(f"✅ EXCELLENT: Layer output accurate")
elif output_error < 0.01:
    print(f"✅ GOOD: Layer output acceptable")
else:
    print(f"⚠️  WARNING: High layer output error")

# Test 5: Check calibrated layer
print("\n5. Testing Calibrated Layer (with scaling)")
print("-" * 80)

from src.calibrated_qins import CalibratedProjectiveLinear

# Create calibration data
calib_input = torch.randn(50, in_dim) * 0.02

calibrated_layer = CalibratedProjectiveLinear.from_linear(
    fp32_layer,
    calibration_input=calib_input
)

print(f"Scale range: [{calibrated_layer.scale.min():.6f}, {calibrated_layer.scale.max():.6f}]")
print(f"Scale mean:  {calibrated_layer.scale.mean():.6f}")

# Test on new data
test_input_new = torch.randn(10, in_dim) * 0.02

with torch.no_grad():
    fp32_out = fp32_layer(test_input_new)
    qins_out = qins_layer(test_input_new)
    calibrated_out = calibrated_layer(test_input_new)

fp32_std = fp32_out.std()
qins_std = qins_out.std()
calibrated_std = calibrated_out.std()

print(f"\nOutput std comparison:")
print(f"  FP32:       {fp32_std:.6f}")
print(f"  QINS:       {qins_std:.6f} (ratio: {qins_std/fp32_std:.4f})")
print(f"  Calibrated: {calibrated_std:.6f} (ratio: {calibrated_std/fp32_std:.4f})")

calibrated_error = (fp32_out - calibrated_out).abs().mean()
standard_error = (fp32_out - qins_out).abs().mean()

print(f"\nMean absolute error:")
print(f"  Standard QINS:   {standard_error:.8f}")
print(f"  Calibrated QINS: {calibrated_error:.8f}")

if calibrated_std / fp32_std > 0.95 and calibrated_std / fp32_std < 1.05:
    print(f"✅ Calibration working: Std ratio close to 1.0")
else:
    print(f"⚠️  Calibration issue: Std ratio off")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
