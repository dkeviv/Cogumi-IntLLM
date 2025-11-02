"""
Test Calibrated QINS on Greedy Multi-Step Generation
Compares standard QINS vs calibrated QINS
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.projective_layer import ProjectiveLinear
from src.calibrated_qins import CalibratedProjectiveLinear, LogitScaler

print("=" * 80)
print("CALIBRATED QINS: Greedy Multi-Step Test")
print("Testing if calibration fixes ranking instability")
print("=" * 80)

# Model definition
class StandardQINSModel(nn.Module):
    """Standard QINS (no calibration)"""
    def __init__(self, vocab_size=5000, hidden_dim=512, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            ProjectiveLinear(hidden_dim if i == 0 else hidden_dim * 2, hidden_dim * 2)
            for i in range(num_layers)
        ])
        self.layers.append(ProjectiveLinear(hidden_dim * 2, hidden_dim))
        self.vocab_proj = ProjectiveLinear(hidden_dim, vocab_size)
        
    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = F.gelu(layer(x))
        return self.vocab_proj(x)

class CalibratedQINSModel(nn.Module):
    """Calibrated QINS with scaling"""
    def __init__(self, vocab_size=5000, hidden_dim=512, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            CalibratedProjectiveLinear(hidden_dim if i == 0 else hidden_dim * 2, hidden_dim * 2)
            for i in range(num_layers)
        ])
        self.layers.append(CalibratedProjectiveLinear(hidden_dim * 2, hidden_dim))
        self.vocab_proj = CalibratedProjectiveLinear(hidden_dim, vocab_size)
        self.logit_scaler = LogitScaler(alpha=1.0)
        
    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = F.gelu(layer(x))
        logits = self.vocab_proj(x)
        return self.logit_scaler(logits)

class FP32Model(nn.Module):
    """FP32 reference"""
    def __init__(self, vocab_size=5000, hidden_dim=512, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim if i == 0 else hidden_dim * 2, hidden_dim * 2)
            for i in range(num_layers)
        ])
        self.layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
        self.vocab_proj = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = F.gelu(layer(x))
        return self.vocab_proj(x)

# Configuration
vocab_size = 5000
hidden_dim = 256  # Smaller for faster testing
num_layers = 3
num_steps = 500  # Reduced for comparison
start_seq_len = 20

print(f"\nConfiguration:")
print(f"  Vocab: {vocab_size:,}, Hidden: {hidden_dim}, Layers: {num_layers}")
print(f"  Decode steps: {num_steps}, Initial context: {start_seq_len}")

# Create FP32 model
print(f"\n{'='*80}")
print("1. Creating FP32 Reference Model")
print(f"{'='*80}")

torch.manual_seed(42)
fp32_model = FP32Model(vocab_size, hidden_dim, num_layers)
fp32_model.eval()

for name, param in fp32_model.named_parameters():
    if 'embed' in name:
        nn.init.normal_(param, mean=0, std=0.02)
    elif 'weight' in name:
        nn.init.normal_(param, mean=0, std=0.02)
    elif 'bias' in name:
        nn.init.zeros_(param)

print(f"✅ FP32 model created")

# Generate calibration data
print(f"\n{'='*80}")
print("2. Generating Calibration Data")
print(f"{'='*80}")

num_calib_samples = 50
calib_contexts = torch.randint(0, vocab_size, (num_calib_samples, start_seq_len))

print(f"✅ Generated {num_calib_samples} calibration contexts")

# Convert to standard QINS (no calibration)
print(f"\n{'='*80}")
print("3. Creating Standard QINS (No Calibration)")
print(f"{'='*80}")

standard_qins = StandardQINSModel(vocab_size, hidden_dim, num_layers)

with torch.no_grad():
    standard_qins.embed.weight.copy_(fp32_model.embed.weight)
    for fp32_layer, qins_layer in zip(fp32_model.layers, standard_qins.layers):
        qins_layer.from_linear(fp32_layer)
    standard_qins.vocab_proj.from_linear(fp32_model.vocab_proj)

standard_qins.eval()
print(f"✅ Standard QINS created")

# Convert to calibrated QINS
print(f"\n{'='*80}")
print("4. Creating Calibrated QINS")
print(f"{'='*80}")

calibrated_qins = CalibratedQINSModel(vocab_size, hidden_dim, num_layers)

with torch.no_grad():
    calibrated_qins.embed.weight.copy_(fp32_model.embed.weight)

# Collect activations for calibration
layer_activations = []

with torch.no_grad():
    # Get embedding output
    embed_out = fp32_model.embed(calib_contexts)
    layer_activations.append(embed_out.reshape(-1, embed_out.shape[-1]))
    
    # Get outputs from each layer
    x = embed_out
    for layer in fp32_model.layers:
        x = F.gelu(layer(x))
        layer_activations.append(x.reshape(-1, x.shape[-1]))

print(f"  Collected {len(layer_activations)} layer activations")

# Convert layers with calibration
with torch.no_grad():
    for i, (fp32_layer, calib_layer) in enumerate(zip(fp32_model.layers, calibrated_qins.layers)):
        calib_input = layer_activations[i]
        converted = CalibratedProjectiveLinear.from_linear(
            fp32_layer,
            calibration_input=calib_input[:100]  # Use subset
        )
        calib_layer.load_state_dict(converted.state_dict())
        print(f"  Layer {i+1}: scale range [{converted.scale.min():.3f}, {converted.scale.max():.3f}]")
    
    # Convert vocab projection
    calib_input = layer_activations[-1]
    vocab_converted = CalibratedProjectiveLinear.from_linear(
        fp32_model.vocab_proj,
        calibration_input=calib_input[:100]
    )
    calibrated_qins.vocab_proj.load_state_dict(vocab_converted.state_dict())
    print(f"  Vocab proj: scale range [{vocab_converted.scale.min():.3f}, {vocab_converted.scale.max():.3f}]")
    
    # Calibrate logit scaler
    fp32_logits = fp32_model(calib_contexts[:10])
    qins_logits_pre = calibrated_qins.vocab_proj(layer_activations[-1][:10 * start_seq_len].reshape(10, start_seq_len, -1))
    
    logit_scaler = LogitScaler.calibrate(
        fp32_logits.reshape(-1, vocab_size),
        qins_logits_pre.reshape(-1, vocab_size)
    )
    calibrated_qins.logit_scaler = logit_scaler
    print(f"  Logit scaler: α = {logit_scaler.alpha.item():.4f}")

calibrated_qins.eval()
print(f"✅ Calibrated QINS created")

# Run greedy generation for all three models
print(f"\n{'='*80}")
print("5. Running Greedy Multi-Step Generation (3-way comparison)")
print(f"{'='*80}")

torch.manual_seed(123)
initial_context = torch.randint(0, vocab_size, (1, start_seq_len))

# Initialize sequences
fp32_seq = initial_context.clone()
standard_seq = initial_context.clone()
calibrated_seq = initial_context.clone()

# Metrics
standard_matches = []
standard_top10 = []
calibrated_matches = []
calibrated_top10 = []

print(f"\nGenerating {num_steps} tokens...")
print(f"Progress (every 50 steps):")

with torch.no_grad():
    for step in range(num_steps):
        # FP32 prediction
        fp32_logits = fp32_model(fp32_seq)[0, -1, :]
        fp32_next = fp32_logits.argmax().unsqueeze(0).unsqueeze(0)
        
        # Standard QINS prediction
        standard_logits = standard_qins(standard_seq)[0, -1, :]
        standard_next = standard_logits.argmax().unsqueeze(0).unsqueeze(0)
        
        # Calibrated QINS prediction
        calibrated_logits = calibrated_qins(calibrated_seq)[0, -1, :]
        calibrated_next = calibrated_logits.argmax().unsqueeze(0).unsqueeze(0)
        
        # Standard QINS metrics
        standard_match = (fp32_next.item() == standard_next.item())
        standard_matches.append(1 if standard_match else 0)
        
        fp32_topk = set(torch.topk(fp32_logits, 10).indices.tolist())
        standard_topk_set = set(torch.topk(standard_logits, 10).indices.tolist())
        standard_overlap = len(fp32_topk & standard_topk_set) / 10
        standard_top10.append(standard_overlap)
        
        # Calibrated QINS metrics
        calibrated_match = (fp32_next.item() == calibrated_next.item())
        calibrated_matches.append(1 if calibrated_match else 0)
        
        calibrated_topk_set = set(torch.topk(calibrated_logits, 10).indices.tolist())
        calibrated_overlap = len(fp32_topk & calibrated_topk_set) / 10
        calibrated_top10.append(calibrated_overlap)
        
        # Progress
        if (step + 1) % 50 == 0:
            std_match = sum(standard_matches[-50:]) / 50 * 100
            std_top10 = sum(standard_top10[-50:]) / 50 * 100
            cal_match = sum(calibrated_matches[-50:]) / 50 * 100
            cal_top10 = sum(calibrated_top10[-50:]) / 50 * 100
            
            print(f"  Step {step+1:3d}:")
            print(f"    Standard:   Match={std_match:5.1f}%, Top-10={std_top10:5.1f}%")
            print(f"    Calibrated: Match={cal_match:5.1f}%, Top-10={cal_top10:5.1f}%")
        
        # Continue sequences
        fp32_seq = torch.cat([fp32_seq, fp32_next], dim=1)
        standard_seq = torch.cat([standard_seq, standard_next], dim=1)
        calibrated_seq = torch.cat([calibrated_seq, calibrated_next], dim=1)

# Results
print(f"\n{'='*80}")
print("6. Results Analysis")
print(f"{'='*80}")

standard_match_rate = sum(standard_matches) / len(standard_matches) * 100
standard_top10_rate = sum(standard_top10) / len(standard_top10) * 100

calibrated_match_rate = sum(calibrated_matches) / len(calibrated_matches) * 100
calibrated_top10_rate = sum(calibrated_top10) / len(calibrated_top10) * 100

print(f"\n📊 Overall Results ({num_steps} steps):")
print(f"\n  Standard QINS (no calibration):")
print(f"    Greedy match rate: {standard_match_rate:.2f}%")
print(f"    Top-10 overlap:    {standard_top10_rate:.2f}%")

print(f"\n  Calibrated QINS:")
print(f"    Greedy match rate: {calibrated_match_rate:.2f}%")
print(f"    Top-10 overlap:    {calibrated_top10_rate:.2f}%")

print(f"\n  Improvement:")
print(f"    Greedy match: {calibrated_match_rate - standard_match_rate:+.2f}%")
print(f"    Top-10 overlap: {calibrated_top10_rate - standard_top10_rate:+.2f}%")

# Visualization
print(f"\n{'='*80}")
print("7. Creating Comparison Visualization")
print(f"{'='*80}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'Standard vs Calibrated QINS - {num_steps} Steps', 
             fontsize=16, fontweight='bold')

window = 25

# Plot 1: Greedy match comparison
ax1 = axes[0, 0]
std_smooth = np.convolve(standard_matches, np.ones(window)/window, mode='valid')
cal_smooth = np.convolve(calibrated_matches, np.ones(window)/window, mode='valid')
ax1.plot(range(window, num_steps+1), std_smooth, label='Standard QINS', linewidth=2, alpha=0.7)
ax1.plot(range(window, num_steps+1), cal_smooth, label='Calibrated QINS', linewidth=2, alpha=0.7)
ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.3, label='Target (90%)')
ax1.set_xlabel('Decode Step')
ax1.set_ylabel('Match Rate')
ax1.set_title('Greedy Match Rate vs FP32')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

# Plot 2: Top-10 overlap comparison
ax2 = axes[0, 1]
std_top10_smooth = np.convolve(standard_top10, np.ones(window)/window, mode='valid')
cal_top10_smooth = np.convolve(calibrated_top10, np.ones(window)/window, mode='valid')
ax2.plot(range(window, num_steps+1), std_top10_smooth, label='Standard QINS', linewidth=2, alpha=0.7)
ax2.plot(range(window, num_steps+1), cal_top10_smooth, label='Calibrated QINS', linewidth=2, alpha=0.7)
ax2.axhline(y=0.95, color='green', linestyle='--', alpha=0.3, label='Target (95%)')
ax2.set_xlabel('Decode Step')
ax2.set_ylabel('Overlap Rate')
ax2.set_title('Top-10 Overlap vs FP32')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.05)

# Plot 3: Cumulative improvement
ax3 = axes[1, 0]
std_cumul = [sum(standard_matches[:i+1])/(i+1) for i in range(len(standard_matches))]
cal_cumul = [sum(calibrated_matches[:i+1])/(i+1) for i in range(len(calibrated_matches))]
ax3.plot(range(1, num_steps+1), std_cumul, label='Standard QINS', linewidth=2, alpha=0.7)
ax3.plot(range(1, num_steps+1), cal_cumul, label='Calibrated QINS', linewidth=2, alpha=0.7)
ax3.axhline(y=0.9, color='green', linestyle='--', alpha=0.3, label='Target (90%)')
ax3.set_xlabel('Decode Step')
ax3.set_ylabel('Cumulative Match Rate')
ax3.set_title('Cumulative Greedy Match Rate')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, 1.05)

# Plot 4: Summary comparison
ax4 = axes[1, 1]
ax4.axis('off')

improvement = calibrated_match_rate - standard_match_rate
improvement_symbol = "✅" if improvement > 5 else "⚠️" if improvement > 0 else "❌"

summary_text = f"""
CALIBRATION IMPACT

Standard QINS:
  Greedy match:  {standard_match_rate:.1f}%
  Top-10 overlap: {standard_top10_rate:.1f}%

Calibrated QINS:
  Greedy match:  {calibrated_match_rate:.1f}%
  Top-10 overlap: {calibrated_top10_rate:.1f}%

Improvement:
  Greedy: {improvement:+.1f}% {improvement_symbol}
  Top-10: {calibrated_top10_rate - standard_top10_rate:+.1f}%

Target: ≥90% match, ≥95% top-10

Status:
{"✅ PASS" if calibrated_match_rate >= 90 and calibrated_top10_rate >= 95 else "⚠️ NEEDS MORE TUNING"}
"""

ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()

output_file = 'calibrated_qins_comparison.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✅ Visualization saved to: {output_file}")
plt.show()

print(f"\n{'='*80}")
print("FINAL ASSESSMENT")
print(f"{'='*80}")

if calibrated_match_rate >= 90:
    print(f"\n🎉 SUCCESS! Calibration brings greedy match to {calibrated_match_rate:.1f}%")
    print(f"   Target threshold (≥90%) achieved!")
elif calibrated_match_rate > standard_match_rate + 10:
    print(f"\n✅ SIGNIFICANT IMPROVEMENT! +{improvement:.1f}% gain from calibration")
    print(f"   Still below 90% target, but trend is positive")
elif calibrated_match_rate > standard_match_rate:
    print(f"\n⚠️ MODEST IMPROVEMENT: +{improvement:.1f}% gain")
    print(f"   Calibration helps but more tuning needed")
else:
    print(f"\n❌ NO IMPROVEMENT: Calibration didn't help significantly")
    print(f"   May need different calibration strategy")

print(f"\n{'='*80}")
print("TEST COMPLETE")
print(f"{'='*80}")
