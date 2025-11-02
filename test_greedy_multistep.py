"""
Greedy Multi-Step Deterministic Test for QINS
Tests exact next-token matching over 512-1000 decode steps
Deterministic (do_sample=False, temperature=0)
Pass criteria: ≥90-95% average match, no downward trend
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

print("=" * 80)
print("QINS GREEDY MULTI-STEP DETERMINISTIC TEST")
print("=" * 80)

# Simulate a realistic language model
class RealisticLM(nn.Module):
    """Realistic language model for testing"""
    def __init__(self, vocab_size=5000, hidden_dim=512, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        
        # Multiple transformer-like layers
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim if i == 0 else hidden_dim * 2, hidden_dim * 2)
            for i in range(num_layers)
        ])
        self.layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
        
        self.vocab_proj = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, input_ids):
        x = self.embed(input_ids)
        
        # Process through layers
        for layer in self.layers:
            x = F.gelu(layer(x))
        
        logits = self.vocab_proj(x)
        return logits

class QINSRealisticLM(nn.Module):
    """QINS version with ProjectiveLinear"""
    def __init__(self, vocab_size=5000, hidden_dim=512, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        
        # Multiple transformer-like layers
        self.layers = nn.ModuleList([
            ProjectiveLinear(hidden_dim if i == 0 else hidden_dim * 2, hidden_dim * 2)
            for i in range(num_layers)
        ])
        self.layers.append(ProjectiveLinear(hidden_dim * 2, hidden_dim))
        
        self.vocab_proj = ProjectiveLinear(hidden_dim, vocab_size)
        
    def forward(self, input_ids):
        x = self.embed(input_ids)
        
        # Process through layers
        for layer in self.layers:
            x = F.gelu(layer(x))
        
        logits = self.vocab_proj(x)
        return logits

# Configuration
vocab_size = 5000
hidden_dim = 512
num_layers = 4
num_steps = 1000  # Decode steps
start_seq_len = 20  # Initial context length

print(f"\nTest Configuration:")
print(f"  Vocabulary size: {vocab_size:,}")
print(f"  Hidden dimension: {hidden_dim}")
print(f"  Number of layers: {num_layers}")
print(f"  Decode steps: {num_steps}")
print(f"  Initial context: {start_seq_len} tokens")
print(f"  Mode: Greedy (do_sample=False, temperature=0)")

# Create FP32 model
print(f"\n{'='*80}")
print("1. Creating FP32 Model")
print(f"{'='*80}")

fp32_model = RealisticLM(vocab_size, hidden_dim, num_layers)
fp32_model.eval()

# Initialize with realistic values
for name, param in fp32_model.named_parameters():
    if 'embed' in name:
        nn.init.normal_(param, mean=0, std=0.02)
    elif 'weight' in name:
        nn.init.normal_(param, mean=0, std=0.02)
    elif 'bias' in name:
        nn.init.zeros_(param)

# Calculate memory
linear_params = sum(
    p.numel() for n, p in fp32_model.named_parameters() 
    if 'embed' not in n
)
fp32_memory = linear_params * 4 / (1024 ** 2)

print(f"✅ FP32 model created")
print(f"   Linear parameters: {linear_params:,}")
print(f"   Linear memory: {fp32_memory:.2f} MB")

# Create QINS model
print(f"\n{'='*80}")
print("2. Creating QINS Model")
print(f"{'='*80}")

qins_model = QINSRealisticLM(vocab_size, hidden_dim, num_layers)

# Copy embeddings and convert linear layers
with torch.no_grad():
    qins_model.embed.weight.copy_(fp32_model.embed.weight)
    
    # Convert all linear layers
    for fp32_layer, qins_layer in zip(fp32_model.layers, qins_model.layers):
        qins_layer.from_linear(fp32_layer)
    
    qins_model.vocab_proj.from_linear(fp32_model.vocab_proj)

qins_model.eval()

# Calculate QINS memory
qins_memory = sum(
    (m.stored.numel() + m.sign.numel()) 
    for m in qins_model.modules() 
    if isinstance(m, ProjectiveLinear)
) / (1024 ** 2)

print(f"✅ QINS model created")
print(f"   Linear memory: {qins_memory:.2f} MB")
print(f"   Compression: {fp32_memory / qins_memory:.2f}×")

# Greedy Multi-Step Generation
print(f"\n{'='*80}")
print("3. Running Greedy Multi-Step Generation")
print(f"{'='*80}")

# Set random seed for reproducibility
torch.manual_seed(42)

# Initial context (same for both models)
initial_context = torch.randint(0, vocab_size, (1, start_seq_len))

fp32_sequence = initial_context.clone()
qins_sequence = initial_context.clone()

# Track metrics
match_history = []  # Per-step match (1 or 0)
logit_errors = []   # Per-step logit error
cumulative_matches = 0

print(f"\nStarting generation from {start_seq_len} token context...")
print(f"Target: {num_steps} greedy decode steps")
print(f"\nProgress (every 50 steps):")

with torch.no_grad():
    for step in range(num_steps):
        # FP32 prediction (greedy)
        fp32_logits = fp32_model(fp32_sequence)
        fp32_next_token = fp32_logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
        
        # QINS prediction (greedy)
        qins_logits = qins_model(qins_sequence)
        qins_next_token = qins_logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
        
        # Calculate metrics
        token_match = (fp32_next_token.item() == qins_next_token.item())
        match_history.append(1 if token_match else 0)
        
        logit_error = (fp32_logits[0, -1, :] - qins_logits[0, -1, :]).abs().mean().item()
        logit_errors.append(logit_error)
        
        if token_match:
            cumulative_matches += 1
        
        # Progress reporting
        if (step + 1) % 50 == 0:
            current_match_rate = cumulative_matches / (step + 1)
            print(f"  Step {step+1:4d}: Match rate = {current_match_rate*100:5.1f}% "
                  f"(Last 50: {sum(match_history[-50:])*2}%)")
        
        # Append tokens to sequences
        fp32_sequence = torch.cat([fp32_sequence, fp32_next_token], dim=1)
        qins_sequence = torch.cat([qins_sequence, qins_next_token], dim=1)

# Calculate statistics
total_match_rate = sum(match_history) / len(match_history)
avg_logit_error = sum(logit_errors) / len(logit_errors)

print(f"\n{'='*80}")
print("4. Results Analysis")
print(f"{'='*80}")

print(f"\n📊 Overall Statistics:")
print(f"  Total steps: {num_steps}")
print(f"  Exact matches: {sum(match_history)}/{num_steps}")
print(f"  Overall match rate: {total_match_rate*100:.2f}%")
print(f"  Average logit error: {avg_logit_error:.6f}")

# Windowed analysis (100-step windows)
window_size = 100
windows = []
for i in range(0, len(match_history), window_size):
    window = match_history[i:i+window_size]
    if len(window) == window_size:
        windows.append(sum(window) / len(window))

print(f"\n📈 Windowed Analysis ({window_size}-step windows):")
for i, rate in enumerate(windows):
    start = i * window_size
    end = start + window_size
    print(f"  Steps {start:4d}-{end:4d}: {rate*100:5.1f}%")

# Detect trend
if len(windows) >= 2:
    first_half_avg = sum(windows[:len(windows)//2]) / (len(windows)//2)
    second_half_avg = sum(windows[len(windows)//2:]) / (len(windows) - len(windows)//2)
    trend_diff = second_half_avg - first_half_avg
    
    print(f"\n📉 Trend Analysis:")
    print(f"  First half average: {first_half_avg*100:.2f}%")
    print(f"  Second half average: {second_half_avg*100:.2f}%")
    print(f"  Trend: {trend_diff*100:+.2f}%", end="")
    
    if trend_diff > 0.01:
        print(" (✅ Improving)")
        trend_status = "improving"
    elif trend_diff < -0.01:
        print(" (⚠️  Declining)")
        trend_status = "declining"
    else:
        print(" (✅ Stable)")
        trend_status = "stable"

# Pass/Fail Assessment
print(f"\n{'='*80}")
print("5. Pass/Fail Assessment")
print(f"{'='*80}")

pass_threshold_min = 0.90  # 90%
pass_threshold_target = 0.95  # 95%

print(f"\n🎯 Criteria:")
print(f"  Minimum pass: ≥{pass_threshold_min*100:.0f}% average match")
print(f"  Target: ≥{pass_threshold_target*100:.0f}% average match")
print(f"  Trend: No downward trend")

print(f"\n📋 Results:")
print(f"  Average match rate: {total_match_rate*100:.2f}%", end=" ")

if total_match_rate >= pass_threshold_target:
    print("🎉 EXCELLENT")
    pass_status = "EXCELLENT"
elif total_match_rate >= pass_threshold_min:
    print("✅ PASS")
    pass_status = "PASS"
else:
    print("❌ FAIL")
    pass_status = "FAIL"

print(f"  Trend: {trend_status}", end=" ")
if trend_status != "declining":
    print("✅ GOOD")
else:
    print("⚠️  CONCERN")

# Overall verdict
print(f"\n{'='*80}")
print("FINAL VERDICT")
print(f"{'='*80}")

if total_match_rate >= pass_threshold_min and trend_status != "declining":
    print(f"\n🎉 TEST PASSED!")
    print(f"\nQINS maintains {total_match_rate*100:.1f}% exact token match over {num_steps} greedy steps.")
    print(f"This demonstrates excellent deterministic behavior with no quality degradation.")
else:
    print(f"\n⚠️  TEST NEEDS ATTENTION")
    if total_match_rate < pass_threshold_min:
        print(f"Match rate ({total_match_rate*100:.1f}%) below minimum threshold ({pass_threshold_min*100:.0f}%)")
    if trend_status == "declining":
        print(f"Declining trend detected - may indicate error accumulation")

# Create visualization
print(f"\n{'='*80}")
print("6. Creating Visualizations")
print(f"{'='*80}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'QINS Greedy Multi-Step Test - {num_steps} Steps', fontsize=16, fontweight='bold')

# Plot 1: Match rate over time (raw)
ax1 = axes[0, 0]
ax1.plot(range(1, num_steps+1), match_history, alpha=0.3, linewidth=0.5, label='Per-step match')

# Smooth with moving average
window = 50
smoothed = np.convolve(match_history, np.ones(window)/window, mode='valid')
ax1.plot(range(window, num_steps+1), smoothed, linewidth=2, color='blue', label=f'{window}-step moving avg')

ax1.axhline(y=pass_threshold_target, color='green', linestyle='--', alpha=0.5, label='Target (95%)')
ax1.axhline(y=pass_threshold_min, color='orange', linestyle='--', alpha=0.5, label='Min pass (90%)')
ax1.set_xlabel('Decode Step', fontsize=12)
ax1.set_ylabel('Match (1=Yes, 0=No)', fontsize=12)
ax1.set_title('Per-Step Exact Token Match', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.1, 1.1)

# Plot 2: Cumulative match rate
ax2 = axes[0, 1]
cumulative_rate = [sum(match_history[:i+1])/(i+1) for i in range(len(match_history))]
ax2.plot(range(1, num_steps+1), cumulative_rate, linewidth=2, color='purple')
ax2.axhline(y=pass_threshold_target, color='green', linestyle='--', alpha=0.5, label='Target (95%)')
ax2.axhline(y=pass_threshold_min, color='orange', linestyle='--', alpha=0.5, label='Min pass (90%)')
ax2.axhline(y=total_match_rate, color='red', linestyle='-', alpha=0.7, 
            label=f'Final: {total_match_rate*100:.1f}%')
ax2.set_xlabel('Decode Step', fontsize=12)
ax2.set_ylabel('Cumulative Match Rate', fontsize=12)
ax2.set_title('Cumulative Match Rate Over Time', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.05)

# Plot 3: Windowed match rates
ax3 = axes[1, 0]
window_centers = [i * window_size + window_size//2 for i in range(len(windows))]
ax3.bar(window_centers, windows, width=window_size*0.8, alpha=0.7, color='steelblue')
ax3.axhline(y=pass_threshold_target, color='green', linestyle='--', alpha=0.5, label='Target (95%)')
ax3.axhline(y=pass_threshold_min, color='orange', linestyle='--', alpha=0.5, label='Min pass (90%)')
ax3.axhline(y=total_match_rate, color='red', linestyle='-', alpha=0.7, 
            label=f'Average: {total_match_rate*100:.1f}%')
ax3.set_xlabel('Decode Step', fontsize=12)
ax3.set_ylabel('Match Rate', fontsize=12)
ax3.set_title(f'{window_size}-Step Window Match Rates', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, 1.05)

# Plot 4: Logit error over time
ax4 = axes[1, 1]
ax4.plot(range(1, num_steps+1), logit_errors, alpha=0.3, linewidth=0.5, color='red', label='Per-step error')

# Smooth logit errors
smoothed_errors = np.convolve(logit_errors, np.ones(window)/window, mode='valid')
ax4.plot(range(window, num_steps+1), smoothed_errors, linewidth=2, color='darkred', 
         label=f'{window}-step moving avg')

ax4.axhline(y=avg_logit_error, color='blue', linestyle='--', alpha=0.5, 
            label=f'Average: {avg_logit_error:.6f}')
ax4.set_xlabel('Decode Step', fontsize=12)
ax4.set_ylabel('Mean Absolute Logit Error', fontsize=12)
ax4.set_title('Logit Error Over Time', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_yscale('log')

plt.tight_layout()

# Save figure
output_file = 'qins_greedy_multistep_test.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✅ Visualization saved to: {output_file}")

# Show plot
print(f"✅ Displaying plot...")
plt.show()

# Summary table
print(f"\n{'='*80}")
print("SUMMARY TABLE")
print(f"{'='*80}")

print(f"""
┌────────────────────────────────────────┬─────────────────────────────┐
│ Metric                                 │ Value                       │
├────────────────────────────────────────┼─────────────────────────────┤
│ Total decode steps                     │ {num_steps:,}                    │
│ Exact token matches                    │ {sum(match_history):,} / {num_steps:,}        │
│ Overall match rate                     │ {total_match_rate*100:.2f}%                  │
│ Average logit error                    │ {avg_logit_error:.6f}               │
│ Memory compression                     │ {fp32_memory/qins_memory:.2f}×                   │
│ First half match rate                  │ {first_half_avg*100:.2f}%                │
│ Second half match rate                 │ {second_half_avg*100:.2f}%                │
│ Trend                                  │ {trend_status.upper():15s}       │
│ Pass threshold (min)                   │ {pass_threshold_min*100:.0f}%                    │
│ Pass threshold (target)                │ {pass_threshold_target*100:.0f}%                    │
│ Status                                 │ {pass_status:15s}       │
└────────────────────────────────────────┴─────────────────────────────┘
""")

print(f"{'='*80}")
print("TEST COMPLETE")
print(f"{'='*80}")
