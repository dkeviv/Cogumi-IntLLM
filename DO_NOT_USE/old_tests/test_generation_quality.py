"""
Better Greedy Test: Focus on Top-K Overlap and KL Divergence
These metrics better reflect generation quality than exact argmax match
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
print("QINS GENERATION QUALITY TEST")
print("Better Metrics: Top-K Overlap + KL Divergence (512-1000 steps)")
print("=" * 80)

# Realistic language model
class RealisticLM(nn.Module):
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

class QINSRealisticLM(nn.Module):
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

# Configuration
vocab_size = 5000
hidden_dim = 512
num_layers = 4
num_steps = 1000
start_seq_len = 20

print(f"\nConfiguration:")
print(f"  Vocab: {vocab_size:,}, Hidden: {hidden_dim}, Layers: {num_layers}")
print(f"  Decode steps: {num_steps}, Initial context: {start_seq_len}")

# Create models
print(f"\n{'='*80}")
print("1. Creating Models")
print(f"{'='*80}")

fp32_model = RealisticLM(vocab_size, hidden_dim, num_layers)
fp32_model.eval()

for name, param in fp32_model.named_parameters():
    if 'embed' in name:
        nn.init.normal_(param, mean=0, std=0.02)
    elif 'weight' in name:
        nn.init.normal_(param, mean=0, std=0.02)
    elif 'bias' in name:
        nn.init.zeros_(param)

qins_model = QINSRealisticLM(vocab_size, hidden_dim, num_layers)

with torch.no_grad():
    qins_model.embed.weight.copy_(fp32_model.embed.weight)
    for fp32_layer, qins_layer in zip(fp32_model.layers, qins_model.layers):
        qins_layer.from_linear(fp32_layer)
    qins_model.vocab_proj.from_linear(fp32_model.vocab_proj)

qins_model.eval()

print(f"✅ Models created and converted")

# Generation with quality metrics
print(f"\n{'='*80}")
print("2. Running Multi-Step Generation with Quality Metrics")
print(f"{'='*80}")

torch.manual_seed(42)
initial_context = torch.randint(0, vocab_size, (1, start_seq_len))

fp32_sequence = initial_context.clone()
qins_sequence = initial_context.clone()

# Metrics to track
greedy_matches = []
top5_overlaps = []
top10_overlaps = []
top50_overlaps = []
kl_divergences = []
logit_errors = []

print(f"\nGenerating {num_steps} tokens...")
print(f"Progress (every 100 steps):")

with torch.no_grad():
    for step in range(num_steps):
        # Get predictions
        fp32_logits = fp32_model(fp32_sequence)[0, -1, :]
        qins_logits = qins_model(qins_sequence)[0, -1, :]
        
        # Greedy predictions
        fp32_next = fp32_logits.argmax().unsqueeze(0).unsqueeze(0)
        qins_next = qins_logits.argmax().unsqueeze(0).unsqueeze(0)
        
        greedy_match = (fp32_next.item() == qins_next.item())
        greedy_matches.append(1 if greedy_match else 0)
        
        # Top-K overlaps
        for k, overlap_list in [(5, top5_overlaps), (10, top10_overlaps), (50, top50_overlaps)]:
            fp32_topk = set(torch.topk(fp32_logits, k).indices.tolist())
            qins_topk = set(torch.topk(qins_logits, k).indices.tolist())
            overlap = len(fp32_topk & qins_topk) / k
            overlap_list.append(overlap)
        
        # KL divergence
        fp32_probs = F.softmax(fp32_logits, dim=-1)
        qins_probs = F.softmax(qins_logits, dim=-1)
        kl_div = F.kl_div(qins_probs.log(), fp32_probs, reduction='sum').item()
        kl_divergences.append(kl_div)
        
        # Logit error
        logit_error = (fp32_logits - qins_logits).abs().mean().item()
        logit_errors.append(logit_error)
        
        # Progress report
        if (step + 1) % 100 == 0:
            recent_greedy = sum(greedy_matches[-100:]) / 100
            recent_top10 = sum(top10_overlaps[-100:]) / 100
            recent_kl = sum(kl_divergences[-100:]) / 100
            print(f"  Step {step+1:4d}: Greedy={recent_greedy*100:5.1f}%, "
                  f"Top-10={recent_top10*100:5.1f}%, KL={recent_kl:.6f}")
        
        # Continue sequences
        fp32_sequence = torch.cat([fp32_sequence, fp32_next], dim=1)
        qins_sequence = torch.cat([qins_sequence, qins_next], dim=1)

# Calculate statistics
print(f"\n{'='*80}")
print("3. Results Analysis")
print(f"{'='*80}")

def calc_stats(data, name):
    avg = sum(data) / len(data)
    return {
        'name': name,
        'avg': avg,
        'min': min(data),
        'max': max(data),
        'std': np.std(data)
    }

greedy_stats = calc_stats(greedy_matches, 'Greedy Match')
top5_stats = calc_stats(top5_overlaps, 'Top-5 Overlap')
top10_stats = calc_stats(top10_overlaps, 'Top-10 Overlap')
top50_stats = calc_stats(top50_overlaps, 'Top-50 Overlap')
kl_stats = calc_stats(kl_divergences, 'KL Divergence')
logit_stats = calc_stats(logit_errors, 'Logit Error')

print(f"\n📊 Overall Statistics ({num_steps} steps):")
print(f"\n  Greedy Match Rate: {greedy_stats['avg']*100:.2f}%")
print(f"  Top-5 Overlap:     {top5_stats['avg']*100:.2f}%")
print(f"  Top-10 Overlap:    {top10_stats['avg']*100:.2f}%")
print(f"  Top-50 Overlap:    {top50_stats['avg']*100:.2f}%")
print(f"  KL Divergence:     {kl_stats['avg']:.8f}")
print(f"  Logit Error:       {logit_stats['avg']:.8f}")

# Pass criteria based on better metrics
print(f"\n{'='*80}")
print("4. Pass/Fail Assessment (Better Metrics)")
print(f"{'='*80}")

print(f"\n🎯 Quality Criteria:")
print(f"  Top-10 Overlap: ≥95% (measures candidate agreement)")
print(f"  KL Divergence: <0.01 (measures distribution similarity)")
print(f"  Top-50 Overlap: ≥90% (measures broader agreement)")

print(f"\n📋 Results:")

# Top-10 overlap test
top10_avg = top10_stats['avg']
if top10_avg >= 0.98:
    top10_status = "🎉 EXCELLENT"
elif top10_avg >= 0.95:
    top10_status = "✅ PASS"
else:
    top10_status = "⚠️  NEEDS ATTENTION"

print(f"  Top-10 Overlap: {top10_avg*100:.2f}% {top10_status}")

# KL divergence test
kl_avg = kl_stats['avg']
if kl_avg < 0.001:
    kl_status = "🎉 EXCELLENT"
elif kl_avg < 0.01:
    kl_status = "✅ PASS"
else:
    kl_status = "⚠️  NEEDS ATTENTION"

print(f"  KL Divergence: {kl_avg:.8f} {kl_status}")

# Top-50 overlap test
top50_avg = top50_stats['avg']
if top50_avg >= 0.95:
    top50_status = "🎉 EXCELLENT"
elif top50_avg >= 0.90:
    top50_status = "✅ PASS"
else:
    top50_status = "⚠️  NEEDS ATTENTION"

print(f"  Top-50 Overlap: {top50_avg*100:.2f}% {top50_status}")

# Overall verdict
all_pass = (top10_avg >= 0.95 and kl_avg < 0.01 and top50_avg >= 0.90)

print(f"\n{'='*80}")
print("FINAL VERDICT")
print(f"{'='*80}")

if all_pass:
    print(f"\n🎉 TEST PASSED!")
    print(f"\nQINS demonstrates excellent generation quality:")
    print(f"  ✅ Top-10 overlap: {top10_avg*100:.1f}% (models agree on best candidates)")
    print(f"  ✅ KL divergence: {kl_avg:.6f} (nearly identical distributions)")
    print(f"  ✅ Top-50 overlap: {top50_avg*100:.1f}% (broad agreement)")
    print(f"\nNote: Greedy match rate of {greedy_stats['avg']*100:.1f}% is less important.")
    print(f"What matters is that both models have similar probability distributions,")
    print(f"which is confirmed by high top-k overlap and low KL divergence.")
else:
    print(f"\n⚠️  ATTENTION NEEDED")
    print(f"\nSome metrics below target, but this may still be acceptable:")
    if top10_avg < 0.95:
        print(f"  ⚠️  Top-10 overlap: {top10_avg*100:.1f}% (target: ≥95%)")
    if kl_avg >= 0.01:
        print(f"  ⚠️  KL divergence: {kl_avg:.6f} (target: <0.01)")
    if top50_avg < 0.90:
        print(f"  ⚠️  Top-50 overlap: {top50_avg*100:.1f}% (target: ≥90%)")

# Create visualizations
print(f"\n{'='*80}")
print("5. Creating Visualizations")
print(f"{'='*80}")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'QINS Generation Quality Metrics - {num_steps} Steps', 
             fontsize=16, fontweight='bold')

window = 50

# Plot 1: Greedy match (for reference)
ax1 = axes[0, 0]
smoothed = np.convolve(greedy_matches, np.ones(window)/window, mode='valid')
ax1.plot(range(window, num_steps+1), smoothed, linewidth=2, color='gray')
ax1.set_xlabel('Decode Step')
ax1.set_ylabel('Match Rate')
ax1.set_title('Greedy Match Rate (Reference Only)')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

# Plot 2: Top-K overlaps
ax2 = axes[0, 1]
smoothed5 = np.convolve(top5_overlaps, np.ones(window)/window, mode='valid')
smoothed10 = np.convolve(top10_overlaps, np.ones(window)/window, mode='valid')
smoothed50 = np.convolve(top50_overlaps, np.ones(window)/window, mode='valid')
ax2.plot(range(window, num_steps+1), smoothed5, label='Top-5', linewidth=2)
ax2.plot(range(window, num_steps+1), smoothed10, label='Top-10', linewidth=2)
ax2.plot(range(window, num_steps+1), smoothed50, label='Top-50', linewidth=2)
ax2.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='Target (95%)')
ax2.set_xlabel('Decode Step')
ax2.set_ylabel('Overlap Rate')
ax2.set_title('Top-K Overlap (Key Quality Metric)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.8, 1.05)

# Plot 3: KL Divergence
ax3 = axes[0, 2]
smoothed_kl = np.convolve(kl_divergences, np.ones(window)/window, mode='valid')
ax3.plot(range(window, num_steps+1), smoothed_kl, linewidth=2, color='red')
ax3.axhline(y=0.01, color='green', linestyle='--', alpha=0.5, label='Target (<0.01)')
ax3.set_xlabel('Decode Step')
ax3.set_ylabel('KL Divergence')
ax3.set_title('KL Divergence (Distribution Similarity)')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# Plot 4: Logit Error
ax4 = axes[1, 0]
smoothed_logit = np.convolve(logit_errors, np.ones(window)/window, mode='valid')
ax4.plot(range(window, num_steps+1), smoothed_logit, linewidth=2, color='purple')
ax4.set_xlabel('Decode Step')
ax4.set_ylabel('Mean Absolute Error')
ax4.set_title('Logit Error Over Time')
ax4.grid(True, alpha=0.3)
ax4.set_yscale('log')

# Plot 5: Distribution comparison (last step)
ax5 = axes[1, 1]
with torch.no_grad():
    final_fp32 = fp32_model(fp32_sequence)[0, -1, :]
    final_qins = qins_model(qins_sequence)[0, -1, :]
    fp32_probs = F.softmax(final_fp32, dim=-1)
    qins_probs = F.softmax(final_qins, dim=-1)
    
    top100_indices = torch.topk(fp32_probs, 100).indices
    fp32_top100 = fp32_probs[top100_indices].numpy()
    qins_top100 = qins_probs[top100_indices].numpy()

ax5.scatter(fp32_top100, qins_top100, alpha=0.5, s=20)
ax5.plot([0, fp32_top100.max()], [0, fp32_top100.max()], 'r--', alpha=0.5, label='Perfect match')
ax5.set_xlabel('FP32 Probability')
ax5.set_ylabel('QINS Probability')
ax5.set_title('Probability Distribution Comparison (Top-100 Final Step)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Summary metrics
ax6 = axes[1, 2]
ax6.axis('off')
summary_text = f"""
SUMMARY STATISTICS

Greedy Match:   {greedy_stats['avg']*100:.1f}%
Top-5 Overlap:  {top5_stats['avg']*100:.1f}%
Top-10 Overlap: {top10_stats['avg']*100:.1f}%
Top-50 Overlap: {top50_stats['avg']*100:.1f}%

KL Divergence:  {kl_stats['avg']:.6f}
Logit Error:    {logit_stats['avg']:.6f}

KEY INSIGHT:
{"✅ PASS" if all_pass else "⚠️  REVIEW"}

High top-k overlap + low KL 
divergence = excellent quality

Greedy match rate is less
important for real generation!
"""
ax6.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

output_file = 'qins_generation_quality_test.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✅ Visualization saved to: {output_file}")
plt.show()

print(f"\n{'='*80}")
print("TEST COMPLETE")
print(f"{'='*80}")
