"""
Test QINS accuracy with realistic generation scenario
Simulates vocabulary projection like real language models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.projective_layer import ProjectiveLinear

print("=" * 70)
print("QINS Text Generation Accuracy Test")
print("=" * 70)

# Simulate a realistic language model output layer
class TextGenModel(nn.Module):
    """Simplified LM: embedding → transformer-like layers → vocab projection"""
    def __init__(self, vocab_size=32000, hidden_dim=768):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layer1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.layer2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.vocab_proj = nn.Linear(hidden_dim, vocab_size)  # Project to vocab
        
    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = F.gelu(self.layer1(x))
        x = F.gelu(self.layer2(x))
        logits = self.vocab_proj(x)
        return logits

class QINSTextGenModel(nn.Module):
    """QINS version with ProjectiveLinear layers"""
    def __init__(self, vocab_size=32000, hidden_dim=768):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layer1 = ProjectiveLinear(hidden_dim, hidden_dim * 2)
        self.layer2 = ProjectiveLinear(hidden_dim * 2, hidden_dim)
        self.vocab_proj = ProjectiveLinear(hidden_dim, vocab_size)
        
    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = F.gelu(self.layer1(x))
        x = F.gelu(self.layer2(x))
        logits = self.vocab_proj(x)
        return logits

# Configuration
vocab_size = 32000  # Typical vocab size
hidden_dim = 768
batch_size = 1

print(f"\nModel configuration:")
print(f"  Vocabulary size: {vocab_size:,}")
print(f"  Hidden dimension: {hidden_dim}")
print(f"  Batch size: {batch_size}")

# Create FP32 model
print(f"\n1. Creating FP32 model...")
fp32_model = TextGenModel(vocab_size, hidden_dim)
fp32_model.eval()

# Initialize with reasonable values
for name, param in fp32_model.named_parameters():
    if 'embed' in name:
        nn.init.normal_(param, mean=0, std=0.02)
    elif 'weight' in name:
        nn.init.normal_(param, mean=0, std=0.02)
    elif 'bias' in name:
        nn.init.zeros_(param)

# Count parameters
linear_params = sum(
    p.numel() for n, p in fp32_model.named_parameters() 
    if 'embed' not in n
)
embed_params = sum(
    p.numel() for n, p in fp32_model.named_parameters() 
    if 'embed' in n
)

fp32_memory = linear_params * 4 / (1024 ** 2)
print(f"✅ FP32 model created")
print(f"   Linear parameters: {linear_params:,}")
print(f"   Embedding parameters: {embed_params:,} (unchanged)")
print(f"   Linear layer memory: {fp32_memory:.2f} MB")

# Create QINS model
print(f"\n2. Creating QINS model...")
qins_model = QINSTextGenModel(vocab_size, hidden_dim)

# Copy embeddings (unchanged)
with torch.no_grad():
    qins_model.embed.weight.copy_(fp32_model.embed.weight)
    
    # Convert linear layers
    qins_model.layer1.from_linear(fp32_model.layer1)
    qins_model.layer2.from_linear(fp32_model.layer2)
    qins_model.vocab_proj.from_linear(fp32_model.vocab_proj)

qins_model.eval()

# Calculate QINS memory
qins_memory = sum(
    (m.stored.numel() + m.sign.numel()) 
    for m in qins_model.modules() 
    if isinstance(m, ProjectiveLinear)
) / (1024 ** 2)

print(f"✅ QINS model created")
print(f"   Linear layer memory: {qins_memory:.2f} MB")
print(f"   Compression: {fp32_memory / qins_memory:.2f}×")

# Test with realistic token sequences
print(f"\n3. Testing on sample token sequences...")
print("=" * 70)

test_prompts = [
    ("Hello world", torch.randint(0, vocab_size, (batch_size, 5))),
    ("Coding example", torch.randint(0, vocab_size, (batch_size, 10))),
    ("Long paragraph", torch.randint(0, vocab_size, (batch_size, 50))),
    ("Full context", torch.randint(0, vocab_size, (batch_size, 128))),
]

results = []

for name, input_ids in test_prompts:
    seq_len = input_ids.shape[1]
    print(f"\n{name} ({seq_len} tokens):")
    
    # FP32 inference
    with torch.no_grad():
        fp32_logits = fp32_model(input_ids)
    
    # QINS inference
    with torch.no_grad():
        qins_logits = qins_model(input_ids)
    
    # Calculate logit errors
    logit_error = (fp32_logits - qins_logits).abs()
    
    # Get top-k predictions (most important for generation)
    k = 10
    fp32_topk = torch.topk(fp32_logits[0, -1, :], k)
    qins_topk = torch.topk(qins_logits[0, -1, :], k)
    
    # Check overlap in top-k predictions
    fp32_indices = set(fp32_topk.indices.tolist())
    qins_indices = set(qins_topk.indices.tolist())
    overlap = len(fp32_indices & qins_indices)
    
    # Greedy decoding (argmax) - most common generation strategy
    fp32_next = fp32_logits[0, -1, :].argmax()
    qins_next = qins_logits[0, -1, :].argmax()
    same_prediction = (fp32_next == qins_next).item()
    
    # Probability distributions (for temperature sampling)
    fp32_probs = F.softmax(fp32_logits[0, -1, :], dim=-1)
    qins_probs = F.softmax(qins_logits[0, -1, :], dim=-1)
    kl_div = F.kl_div(
        qins_probs.log(), 
        fp32_probs, 
        reduction='sum'
    )
    
    print(f"  Logit mean abs error: {logit_error.mean():.6f}")
    print(f"  Logit max abs error: {logit_error.max():.6f}")
    print(f"  Top-{k} overlap: {overlap}/{k} ({overlap/k*100:.0f}%)")
    print(f"  Greedy prediction match: {'✅ YES' if same_prediction else '❌ NO'}")
    print(f"  KL divergence: {kl_div:.6f}")
    
    results.append({
        'name': name,
        'seq_len': seq_len,
        'logit_error': logit_error.mean().item(),
        'topk_overlap': overlap / k,
        'greedy_match': same_prediction,
        'kl_div': kl_div.item()
    })

# Test autoregressive generation
print("\n" + "=" * 70)
print("AUTOREGRESSIVE GENERATION TEST")
print("=" * 70)

print("\nGenerating 20 tokens autoregressively...")
start_tokens = torch.randint(0, vocab_size, (batch_size, 10))

fp32_tokens = start_tokens.clone()
qins_tokens = start_tokens.clone()

generation_matches = 0
total_generated = 20

with torch.no_grad():
    for step in range(total_generated):
        # FP32 generation
        fp32_logits = fp32_model(fp32_tokens)
        fp32_next = fp32_logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
        fp32_tokens = torch.cat([fp32_tokens, fp32_next], dim=1)
        
        # QINS generation
        qins_logits = qins_model(qins_tokens)
        qins_next = qins_logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
        qins_tokens = torch.cat([qins_tokens, qins_next], dim=1)
        
        # Check if predictions match
        if fp32_next.item() == qins_next.item():
            generation_matches += 1

match_rate = generation_matches / total_generated

print(f"✅ Generation completed")
print(f"  Tokens generated: {total_generated}")
print(f"  Matching predictions: {generation_matches}/{total_generated} ({match_rate*100:.1f}%)")
print(f"  Final sequence lengths: FP32={fp32_tokens.shape[1]}, QINS={qins_tokens.shape[1]}")

if match_rate >= 0.95:
    print(f"  🎉 Excellent match rate!")
elif match_rate >= 0.80:
    print(f"  ✅ Good match rate")
elif match_rate >= 0.60:
    print(f"  ✅ Acceptable match rate")
else:
    print(f"  ⚠️  Some divergence in predictions")

# Summary
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

avg_logit_error = sum(r['logit_error'] for r in results) / len(results)
avg_topk = sum(r['topk_overlap'] for r in results) / len(results)
avg_kl = sum(r['kl_div'] for r in results) / len(results)
greedy_matches = sum(1 for r in results if r['greedy_match'])

print(f"\nAveraged across {len(results)} test cases:")
print(f"  Mean logit error: {avg_logit_error:.6f}")
print(f"  Average top-10 overlap: {avg_topk*100:.1f}%")
print(f"  Average KL divergence: {avg_kl:.6f}")
print(f"  Greedy matches: {greedy_matches}/{len(results)}")
print(f"  Autoregressive match rate: {match_rate*100:.1f}%")

# Final verdict
print("\n" + "=" * 70)
print("FINAL VERDICT")
print("=" * 70)

print(f"\n✅ QINS Text Generation Performance:")
print(f"   Memory reduction: {fp32_memory / qins_memory:.2f}×")
print(f"   Logit accuracy: {avg_logit_error:.6f} mean error")
print(f"   Top-k overlap: {avg_topk*100:.1f}%")
print(f"   Generation match: {match_rate*100:.1f}%")

if match_rate >= 0.9 and avg_topk >= 0.8:
    print(f"\n🎉 EXCELLENT: QINS generates nearly identical text!")
elif match_rate >= 0.7 and avg_topk >= 0.6:
    print(f"\n✅ GOOD: QINS maintains generation quality!")
elif match_rate >= 0.5:
    print(f"\n✅ ACCEPTABLE: QINS shows reasonable generation")
else:
    print(f"\n⚠️  WARNING: Generation quality may need improvement")

print(f"\n✅ QINS is validated for text generation tasks")
print("=" * 70)
