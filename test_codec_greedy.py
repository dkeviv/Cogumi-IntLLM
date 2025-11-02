"""
Test Pattern A (Codec-at-Rest) vs Failed Calibration Approach

This verifies that treating QINS as a storage codec (never computing in QINS domain)
works correctly, while the old approach (computing in QINS domain with scaling)
caused 0% match.

Key difference:
- OLD (FAILED): Compute in QINS domain, apply scales to fix statistics
  → Mixed nonlinear coordinates with linear ops → 0% match
  
- NEW (Pattern A): Compute always in FP, QINS only for storage
  → Preserves computational semantics → Should match FP32
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.qins_codec import QINSLinear


class SimpleTransformer(nn.Module):
    """Simple transformer for testing (all FP32)"""
    
    def __init__(self, vocab_size: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn_qkv': nn.Linear(hidden_dim, hidden_dim * 3),
                'attn_out': nn.Linear(hidden_dim, hidden_dim),
                'mlp_up': nn.Linear(hidden_dim, hidden_dim * 4),
                'mlp_down': nn.Linear(hidden_dim * 4, hidden_dim),
            })
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.embed.weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (simple, no attention mask)"""
        h = self.embed(x)  # (batch, seq_len, hidden)
        
        for layer in self.layers:
            # Self-attention (simplified, no proper masking)
            qkv = layer['attn_qkv'](h)
            q, k, v = qkv.chunk(3, dim=-1)
            
            # Attention scores (simplified)
            attn = torch.matmul(q, k.transpose(-1, -2)) / (q.size(-1) ** 0.5)
            attn = torch.softmax(attn, dim=-1)
            
            # Attention output
            attn_out = torch.matmul(attn, v)
            attn_out = layer['attn_out'](attn_out)
            
            h = h + attn_out
            
            # MLP
            mlp = layer['mlp_up'](h)
            mlp = torch.relu(mlp)
            mlp = layer['mlp_down'](mlp)
            
            h = h + mlp
        
        h = self.norm(h)
        logits = self.lm_head(h)
        
        return logits


def convert_to_qins_codec(model: SimpleTransformer) -> SimpleTransformer:
    """
    Convert model to use QINS codec-at-rest (Pattern A).
    
    Strategy:
    - All MLP and attention linear layers → QINSLinear
    - Embedding and LayerNorm stay in FP32
    - Forward pass computes in FP (QINSLinear decodes internally)
    """
    print("Converting to QINS codec-at-rest...")
    
    converted = 0
    for layer_idx, layer in enumerate(model.layers):
        for name, module in layer.items():
            if isinstance(module, nn.Linear):
                # Convert to QINSLinear
                qins_module = QINSLinear.from_linear(module)
                layer[name] = qins_module
                converted += 1
                print(f"  Layer {layer_idx} {name}: {module.weight.shape}")
    
    print(f"✓ Converted {converted} linear layers to QINS codec")
    return model


@torch.no_grad()
def greedy_generate(
    model: nn.Module,
    prompt: torch.Tensor,
    max_steps: int = 500
) -> torch.Tensor:
    """
    Greedy generation (deterministic).
    
    Args:
        model: Transformer model
        prompt: Initial token IDs (batch, seq_len)
        max_steps: Number of tokens to generate
        
    Returns:
        generated: Token IDs (batch, seq_len + max_steps)
    """
    model.eval()
    
    tokens = prompt.clone()
    
    for step in range(max_steps):
        # Forward pass
        logits = model(tokens)  # (batch, seq_len, vocab)
        
        # Get next token (greedy)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (batch, 1)
        
        # Append
        tokens = torch.cat([tokens, next_token], dim=1)
        
        if (step + 1) % 100 == 0:
            print(f"  Step {step + 1}/{max_steps}")
    
    return tokens


def compare_models(
    fp32_model: SimpleTransformer,
    qins_model: SimpleTransformer,
    vocab_size: int,
    num_steps: int = 500
):
    """
    Compare FP32 vs QINS codec-at-rest greedy generation.
    
    This should show HIGH match rate (>90%) because:
    - QINS is only used for storage
    - All computation happens in FP domain
    - No coordinate system mixing
    """
    print("\n" + "=" * 80)
    print("Greedy Generation Comparison")
    print("=" * 80)
    
    # Create prompt
    prompt = torch.randint(0, vocab_size, (1, 10))
    print(f"\nPrompt: {prompt.tolist()[0]}")
    
    # Generate with FP32
    print("\nGenerating with FP32 model...")
    fp32_tokens = greedy_generate(fp32_model, prompt, max_steps=num_steps)
    
    # Generate with QINS codec
    print("\nGenerating with QINS codec model...")
    qins_tokens = greedy_generate(qins_model, prompt, max_steps=num_steps)
    
    # Compare
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    
    fp32_seq = fp32_tokens[0, len(prompt[0]):].tolist()
    qins_seq = qins_tokens[0, len(prompt[0]):].tolist()
    
    # Match rate
    matches = sum(f == q for f, q in zip(fp32_seq, qins_seq))
    match_rate = matches / len(fp32_seq) * 100
    
    print(f"\nGreedy match rate: {match_rate:.2f}% ({matches}/{len(fp32_seq)})")
    
    # First divergence
    first_diff = None
    for i, (f, q) in enumerate(zip(fp32_seq, qins_seq)):
        if f != q:
            first_diff = i
            break
    
    if first_diff is not None:
        print(f"First divergence at step {first_diff}")
        start = max(0, first_diff - 3)
        end = min(len(fp32_seq), first_diff + 4)
        print(f"  FP32: {fp32_seq[start:end]}")
        print(f"  QINS: {qins_seq[start:end]}")
    else:
        print("Perfect match! No divergence.")
    
    # Top-10 overlap at divergence points
    if first_diff is not None:
        print("\n" + "-" * 80)
        print("Checking top-10 logits at first divergence point...")
        
        # Recompute logits at divergence
        context_fp32 = fp32_tokens[0, :len(prompt[0]) + first_diff].unsqueeze(0)
        context_qins = qins_tokens[0, :len(prompt[0]) + first_diff].unsqueeze(0)
        
        logits_fp32 = fp32_model(context_fp32)[0, -1, :]
        logits_qins = qins_model(context_qins)[0, -1, :]
        
        top10_fp32 = logits_fp32.topk(10).indices.tolist()
        top10_qins = logits_qins.topk(10).indices.tolist()
        
        overlap = len(set(top10_fp32) & set(top10_qins))
        
        print(f"Top-10 overlap: {overlap}/10")
        print(f"  FP32 top-10: {top10_fp32}")
        print(f"  QINS top-10: {top10_qins}")
        
        # Logit statistics
        logit_error = (logits_fp32 - logits_qins).abs().mean().item()
        print(f"Mean logit error: {logit_error:.8f}")
    
    # Memory comparison
    print("\n" + "=" * 80)
    print("Memory Usage")
    print("=" * 80)
    
    def count_params(model):
        fp32_params = 0
        qins_params = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                fp32_params += module.weight.numel()
                if module.bias is not None:
                    fp32_params += module.bias.numel()
            elif isinstance(module, QINSLinear):
                # QINS storage: 2 bytes per weight (uint8 + int8)
                qins_params += module.stored.numel()
                if module.bias is not None:
                    fp32_params += module.bias.numel()
        
        fp32_bytes = fp32_params * 4
        qins_bytes = qins_params * 2 + (fp32_params - qins_params) * 4  # QINS + remaining FP32
        
        return fp32_bytes, qins_bytes
    
    fp32_bytes, _ = count_params(fp32_model)
    _, qins_bytes = count_params(qins_model)
    
    print(f"FP32 model: {fp32_bytes / (1024**2):.2f} MB")
    print(f"QINS model: {qins_bytes / (1024**2):.2f} MB")
    print(f"Compression: {fp32_bytes / qins_bytes:.2f}×")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("Verdict")
    print("=" * 80)
    
    if match_rate >= 90:
        print("✅ EXCELLENT: Codec-at-rest achieves ≥90% match")
        print("   This proves QINS works when used as storage only!")
    elif match_rate >= 50:
        print("⚠️  MODERATE: 50-90% match")
        print("   Better than calibration (0%), but room for improvement")
    elif match_rate >= 10:
        print("⚠️  LOW: 10-50% match")
        print("   Still better than calibration approach (0%)")
    else:
        print("❌ FAILED: <10% match")
        print("   Something still wrong with implementation")
    
    print("\nComparison to previous approaches:")
    print("  Standard QINS (compute in QINS domain): 6.4% match")
    print("  Calibrated QINS (scales + QINS compute): 0.0% match ← CATASTROPHIC")
    print(f"  Codec-at-rest (Pattern A): {match_rate:.2f}% match")
    
    return match_rate


def main():
    print("=" * 80)
    print("QINS Pattern A (Codec-at-Rest) Test")
    print("=" * 80)
    print("\nThis test verifies that QINS works correctly when:")
    print("1. Used ONLY for storage (weights stored in QINS format)")
    print("2. Decoded immediately before computation")
    print("3. ALL compute happens in FP domain")
    print("\nThis avoids mixing nonlinear coordinates with linear ops.")
    print("=" * 80)
    
    # Configuration
    vocab_size = 5000
    hidden_dim = 256
    n_layers = 3
    num_steps = 500
    
    print(f"\nModel configuration:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Layers: {n_layers}")
    print(f"  Generation steps: {num_steps}")
    
    # Create FP32 model
    print("\n" + "-" * 80)
    print("Creating FP32 baseline model...")
    torch.manual_seed(42)
    fp32_model = SimpleTransformer(vocab_size, hidden_dim, n_layers)
    
    # Initialize with small weights
    for param in fp32_model.parameters():
        if param.dim() > 1:
            nn.init.normal_(param, mean=0, std=0.02)
    
    fp32_model.eval()
    print("✓ FP32 model ready")
    
    # Create QINS codec model
    print("\n" + "-" * 80)
    torch.manual_seed(42)
    qins_model = SimpleTransformer(vocab_size, hidden_dim, n_layers)
    
    # Initialize with same weights
    for param in qins_model.parameters():
        if param.dim() > 1:
            nn.init.normal_(param, mean=0, std=0.02)
    
    # Convert to QINS codec
    qins_model = convert_to_qins_codec(qins_model)
    qins_model.eval()
    print("✓ QINS codec model ready")
    
    # Compare
    match_rate = compare_models(fp32_model, qins_model, vocab_size, num_steps)
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("\nKey insight from this test:")
    print("QINS is a NONLINEAR coordinate transformation, not linear quantization!")
    print("\nCorrect usage (Pattern A - Codec-at-Rest):")
    print("  ✅ Store weights in QINS format (2× memory savings)")
    print("  ✅ Decode to FP before every computation")
    print("  ✅ Never expose QINS tensors to LayerNorm, softmax, or linear ops")
    print("  ✅ Compute always in FP domain")
    print("\nIncorrect usage (what caused 0% match):")
    print("  ❌ Compute in QINS domain (mixed nonlinear coords with linear ops)")
    print("  ❌ Apply FP weights to QINS activations")
    print("  ❌ Feed QINS tensors to LayerNorm")
    print("  ❌ Try to 'fix' with per-channel scales (α, S)")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
