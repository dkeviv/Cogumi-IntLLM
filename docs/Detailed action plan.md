Here’s a crisp, step-by-step action plan in tables so there’s no ambiguity.
A) Pattern-A Hardening (lossless; storage-only)
Step
Goal
Scope (what to touch)
Change to Make
Validation (minimal)
Success Criteria
Quantization Allowed?
Guardrails / Don’ts
Exit Gate
A1
Prove codec is lossless
QINS encode/decode fns
Keep FP compute; ensure encode→decode round-trips
Round-trip stats on tensors you wrap: cosine ≥ 0.999999, max err ≤ 1e-6
Pass on all tested tensors
No
Don’t alter LN/softmax/Q/K paths
All checks pass
A2
Greedy parity
Full model (Phi-3.5)
No code changes; just A/B run
5 prompts × 1,000 steps greedy; token-by-token match
100% match on all runs
No
Fresh cache per prompt; right padding
100% match
A3
Sampling sanity
Full model
No code changes
1 prompt set; temp=0.7, top_p=0.9, 256 tokens; KL(logits), top-k overlap
KL ~ FP noise; top-10 ≥ 97%
No
Don’t over-interpret wording diffs
KL/overlap OK
A4
Bit-pack storage
KV-V (and optional big activations)
Pack QINS values to 8/6 bits; unpack→decode right before compute
Microbench pack/unpack; end-to-end greedy 1k steps
≤1% latency overhead; 100% match
No (still lossless if you pack integers with full precision)
Don’t touch K/Q/LN/softmax
Overhead acceptable
A5
Fused decode path
Matmul boundary only
Move decode closer to GEMM (custom op/Triton stub)
Microbench vs Python decode; 1k-step greedy
Faster or equal; 100% match
No
Keep API identical to FP path
Perf OK

B) Transition to QINS Compute (Pattern-B bring-up)
Step
Goal
Scope (what to touch)
Change to Make
Validation (minimal)
Success Criteria
Quantization Allowed?
Guardrails / Don’ts
Exit Gate
B1
Safe low-risk quantization
KV-V only
QINS-INT8 (optionally INT6) for V in cache; dequant→decode before use
Greedy 1k steps; KL on logits; long-context 2k–8k
Greedy ≥ 98–100%; KL small/flat
Yes (KV-V only)
Don’t quantize Q/K/LN/emb/lm_head
Metrics OK
B2
Weight-transport prototype
v_proj only(one block)
Compute per-channel Jacobians; build W′; run QINS-domain v-proj
Layer microbench; short greedy (256)
Stable outputs; speedup ≥ FP decode path
No (beyond KV-V)
Don’t touch LN/softmax/Q/K
Stable + faster
B3
QINS matmul op (v-proj)
v_proj
Implement QINS_matmul(E(x), W′) → QINS(out); decode after block
Layer-level speed & correctness
>1.2× layer speed; block-level parity
No
Keep fallback to FP path
Meets speed & parity
B4
Extend transport
MLP down → up → attn out_proj
Repeat B2–B3 for each; one at a time
Per-path greedy 1k; KL; latency
Parity maintained; net speed gain
No
Q/K still FP; LN unchanged
All extended
B5
KV fast-path
KV-V
Fused decompress→QINS_matmul(skip FP decode)
End-to-end throughput on long context
Bandwidth & latency win
Yes (KV-V)
Ensure exact cache lengths
Perf OK

C) Full QINS Inference Engine & Training Prep
Step
Goal
Scope (what to touch)
Change to Make
Validation (minimal)
Success Criteria
Quantization Allowed?
Guardrails / Don’ts
Exit Gate
C1
Full QINS inference (ex-Q/K)
All Linear paths except Q/K, LN, emb, lm_head
Transport remaining weights; QINS-domain compute; decode only at block boundaries
Greedy 1k; sampling; ΔPPL small; long-context
Parity within noise; clear speed/energy gains
Yes (KV-V; optional acts)
Keep Q/K/LN/emb/lm_head in FP for now
Parity + speed
C2
Compression polish
Weights W′, KV, acts
Entropy coding where profitable; fused decompress→MAC
Perf & memory benchmarks
Net win over bit-pack; no regressions
Yes
Avoid adding CPU bottlenecks
Net benefit
C3
Training pilot (QINS-SGD/Adam)
Small transformer
Implement QINS-domain grads & optimizer; train from scratch
Loss curve vs FP; final eval
Converges stably; parity/negligible gap
N/A
Start with INT8-like precision
Stable training
C4
Expand training
Larger models
Curriculum to bigger LLMs; add QINS-aware LN if needed
Standard evals
Competitive or better
N/A
Keep baseline FP run for control
Meets bar



Phase 3 — QINS Training Program (C1 → C5)
C1 — Pilot: Train a Small Transformer From Scratch (QINS-native)
Item
Plan
Objective
Prove QINS-domain training stability and convergence on a small model.
Model
GPT-2-small–class: 12 layers, d_model 768, n_heads 12 (~125–160M params).
Data
TinyPile or OpenWebText2 subset (5–10B tokens) or 50–100GB text; 1 epoch target.
Arithmetic
Forward in QINS domain for designated Linear paths (those already transported in Phase B), decode only at block exits. Keep Q/K, LN, embeddings, lm_head in FP16/BF16.
Optimizer
QINS-AdamW (AdamW with per-tensor preconditioning in QINS coords); LR warmup + cosine decay; wd 0.1; β=(0.9, 0.95).
Batch/Seq
Global tokens/sec tuned for your hardware; start with seq_len=1024, global batch = 512–1024 seqs (accumulation if needed).
Deliverables
Loss curve vs FP baseline; training logs; ablation: with/without transport; checkpoint at 10%, 50%, 100%.
Validation
Perplexity on WikiText-2/PP; greedy 1k match vs FP teacher on held-out prompts; sampling KL/logit cosine.
Success
Converges without instability; PPL within +3% of FP baseline; no gradient pathologies.
Risks & Guards
If gradients spike: enable gradient clipping 1.0; reduce LR by 20%; fallback to partial QINS coverage (V-path + MLP down first).


C2 — Finetune FP Model in QINS Domain (Bridge FP → QINS)
Item
Plan
Objective
Demonstrate smooth adaptation of an FP-pretrained model into QINS compute regime.
Base Model
Phi-2 (2.7B) or Mistral-7B, whichever infra supports; start with 1–3B range if resources limited.
Conversion
Apply Phase-B transport to covered Linear layers (exclude Q/K, LN, emb, lm_head).
Training
1–3 epochs on curated corpus (RedPajama/OpenHermes blend); LR 1e-5–5e-5; QINS-AdamW.
Validation
Task PPL + small evals (HellaSwag, PIQA, ARC-easy) vs original FP checkpoint.
Success
PPL regression ≤ 1%; downstream eval within noise; stable loss.
Risks & Guards
If loss bumps >1%: reduce transport coverage (only v_proj + MLP down), lower LR, freeze embeddings.


C3 — Distillation: FP Teacher → QINS Student
Item
Plan
Objective
Recover/boost quality and regularize QINS quirks via teacher signals.
Teacher
FP model (same architecture/size as student).
Student
QINS-domain (transported) model initialized from C2 or from scratch (for small scale).
Losses
CE(y, y*) + λ_KL·KL(logits_QINS‖logits_FP) + (optional) λ_hid·MSE on select hidden states. Start λ_KL=0.5, λ_hid=0.05.
Schedule
1–2 epochs; small LR (5e-6–1e-5); mixed batches (teacher frozen).
Validation
Same as C2 + greedy match drift curves (1k) to ensure autoregressive stability.
Success
PPL meets or beats FP baseline on held-out; sampling quality subjective parity; no long-horizon drift.
Risks & Guards
If KL dominates, anneal λ_KL; if training slows, drop λ_hid.


C4 — Medium-Scale QINS Pretraining (Show Viability at Scale)
Item
Plan
Objective
Prove QINS training viability beyond toy scale.
Model
1–3B parameters (24–32 layers, d_model≈2048).
Data
100–300B tokens mixed corpus; standard dedup/filtering.
Infra
Multi-node; checkpointing with QINS-INT8 at rest for activations; ZeRO-style optimizer sharding.
Arithmetic
Maximize QINS coverage per B-phase results; keep exclusion list (Q/K, LN, emb, lm_head) in FP16/BF16.
Validation
Weekly PPL snapshots; periodic eval suite; long-context stability checks.
Success
Convergence envelope matches FP runs; throughput & memory metrics show clear wins (≥1.3–2× effective tokens/$ vs FP16).
Risks & Guards
If divergence late: lower LR; enable EMA; widen LN eps; rollback transport on suspect paths.


C5 — QINS-Native Foundation Model (Milestone)
Item
Plan
Objective
First end-to-end QINS-trained LLM (public milestone).
Model
≥7B parameters (or 1–3B if resources constrained) trained from scratch in QINS domain.
Training
Full corpus (multi-hundred B tokens), full QINS coverage per learnings; consider QINS-aware LayerNorm variant (affine in FP, normalize with decoded stats or QINS-native norm).
Distillation Boost
Optional final pass: FP teacher KL to polish.
Validation
Full eval battery (MMLU, HellaSwag, GSM8K few-shot, HumanEval w/ prompt caching controls).
Success
Competitive with FP peers at similar scale; throughput/watt advantage demonstrated.
Launch Artifacts
Tech report, training card, ablations (coverage, bit-depths), kernels appendix, reproducibility details.


Cross-Cutting Tracks (run in parallel)
Track
What to build
When used
Eval Harness
Round-trip metrics, greedy 1k, sampling KL, PPL, long-context curves
C1 onward
Kernel Path
Fused decode, QINS matmul (from Phase B), pack/unpack ops
C1 onward
Optimizer
QINS-AdamW + hooks for per-tensor α (density) scheduling
C1 onward
Logging
Scalars + histograms for encoded/decoded distributions; Jacobian stats
C1 onward
Safety
FP fallback toggles per layer; anomaly detection (grad norms, NaNs)
Always


Gating & Go/No-Go Criteria
Gate
Proceed if
Else do
C1 → C2
QINS-tiny converges; PPL within +3% FP
Reduce coverage; tune LR/clip; fix optimizer preconditioning
C2 → C3
Finetune parity within 1% PPL
Roll back transport on weak paths; lower LR; add KL warmup
C3 → C4
Distilled student ≥ teacher or within noise
Increase distillation data/epochs; tune λ_KL/λ_hid
C4 → C5
1–3B pretrain stable; tokens/$ win ≥1.3×
Optimize kernels; re-balance QINS coverage; improve pack/decode


Precision & Coverage Policy (to avoid confusion)
Keep FP (BF16/FP16): Q, K, LayerNorm (stats+affine), embeddings, lm_head (until dedicated study).
QINS-domain (transported): v_proj, attn out_proj, MLP up/down (expand gradually).
At-rest compression: KV-V QINS-INT8; optional activations QINS-INT8 (decode before compute).
Bit-depth experiments: start INT8; evaluate INT6 on KV-V & some acts after stability.
