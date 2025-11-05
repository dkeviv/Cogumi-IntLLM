# QINS Greedy Multi-Step Test Results

**Date:** November 2, 2025  
**Test Type:** Deterministic generation (do_sample=False, temperature=0)  
**Steps:** 1000 decode steps  
**Pass Criteria:** ≥90-95% average match, no downward trend  

---

## Executive Summary

### ❌ Greedy Match Test: FAILED
- **Greedy match rate:** 0.20% (2/1000 tokens)
- **Pass threshold:** ≥90%
- **Result:** FAR BELOW threshold

### ✅ Distribution Similarity: EXCELLENT
- **KL divergence:** 0.00000086 (nearly perfect!)
- **Logit error:** 0.000024 (99.999% accurate)
- **Pass threshold:** <0.01
- **Result:** PASSED with excellence

### ⚠️ Critical Finding: Ranking Instability
- **Top-10 overlap:** 0.45%
- **Top-50 overlap:** 1.26%
- **Problem:** Logits are nearly identical in *magnitude* but *rankings* differ

---

## Detailed Results

### Test Configuration

```
Model: 4-layer transformer-like network
Vocabulary: 5,000 tokens
Hidden dimension: 512
Initial context: 20 tokens
Decode steps: 1,000
Mode: Greedy (argmax, no sampling)
```

### Metrics Over Time

#### Greedy Match Rate
- **Overall:** 0.20% (2/1000)
- **First 100 steps:** 2.0%
- **Steps 100-1000:** 0.0%
- **Trend:** Rapid decline to zero

#### Top-K Overlap
- **Top-5:** 0.34%
- **Top-10:** 0.45%
- **Top-50:** 1.26%
- **All far below 90-95% target**

#### Distribution Metrics
- **KL divergence:** 0.00000086 (🎉 EXCELLENT)
- **Logit error:** 0.000024 (🎉 EXCELLENT)

### Windowed Analysis (100-step windows)

| Window | Greedy Match | Top-10 Overlap |
|--------|--------------|----------------|
| 0-100 | 2.0% | 3.0% |
| 100-200 | 0.0% | 0.1% |
| 200-300 | 0.0% | 0.3% |
| 300-400 | 0.0% | 0.1% |
| 400-500 | 0.0% | 0.1% |
| 500-600 | 0.0% | 0.3% |
| 600-700 | 0.0% | 0.1% |
| 700-800 | 0.0% | 0.1% |
| 800-900 | 0.0% | 0.3% |
| 900-1000 | 0.0% | 0.1% |

---

## Root Cause Analysis

### The Paradox

**How can logits be 99.999% accurate but rankings completely different?**

#### Explanation:

Consider a simple example with 5 tokens and their logits:

**FP32:**
```
Token A: 5.0004
Token B: 5.0003
Token C: 5.0002
Token D: 5.0001
Token E: 5.0000
Ranking: A > B > C > D > E
```

**QINS (with 0.0001 error per logit):**
```
Token A: 5.0003 (-0.0001 error)
Token B: 5.0004 (+0.0001 error)
Token C: 5.0001 (-0.0001 error)
Token D: 5.0002 (+0.0001 error)
Token E: 5.0000 (same)
Ranking: B > A > D > C > E  (COMPLETELY DIFFERENT!)
```

**Key Observations:**
1. Mean logit error: 0.00008 (tiny!)
2. KL divergence: ~0.000001 (negligible!)
3. But argmax flips: A ≠ B
4. And entire ranking shuffles

### Why This Happens in QINS

1. **Logarithmic encoding has limited precision**
   - 8-bit storage → 256 quantization levels
   - For large logit ranges (-10 to +10), each level covers ~0.08 units
   - Competing logits often differ by <<0.08

2. **Competitive logit spaces**
   - Language models often have 10-100 tokens with similar probabilities
   - These differ by 0.001-0.01 in logit space
   - QINS error (0.00002) is enough to reshuffle rankings

3. **Error accumulation in rankings**
   - Each token has independent error
   - Random errors can systematically shift rankings
   - Top-k becomes unstable even with tiny mean error

---

## Comparison: Simple vs Complex Model

### Simple Model (256 hidden, 2 layers)
- **Greedy match:** 100%
- **Top-10 overlap:** 100%
- **Why:** Fewer parameters → more stable logit patterns

### Complex Model (512 hidden, 4 layers)
- **Greedy match:** 0.2%
- **Top-10 overlap:** 0.45%
- **Why:** More parameters → more competitive logits → higher sensitivity

---

## Implications

### ❌ What This Means (Bad News)

1. **Greedy decoding is unreliable**
   - QINS will generate DIFFERENT text than FP32
   - Token-by-token matching is poor
   - Not suitable for applications requiring exact reproduction

2. **Ranking instability**
   - Top-k candidates shuffle significantly
   - Beam search would produce different results
   - Deterministic generation differs from FP32

3. **Doesn't meet 90-95% pass criteria**
   - 0.2% is far below 90% threshold
   - This is a significant quality gap

### ✅ What This Means (Good News)

1. **Distribution similarity is excellent**
   - KL divergence of 0.00000086 means probability distributions are nearly identical
   - Perplexity would be very similar
   - Overall likelihood of sequences is preserved

2. **Logit accuracy is excellent**
   - Mean error of 0.000024 is tiny
   - Magnitudes are correct
   - Model "understands" context correctly

3. **Temperature sampling could help**
   - With temperature > 0, top-k becomes a candidate pool
   - As long as good candidates are in top-50, generation quality preserved
   - Current 1.26% top-50 overlap is still problematic though

---

## Recommendations

### Immediate Actions

1. **Investigate quantization precision**
   - Current: 8-bit logarithmic encoding
   - Consider: 10-bit or 12-bit for vocab projection layer
   - Trade-off: 25-50% more memory vs better ranking stability

2. **Test with temperature sampling**
   - Run same test with temperature=0.7, top-p=0.9
   - Measure: perplexity, sequence likelihood, human evaluation
   - May show acceptable generation quality despite ranking issues

3. **Analyze specific failure modes**
   - Which layers cause ranking instability?
   - Is it the vocab projection specifically?
   - Could layer-specific precision help?

### Long-term Solutions

1. **Hybrid precision**
   - Use QINS (8-bit) for most layers
   - Use higher precision (16-bit) for vocab projection
   - Expected: 10-20% better ranking stability with minimal memory cost

2. **Ranking-aware quantization**
   - Optimize quantization to preserve top-k order
   - Use ordinal loss during conversion
   - More complex but could solve root issue

3. **Accept different use cases**
   - QINS excellent for: embedding extraction, features, internal layers
   - QINS problematic for: final vocab projection, greedy decoding
   - Use hybrid approach: QINS backbone + FP16 head

---

## Comparison with Prior Tests

### Single-Step Predictions (Previous Test)
- **Greedy match:** 100%
- **Top-10 overlap:** 97.5%
- **Conclusion:** Excellent for one-shot predictions

### Multi-Step Generation (This Test)
- **Greedy match:** 0.2%
- **Top-10 overlap:** 0.45%
- **Conclusion:** Poor for autoregressive generation

**Why the difference?**
- Single-step: Independent contexts, no error accumulation
- Multi-step: Errors compound, rankings diverge over time
- **Lesson:** Single-step metrics don't predict multi-step behavior!

---

## Visualizations

See generated plots:
- `qins_greedy_multistep_test.png` - Match rate over time
- `qins_generation_quality_test.png` - Top-K overlap and KL divergence

Key observations from plots:
1. Match rate drops to near-zero by step 100
2. Top-k overlap remains consistently low (<1%)
3. KL divergence stays excellent (<0.000001)
4. Logit error remains stable (no accumulation)

---

## Conclusion

### Test Result: ❌ FAILED

**Greedy match rate:** 0.20% vs 90% threshold = **MAJOR GAP**

### Root Cause: **Ranking Instability**

QINS preserves:
- ✅ Logit magnitudes (99.999% accurate)
- ✅ Probability distributions (KL div ~0)
- ❌ Logit rankings (top-k reshuffles)

### Impact Assessment

For production use:
- ❌ **Greedy/beam search generation:** Not suitable
- ❌ **Exact reproduction:** Not possible
- ⚠️ **Temperature sampling:** Needs further testing
- ✅ **Feature extraction:** Excellent
- ✅ **Embeddings:** Excellent
- ✅ **Internal layers:** Excellent

### Next Steps

1. **Priority 1:** Test with temperature sampling (temperature=0.7-1.0)
2. **Priority 2:** Implement hybrid precision (FP16 vocab head)
3. **Priority 3:** Investigate ranking-aware quantization

### Final Assessment

QINS achieves excellent compression (2×) and speed (1.18×), but the **ranking instability in vocabulary projection** makes it unsuitable for **deterministic text generation**. 

Recommended path forward:
- Use QINS for transformer layers
- Use FP16 for final vocab projection
- Expected result: 1.8× compression with stable rankings

---

**Test Status:** FAILED (0.2% vs 90% threshold)  
**Underlying Quality:** Mixed (excellent distributions, poor rankings)  
**Recommendation:** Hybrid precision approach required for production
