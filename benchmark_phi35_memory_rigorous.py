#!/usr/bin/env python3
"""
Rigorous Memory Benchmark for QINS Pattern A on Phi-3.5

Measures:
1. Actual weight tensor memory (bytes)
2. Total model memory (including buffers)
3. Compression ratio
4. Generation quality (token match)

No approximations - direct measurement from tensor storage.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import gc

# Add src to path
sys.path.insert(0, 'src')

from qins_weight_codec import convert_linear_to_qins, QINSWeightLinear


def get_model_memory_breakdown(model):
    """
    Get detailed memory breakdown of model.
    Returns bytes for each category.
    """
    weight_memory = 0
    buffer_memory = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        size_bytes = param.numel() * param.element_size()
        weight_memory += size_bytes
        total_params += param.numel()
    
    for name, buffer in model.named_buffers():
        size_bytes = buffer.numel() * buffer.element_size()
        buffer_memory += size_bytes
    
    return {
        'weight_memory_bytes': weight_memory,
        'buffer_memory_bytes': buffer_memory,
        'total_memory_bytes': weight_memory + buffer_memory,
        'total_params': total_params
    }


def get_qins_layer_memory(model):
    """
    Get memory usage specifically from QINS-encoded layers.
    """
    qins_memory = 0
    qins_param_count = 0
    fp32_equiv_memory = 0
    
    for name, module in model.named_modules():
        if isinstance(module, QINSWeightLinear):
            # QINS encoded weights
            if hasattr(module, 'w_encoded'):
                qins_size = module.w_encoded.numel() * module.w_encoded.element_size()
                qins_memory += qins_size
                qins_param_count += module.w_encoded.numel()
                
                # FP32 equivalent would be 4 bytes per element
                fp32_equiv_memory += module.w_encoded.numel() * 4
            
            # Bias (if exists)
            if hasattr(module, 'bias') and module.bias is not None:
                bias_size = module.bias.numel() * module.bias.element_size()
                qins_memory += bias_size
                fp32_equiv_memory += bias_size  # Bias stays FP32
    
    return {
        'qins_memory_bytes': qins_memory,
        'qins_param_count': qins_param_count,
        'fp32_equiv_bytes': fp32_equiv_memory,
        'compression_ratio': fp32_equiv_memory / qins_memory if qins_memory > 0 else 1.0
    }


def compare_models():
    """
    Compare FP32 vs QINS model memory usage.
    """
    print("="*70)
    print("RIGOROUS MEMORY BENCHMARK - Phi-3.5-mini Pattern A")
    print("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test prompt
    prompt = "The capital of France is"
    
    # ========================================================================
    # BASELINE: FP32 Model
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: Measuring FP32 Baseline")
    print("="*70)
    
    print("\nLoading FP32 model...")
    model_fp32 = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model_fp32.eval()
    
    fp32_breakdown = get_model_memory_breakdown(model_fp32)
    
    print("\nFP32 Model Memory:")
    print(f"  Parameters:  {fp32_breakdown['weight_memory_bytes'] / (1024**2):.2f} MB")
    print(f"  Buffers:     {fp32_breakdown['buffer_memory_bytes'] / (1024**2):.2f} MB")
    print(f"  Total:       {fp32_breakdown['total_memory_bytes'] / (1024**2):.2f} MB")
    print(f"  Param count: {fp32_breakdown['total_params']:,}")
    
    # Generate baseline
    print("\nGenerating baseline...")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        output_fp32 = model_fp32.generate(
            input_ids,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    tokens_fp32 = output_fp32[0].tolist()
    text_fp32 = tokenizer.decode(output_fp32[0], skip_special_tokens=False)
    print(f"Generated: '{text_fp32}'")
    
    # Clear memory
    del model_fp32
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ========================================================================
    # QINS Model
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: Measuring QINS Model")
    print("="*70)
    
    print("\nLoading model for QINS conversion...")
    model_qins = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model_qins.eval()
    
    # Measure before conversion
    before_breakdown = get_model_memory_breakdown(model_qins)
    print(f"\nBefore QINS conversion: {before_breakdown['total_memory_bytes'] / (1024**2):.2f} MB")
    
    # Convert to QINS
    print("\nConverting to QINS (v_proj, o_proj, gate_proj, up_proj, down_proj)...")
    target_names = ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model_qins = convert_linear_to_qins(model_qins, target_names=target_names, verbose=False)
    
    # Measure after conversion
    after_breakdown = get_model_memory_breakdown(model_qins)
    qins_stats = get_qins_layer_memory(model_qins)
    
    print("\nQINS Model Memory:")
    print(f"  Parameters:  {after_breakdown['weight_memory_bytes'] / (1024**2):.2f} MB")
    print(f"  Buffers:     {after_breakdown['buffer_memory_bytes'] / (1024**2):.2f} MB")
    print(f"  Total:       {after_breakdown['total_memory_bytes'] / (1024**2):.2f} MB")
    
    print("\nQINS-Encoded Layers Only:")
    print(f"  QINS storage:     {qins_stats['qins_memory_bytes'] / (1024**2):.2f} MB")
    print(f"  FP32 equivalent:  {qins_stats['fp32_equiv_bytes'] / (1024**2):.2f} MB")
    print(f"  Params encoded:   {qins_stats['qins_param_count']:,}")
    print(f"  Compression:      {qins_stats['compression_ratio']:.2f}×")
    
    # Generate with QINS
    print("\nGenerating with QINS...")
    with torch.no_grad():
        output_qins = model_qins.generate(
            input_ids,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    tokens_qins = output_qins[0].tolist()
    text_qins = tokenizer.decode(output_qins[0], skip_special_tokens=False)
    print(f"Generated: '{text_qins}'")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # Token match
    match_count = sum(1 for a, b in zip(tokens_fp32, tokens_qins) if a == b)
    match_pct = match_count / len(tokens_fp32) * 100
    
    print("\nQuality:")
    print(f"  Token match: {match_count}/{len(tokens_fp32)} ({match_pct:.1f}%)")
    if match_pct == 100:
        print("  ✅ PERFECT MATCH")
    else:
        print("  ❌ MISMATCH DETECTED")
    
    print("\nMemory (Total Model):")
    print(f"  FP32:  {fp32_breakdown['total_memory_bytes'] / (1024**2):.2f} MB")
    print(f"  QINS:  {after_breakdown['total_memory_bytes'] / (1024**2):.2f} MB")
    print(f"  Saved: {(fp32_breakdown['total_memory_bytes'] - after_breakdown['total_memory_bytes']) / (1024**2):.2f} MB")
    
    total_compression = fp32_breakdown['total_memory_bytes'] / after_breakdown['total_memory_bytes']
    print(f"  Total model compression: {total_compression:.2f}×")
    
    print("\nMemory (QINS-Encoded Layers Only):")
    print(f"  FP32 equivalent: {qins_stats['fp32_equiv_bytes'] / (1024**2):.2f} MB")
    print(f"  QINS storage:    {qins_stats['qins_memory_bytes'] / (1024**2):.2f} MB")
    print(f"  Saved:           {(qins_stats['fp32_equiv_bytes'] - qins_stats['qins_memory_bytes']) / (1024**2):.2f} MB")
    print(f"  Compression:     {qins_stats['compression_ratio']:.2f}×")
    
    # Check if quantization is actually enabled
    print("\n" + "="*70)
    print("QUANTIZATION CHECK")
    print("="*70)
    
    quantized = False
    for name, module in model_qins.named_modules():
        if isinstance(module, QINSWeightLinear):
            if hasattr(module, 'w_encoded'):
                dtype = module.w_encoded.dtype
                print(f"\nFirst QINS layer found: {name}")
                print(f"  Encoded weight dtype: {dtype}")
                print(f"  Element size: {module.w_encoded.element_size()} bytes")
                
                if dtype == torch.uint8:
                    print("  ✅ QUANTIZED (uint8)")
                    quantized = True
                elif dtype == torch.float32:
                    print("  ❌ NOT QUANTIZED (float32)")
                    quantized = False
                else:
                    print(f"  ⚠️  UNEXPECTED dtype: {dtype}")
                break
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if match_pct == 100:
        print("✅ Quality: PASS (100% token match)")
    else:
        print("❌ Quality: FAIL (mismatch detected)")
    
    if quantized:
        print(f"✅ Quantization: ENABLED (uint8)")
        print(f"✅ Compression: {qins_stats['compression_ratio']:.2f}× in encoded layers")
    else:
        print("❌ Quantization: DISABLED (float32)")
        print("⚠️  No actual compression (storing as float32)")
    
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    if match_pct == 100 and not quantized:
        print("✅ Pattern A codec: VALIDATED (lossless)")
        print("⚠️  Compression: NOT ENABLED (need to enable quantization)")
        print("\nNext step: Enable quantization in convert_linear_to_qins()")
    elif match_pct == 100 and quantized:
        print("✅ Pattern A codec: VALIDATED (lossless)")
        print(f"✅ Compression: WORKING ({qins_stats['compression_ratio']:.2f}× in encoded layers)")
    else:
        print("❌ FAILED: Token mismatch detected")
        print("⚠️  Do not enable quantization until codec is fixed")


if __name__ == "__main__":
    compare_models()
