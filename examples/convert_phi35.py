"""
FILE: convert_phi35.py
PURPOSE: Convert Phi-3.5-mini to QINS and compress
DEPENDENCIES: transformers, torch, src modules

WORKFLOW:
1. Download Phi-3.5-mini from HuggingFace (~7.6 GB FP32)
2. Convert all Linear layers to ProjectiveLinear
3. Verify conversion accuracy
4. Compress with sparsity + Huffman
5. Save compressed model (~1.9 GB)

USAGE:
    python examples/convert_phi35.py --output models/phi35-qins.compressed
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.converter import convert_model_to_projective, measure_conversion_error, get_model_statistics
from src.model_loader import save_compressed_model
from src.compression import ProjectiveCompressor


def main():
    parser = argparse.ArgumentParser(description="Convert Phi-3.5 to QINS")
    parser.add_argument(
        "--output",
        type=str,
        default="models/phi35-qins.compressed",
        help="Output path for compressed model"
    )
    parser.add_argument(
        "--skip-compression",
        action="store_true",
        help="Skip compression (save uncompressed QINS)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/Phi-3.5-mini-instruct",
        help="HuggingFace model ID"
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Converting Phi-3.5-mini to QINS")
    print("=" * 60)
    
    # Step 1: Load FP32 model
    print("\n📥 Downloading Phi-3.5-mini from HuggingFace...")
    print("(This will download ~7.6 GB)")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters)")
    
    # Get FP32 statistics
    print("\n📊 FP32 Model Statistics:")
    fp32_stats = get_model_statistics(model)
    print(f"  Total parameters: {fp32_stats['total_params'] / 1e9:.2f}B")
    print(f"  Memory size: {fp32_stats['memory_fp32_gb']:.2f} GB")
    print(f"  Linear layers: {fp32_stats['linear_layers']}")
    
    # Step 2: Convert to QINS
    print("\n🔄 Converting to QINS (INT8 projective)...")
    
    model_qins = convert_model_to_projective(model, scale=256, verbose=True)
    
    print("✓ Conversion complete")
    
    # Get QINS statistics
    print("\n📊 QINS Model Statistics:")
    qins_stats = get_model_statistics(model_qins)
    print(f"  Total parameters: {qins_stats['total_params'] / 1e9:.2f}B")
    print(f"  Memory size: {qins_stats['memory_int8_gb']:.2f} GB")
    print(f"  Projective layers: {qins_stats['projective_layers']}")
    print(f"  Compression ratio: {qins_stats['compression_ratio']:.2f}×")
    
    # Step 3: Verify accuracy
    print("\n✅ Verifying conversion accuracy...")
    print("(Testing sample layers)")
    
    # Find first few Linear/ProjectiveLinear layers to test
    linear_count = 0
    for name, module in model_qins.named_modules():
        if hasattr(module, 'stored') and hasattr(module, 'sign'):  # ProjectiveLinear
            linear_count += 1
            if linear_count <= 3:  # Test first 3 layers
                # Get original layer from FP32 model
                original_module = model
                for attr in name.split('.'):
                    if attr:
                        original_module = getattr(original_module, attr)
                
                # Measure error (requires both layer objects)
                if hasattr(original_module, 'weight'):
                    try:
                        mean_abs_err, max_abs_err, mean_rel_err = measure_conversion_error(
                            original_module,
                            module,
                            num_samples=100
                        )
                        print(f"  Layer {name}:")
                        print(f"    Mean absolute error: {mean_abs_err:.6f}")
                        print(f"    Max absolute error: {max_abs_err:.6f}")
                        print(f"    Mean relative error: {mean_rel_err:.6f}")
                    except Exception as e:
                        print(f"  Layer {name}: Could not measure error ({str(e)})")
    
    # Step 4: Compress
    if not args.skip_compression:
        print("\n🗜️  Compressing weights...")
        
        compressor = ProjectiveCompressor(phase=1)  # Phase 1: sparsity + Huffman
        save_compressed_model(model_qins, str(args.output), compressor)
        
        # Get compressed file size
        compressed_size_mb = output_path.stat().st_size / (1024 ** 2)
        print(f"✓ Saved to {args.output}")
        print(f"  Compressed size: {compressed_size_mb:.2f} MB")
        print(f"  Total compression: {fp32_stats['memory_fp32_gb'] * 1024 / compressed_size_mb:.2f}×")
    else:
        print("\n💾 Saving uncompressed QINS model...")
        torch.save(model_qins.state_dict(), args.output)
        
        # Get file size
        file_size_mb = output_path.stat().st_size / (1024 ** 2)
        print(f"✓ Saved to {args.output}")
        print(f"  File size: {file_size_mb:.2f} MB")
    
    print("\n" + "=" * 60)
    print("✓ CONVERSION COMPLETE")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Test inference:")
    print(f"   python examples/demo_chat.py --model {args.output}")
    print(f"2. Or load in Python:")
    print(f"   from src.model_loader import QINSModelLoader")
    print(f"   loader = QINSModelLoader()")
    print(f"   model, tokenizer = loader.load('{args.output}')")


if __name__ == "__main__":
    main()
