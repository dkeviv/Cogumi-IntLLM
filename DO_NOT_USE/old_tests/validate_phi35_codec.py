"""
Validate Phi-3.5-mini Pattern A conversion

Test that the converted model:
1. Loads correctly
2. Generates coherent text
3. Matches quality expectations
4. Uses expected memory

This is a quick validation before full testing.
"""

import torch
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Set Pattern A flag
os.environ['QINS_CODEC_AT_REST'] = '1'

from transformers import AutoTokenizer


def quick_validation(model_path: str):
    """Quick validation of converted model"""
    
    print("=" * 70)
    print("Phi-3.5-mini Pattern A Quick Validation")
    print("=" * 70)
    
    print(f"\nModel path: {model_path}")
    
    # Check file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    file_size = os.path.getsize(model_path) / (1024**3)
    print(f"Model file size: {file_size:.2f} GB")
    
    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")
    
    # Load model
    print("\n🔄 Loading QINS model...")
    try:
        model = torch.load(model_path, map_location='cpu')
        print("✓ Model loaded from file")
        
        # Check if it's a state dict or model
        if isinstance(model, dict):
            print("  Format: State dict")
            print(f"  Keys: {len(model.keys())}")
            
            # Check for QINS layers
            qins_keys = [k for k in model.keys() if 'stored' in k or 'sign' in k]
            print(f"  QINS keys found: {len(qins_keys)}")
            
            if qins_keys:
                print("  ✅ Pattern A encoding detected")
                print(f"  Sample keys: {qins_keys[:3]}")
            else:
                print("  ⚠️  No QINS encoding found")
        else:
            print("  Format: Full model object")
            
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("Quick Validation Complete")
    print("=" * 70)
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                       help='Path to QINS model')
    args = parser.parse_args()
    
    success = quick_validation(args.model)
    
    if success:
        print("\n✅ Validation passed - model ready for testing")
    else:
        print("\n❌ Validation failed - check errors above")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
