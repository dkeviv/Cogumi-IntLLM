"""
FILE: model_loader.py
PURPOSE: Load compressed QINS models for inference
DEPENDENCIES: torch, transformers, compression.py

CRITICAL CONCEPTS:
- Load compressed weights from disk
- Decompress through pipeline (reverse of compression)
- Reconstruct QINS model structure
- Move to appropriate device (CPU/MPS)

WORKFLOW:
1. Load compressed file
2. Decompress weights
3. Load model architecture
4. Replace Linear layers with ProjectiveLinear
5. Load decompressed weights into model
6. Move to device
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pathlib import Path
import pickle
from typing import Tuple, Optional
from .projective_layer import ProjectiveLinear
from .compression import ProjectiveCompressor
from .converter import convert_model_to_projective


class QINSModelLoader:
    """
    Load and prepare QINS models for inference.
    
    Handles:
    - Decompression of stored weights
    - Model architecture reconstruction
    - Device placement (CPU/MPS/CUDA)
    - Memory-efficient loading
    
    Usage:
        loader = QINSModelLoader()
        model, tokenizer = loader.load("models/phi35-qins-compressed.bin")
        model.eval()
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize loader.
        
        Args:
            device: Target device. If None, auto-detect (MPS > CUDA > CPU)
        """
        if device is None:
            device = self._detect_device()
        
        self.device = device
        self.compressor = ProjectiveCompressor()
        
        print(f"QINSModelLoader initialized on device: {device}")
    
    def _detect_device(self) -> str:
        """
        Auto-detect best available device.
        
        Priority: MPS (M4) > CUDA > CPU
        
        Returns:
            Device string: "mps", "cuda", or "cpu"
        
        Algorithm:
        1. Check torch.backends.mps.is_available() for M4
        2. Check torch.cuda.is_available() for NVIDIA
        3. Fallback to CPU
        """
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def load(
        self, 
        compressed_path: str,
        model_name: str = "microsoft/Phi-3.5-mini-instruct"
    ) -> Tuple[nn.Module, AutoTokenizer]:
        """
        Load compressed QINS model.
        
        Args:
            compressed_path: Path to .compressed file
            model_name: HuggingFace model ID (for architecture)
        
        Returns:
            (model, tokenizer) tuple ready for inference
        
        Algorithm:
        1. Load compressed file from disk
        2. Decompress weights using ProjectiveCompressor
        3. Load base model architecture (empty weights)
        4. Convert to ProjectiveLinear layers
        5. Load decompressed QINS weights
        6. Move to device
        7. Load tokenizer
        8. Return (model, tokenizer)
        """
        print(f"Loading compressed model from {compressed_path}...")
        
        # Load compressed file
        with open(compressed_path, 'rb') as f:
            compressed_data = f.read()
        
        # Decompress weights
        weights_dict = self.compressor.decompress(compressed_data)
        
        print(f"Decompressed {len(weights_dict)} weight tensors")
        
        # Load model architecture (empty)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        
        # Convert to QINS
        model = convert_model_to_projective(model, verbose=False)
        
        # Convert numpy arrays to torch tensors
        torch_weights = {}
        for name, weight in weights_dict.items():
            torch_weights[name] = torch.from_numpy(weight)
        
        # Load decompressed weights into model
        model.load_state_dict(torch_weights, strict=False)
        
        # Move to device
        model = model.to(self.device)
        model.eval()  # Set to inference mode
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✓ Model loaded on {self.device}")
        
        return model, tokenizer
    
    def load_from_pretrained(
        self,
        model_name: str = "microsoft/Phi-3.5-mini-instruct"
    ) -> Tuple[nn.Module, AutoTokenizer]:
        """
        Load and convert model from HuggingFace (no compression).
        
        Use this for quick testing without pre-compressed weights.
        Will download full FP32 model then convert to QINS.
        
        Args:
            model_name: HuggingFace model ID
        
        Returns:
            (model, tokenizer) in QINS format
        
        WARNING: This loads FP32 first (uses ~8GB), then converts.
        For production, use pre-compressed weights with load().
        """
        print(f"Loading {model_name} from HuggingFace...")
        print("WARNING: This downloads FP32 weights (~7.6 GB)")
        
        # Load FP32 model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        # Convert to QINS
        model = convert_model_to_projective(model, verbose=True)
        
        # Move to device
        model = model.to(self.device)
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer


def save_compressed_model(
    model: nn.Module,
    save_path: str,
    compressor: Optional[ProjectiveCompressor] = None
):
    """
    Save QINS model in compressed format.
    
    Args:
        model: QINS model (with ProjectiveLinear layers)
        save_path: Where to save .compressed file
        compressor: Compression instance (creates if None)
    
    Algorithm:
    1. Extract all ProjectiveLinear weight tensors
    2. Create weights_dict {layer_name: stored_weights}
    3. Compress using ProjectiveCompressor
    4. Save to disk
    """
    if compressor is None:
        compressor = ProjectiveCompressor()
    
    print(f"Saving compressed model to {save_path}...")
    
    # Extract QINS weights
    weights_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, ProjectiveLinear):
            weights_dict[f"{name}.stored"] = module.stored.cpu().numpy()
            weights_dict[f"{name}.sign"] = module.sign.cpu().numpy()
    
    # Compress
    compressed_data = compressor.compress(weights_dict)
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        f.write(compressed_data)
    
    print(f"✓ Saved compressed model ({len(compressed_data)} bytes)")
