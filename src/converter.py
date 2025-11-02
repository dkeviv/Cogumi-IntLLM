"""
FILE: converter.py
PURPOSE: Convert entire models from nn.Linear to ProjectiveLinear
DEPENDENCIES: torch.nn, projective_layer.ProjectiveLinear

CRITICAL CONCEPTS:
- Must recursively traverse all submodules
- Use setattr() to replace modules in parent
- Preserve all non-Linear modules unchanged
- Track conversion count for validation

ALGORITHM:
1. Define recursive helper
2. For each module child:
   - If nn.Linear: create ProjectiveLinear, copy weights, replace
   - Else: recurse into child's children
3. Return modified model (in-place)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Any

from .projective_layer import ProjectiveLinear


def convert_model_to_projective(
    model: nn.Module,
    scale: int = 256,
    verbose: bool = True
) -> nn.Module:
    """
    Convert all nn.Linear layers in model to ProjectiveLinear.
    
    Traverses model recursively and replaces each Linear layer with
    ProjectiveLinear while preserving learned weights.
    
    Args:
        model: PyTorch model to convert (e.g., HuggingFace model)
        scale: Projective scale factor
        verbose: Whether to print progress
    
    Returns:
        Modified model (in-place modification)
    
    Algorithm:
        1. Initialize conversion counter
        2. Define recursive helper function:
            a. For each child in module.named_children():
                - Check if isinstance(child, nn.Linear)
                - If Linear: create ProjectiveLinear, convert, replace
                - If not: recurse into child
        3. Call helper on root module
        4. Print summary if verbose
    
    Example:
        >>> model = AutoModelForCausalLM.from_pretrained('microsoft/Phi-3.5-mini-instruct')
        >>> model = convert_model_to_projective(model)
        >>> # All Linear layers now ProjectiveLinear
    """
    converted_count = 0
    
    def _convert_recursive(module: nn.Module, name: str = ''):
        """Recursively convert Linear layers."""
        nonlocal converted_count
        
        # Iterate through module's children
        for child_name, child in list(module.named_children()):
            # Build full name
            full_name = f"{name}.{child_name}" if name else child_name
            
            # Check if this is a Linear layer
            if isinstance(child, nn.Linear):
                # Create ProjectiveLinear with same dimensions
                proj_layer = ProjectiveLinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    scale=scale
                )
                
                # Convert weights from FP32 to QINS
                proj_layer.from_linear(child)
                
                # Replace in parent module
                setattr(module, child_name, proj_layer)
                
                # Increment counter
                converted_count += 1
                
                # Print progress if verbose
                if verbose:
                    print(f"  ✓ Converted: {full_name} "
                          f"[{child.in_features} → {child.out_features}]")
            else:
                # Recursively process child modules
                _convert_recursive(child, full_name)
    
    if verbose:
        print("=" * 60)
        print("Converting model to Projective INT8")
        print("=" * 60)
    
    # Call recursive helper
    _convert_recursive(model)
    
    if verbose:
        print(f"\n✓ Converted {converted_count} Linear layers")
        print("=" * 60)
    
    return model


def measure_conversion_error(
    original_layer: nn.Linear,
    projective_layer: ProjectiveLinear,
    num_samples: int = 100
) -> Tuple[float, float, float]:
    """
    Measure error introduced by projective conversion.
    
    Generates random inputs, passes through both layers, computes error.
    
    Args:
        original_layer: Original nn.Linear layer
        projective_layer: Converted ProjectiveLinear layer
        num_samples: Number of random test inputs
    
    Returns:
        (mean_absolute_error, max_absolute_error, mean_relative_error)
    
    Algorithm:
        1. Generate random inputs: torch.randn(num_samples, in_features)
        2. Forward through original: y_orig
        3. Forward through converted: y_conv
        4. Compute absolute error: |y_orig - y_conv|
        5. Compute relative error: |y_orig - y_conv| / (|y_orig| + 1e-8)
        6. Return statistics
    
    Acceptance Criteria:
        - mean_absolute_error < 0.5
        - max_absolute_error < 2.0
        - mean_relative_error < 0.05 (5%)
    """
    # Generate random inputs
    x = torch.randn(num_samples, original_layer.in_features)
    
    # Forward through both layers
    with torch.no_grad():
        y_orig = original_layer(x)
        y_conv = projective_layer(x)
    
    # Compute errors
    abs_error = torch.abs(y_orig - y_conv)
    rel_error = abs_error / (torch.abs(y_orig) + 1e-8)
    
    # Return statistics
    mean_abs_error = abs_error.mean().item()
    max_abs_error = abs_error.max().item()
    mean_rel_error = rel_error.mean().item()
    
    return mean_abs_error, max_abs_error, mean_rel_error


def get_model_statistics(model: nn.Module) -> Dict[str, Any]:
    """
    Collect statistics about model composition.
    
    Useful for validation and debugging.
    
    Returns:
        Dictionary with:
        - total_params: Total parameter count
        - linear_layers: Number of Linear layers
        - projective_layers: Number of ProjectiveLinear layers
        - memory_fp32_gb: Estimated FP32 memory (GB)
        - memory_int8_gb: Estimated INT8 memory (GB)
        - compression_ratio: FP32 / INT8
    """
    total_params = 0
    linear_layers = 0
    projective_layers = 0
    fp32_memory = 0
    int8_memory = 0
    
    def _count_recursive(module: nn.Module):
        nonlocal total_params, linear_layers, projective_layers, fp32_memory, int8_memory
        
        for child in module.children():
            if isinstance(child, nn.Linear):
                linear_layers += 1
                params = child.in_features * child.out_features
                if child.bias is not None:
                    params += child.out_features
                total_params += params
                fp32_memory += params * 4  # 4 bytes per FP32
            elif isinstance(child, ProjectiveLinear):
                projective_layers += 1
                params = child.in_features * child.out_features
                if child.bias is not None:
                    params += child.out_features
                total_params += params
                # stored (1 byte) + sign (1 byte) + bias (4 bytes if present)
                int8_memory += child.in_features * child.out_features * 2
                if child.bias is not None:
                    int8_memory += child.out_features * 4
                # Add LUT size (256 * 4 bytes)
                int8_memory += 256 * 4
            else:
                _count_recursive(child)
    
    _count_recursive(model)
    
    return {
        'total_params': total_params,
        'linear_layers': linear_layers,
        'projective_layers': projective_layers,
        'memory_fp32_gb': fp32_memory / (1024 ** 3),
        'memory_int8_gb': int8_memory / (1024 ** 3),
        'compression_ratio': fp32_memory / int8_memory if int8_memory > 0 else 0
    }
