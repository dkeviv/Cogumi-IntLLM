#!/usr/bin/env python3
"""Analyze the actual vs expected size of the QINS model."""

import torch
import os

model_path = 'models/phi35-qins-codec.pt'

# File size on disk
file_size = os.path.getsize(model_path)
print('=' * 70)
print('QINS Model Size Analysis')
print('=' * 70)
print()
print(f'File on disk: {file_size / (1024**3):.2f} GB')
print()

# Load model
model_data = torch.load(model_path, map_location='cpu')

# Calculate actual tensor sizes
total_tensor_bytes = 0
qins_compressed_bytes = 0
fp32_bytes = 0
meta_bytes = 0

for key, tensor in model_data.items():
    tensor_bytes = tensor.element_size() * tensor.numel()
    total_tensor_bytes += tensor_bytes
    
    if 'stored' in key or 'sign' in key:
        qins_compressed_bytes += tensor_bytes
    elif 'log_min' in key or 'log_max' in key:
        meta_bytes += tensor_bytes
    else:
        fp32_bytes += tensor_bytes

print('Raw tensor data (in memory):')
print(f'  QINS compressed: {qins_compressed_bytes / (1024**3):.2f} GB')
print(f'  FP32 parts:      {fp32_bytes / (1024**3):.2f} GB')
print(f'  Metadata:        {meta_bytes / (1024**2):.2f} MB')
print(f'  Total tensors:   {total_tensor_bytes / (1024**3):.2f} GB')
print()

# Serialization overhead
overhead = file_size - total_tensor_bytes
overhead_pct = (overhead / file_size) * 100

print('Serialization overhead:')
print(f'  Overhead: {overhead / (1024**3):.2f} GB ({overhead_pct:.1f}% of file)')
print(f'  This is PyTorch pickle metadata, keys, structure, etc.')
print()

# Expected if FP32
# QINS has separate stored+sign tensors for each weight
# So qins_compressed_bytes represents BOTH stored AND sign
# Each parameter has 1 byte stored + 1 byte sign = 2 bytes total
qins_weight_count = qins_compressed_bytes // 2  # Divide by 2 because we have both stored and sign
fp32_param_count = fp32_bytes // 4  # FP32 is 4 bytes
total_params = qins_weight_count + fp32_param_count

# If everything was FP32
expected_fp32_bytes = (qins_weight_count * 4) + fp32_bytes  # QINS weights as FP32 + existing FP32
actual_compression = expected_fp32_bytes / total_tensor_bytes

print('Compression analysis:')
print(f'  Total parameters: {total_params / 1e9:.2f}B')
print(f'  If all FP32: {expected_fp32_bytes / (1024**3):.2f} GB')
print(f'  QINS encoded: {total_tensor_bytes / (1024**3):.2f} GB')
print(f'  Compression ratio: {actual_compression:.2f}×')
print()

print('=' * 70)
print('EXPLANATION')
print('=' * 70)
print()
print('The file is 7.3 GB on disk but only ~3.7 GB of actual tensor data.')
print('The rest (~3.6 GB) is PyTorch serialization overhead.')
print()
print('When loaded into memory, the model uses only ~3.7 GB, which is')
print('about 2× smaller than the FP32 equivalent (~7.6 GB).')
print()
print('PyTorch\'s torch.save() is not optimized for INT8 storage.')
print('We can reduce this with proper compression (Huffman, etc.).')
