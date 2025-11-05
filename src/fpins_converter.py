"""
FPINS Converter Prototype (FP32 <-> FPINS)

- Default depth: 2 (produces 3 bytes per weight)
- Uses the QINS mapping conventions defined in qins_lookup_tables.py
- Encoding algorithm: factor target product P = 256 / |w| into (L+1) integers in [1,256]
  using a greedy root-based allocation, then map to binary storage (0-255)
- Decoding: reconstruct product P' and return sign * (256 / P')

This is a prototype for experimentation and benchmarking.
"""

from typing import Tuple, List
import numpy as np

try:
    # Preferred: package import when used as installed package
    from src.qins_lookup_tables import qins_to_binary, binary_to_qins
except Exception:
    # Fallback: load module directly from file for standalone execution
    import importlib.util
    import os

    qins_path = os.path.join(os.path.dirname(__file__), 'qins_lookup_tables.py')
    spec = importlib.util.spec_from_file_location('qins_lookup_tables', qins_path)
    qins_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qins_mod)
    qins_to_binary = qins_mod.qins_to_binary
    binary_to_qins = qins_mod.binary_to_qins


def float_to_fpins_levels(value: float, depth: int = 2) -> List[int]:
    """
    Encode a single float (magnitude) to FPINS levels (QINS values), depth=L -> L+1 levels.

    Args:
        value: real-valued scalar (can be negative)
        depth: number of extra levels (L). levels = depth + 1

    Returns:
        List of binary storage bytes (uint8) length = depth + 1
    """
    if value == 0.0:
        # Represent exact zero as "near-zero" (QINS 256 -> binary 0) across levels
        return [0] * (depth + 1)

    sign = -1 if value < 0 else 1
    mag = abs(value)

    # Target magnitude μ should be in reasonable range; clamp
    # We'll map typical neural weights to μ in [1e-6, 256]
    eps = 1e-12
    mag = max(mag, eps)

    # Desired product P such that μ = 256 / P ≈ mag  => P = 256 / mag
    P_target = 256.0 / mag

    levels_qins = []
    remaining = P_target
    levels = depth + 1

    for i in range(levels):
        # take the (levels-i)-th root to approximate even factorization
        root = remaining ** (1.0 / (levels - i))
        k = int(round(root))
        # clamp to [1, 256]
        k = max(1, min(256, k))
        levels_qins.append(k)
        # update remaining
        remaining = remaining / k if k != 0 else remaining

    # Final adjust: make sure product isn't zero and within range
    # Convert qins values into binary storage
    levels_bin = [qins_to_binary(int(k)) for k in levels_qins]
    return levels_bin


def fpins_levels_to_float(levels_bin: List[int]) -> float:
    """
    Decode FPINS binary levels (list of uint8) back to float magnitude.

    Args:
        levels_bin: list of binary storage bytes (0..255)
    Returns:
        Reconstructed float magnitude (positive)
    """
    if all(b == 0 for b in levels_bin):
        # Representing near-zero
        return 1.0  # μ = 1.0 when QINS value 256 -> conservative fallback

    # Convert binary to QINS values
    qins_vals = [binary_to_qins(int(b)) for b in levels_bin]
    P = 1
    for k in qins_vals:
        P *= int(k)
    # Reconstruct magnitude: μ = 256 / P
    mag = 256.0 / float(P)
    return mag


def encode_tensor_fp32_to_fpins(tensor: np.ndarray, depth: int = 2) -> np.ndarray:
    """
    Encode a NumPy FP32 tensor into FPINS binary levels.

    Returns array shape: tensor.shape + (levels, ) dtype=uint8
    """
    levels = depth + 1
    out_shape = tensor.shape + (levels,)
    out = np.zeros(out_shape, dtype=np.uint8)

    it = np.nditer(tensor, flags=['multi_index', 'refs_ok'])
    for x in it:
        idx = it.multi_index
        val = float(x)
        sign = -1 if val < 0 else 1
        levels_bin = float_to_fpins_levels(abs(val), depth=depth)
        for l in range(levels):
            out[idx + (l,)] = levels_bin[l]
    return out


def decode_tensor_fpins_to_fp32(fpins: np.ndarray) -> np.ndarray:
    """
    Decode FPINS array back to FP32 tensor.

    Input shape: tensor.shape + (levels,)
    Returns: tensor.shape dtype=float32
    """
    levels = fpins.shape[-1]
    out_shape = fpins.shape[:-1]
    out = np.zeros(out_shape, dtype=np.float32)

    it = np.nditer(out, flags=['multi_index', 'refs_ok'], op_flags=['readwrite'])
    for x in it:
        idx = it.multi_index
        levels_bin = [int(fpins[idx + (l,)]) for l in range(levels)]
        mag = fpins_levels_to_float(levels_bin)
        out[idx] = float(mag)
    return out


# Small self-test when run as script
if __name__ == '__main__':
    print('FPINS converter self-test (depth=2)')
    vals = np.array([0.5, 0.1, -0.25, 0.003, 1.0, 0.0001], dtype=np.float32)
    print('Original:', vals)

    enc = encode_tensor_fp32_to_fpins(vals, depth=2)
    print('Encoded (levels):')
    print(enc)

    dec = decode_tensor_fpins_to_fp32(enc)
    print('Decoded magnitudes:', dec)

    # Relative error
    rel_err = np.abs((np.abs(vals) - dec) / (np.abs(vals) + 1e-12))
    print('Relative errors:', rel_err)
