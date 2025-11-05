# Standalone runner for FPINS converter test (avoids package init side-effects)
import importlib.util
import sys
from pathlib import Path

# Load qins_lookup_tables as a module without importing package
spec = importlib.util.spec_from_file_location('qins_lookup_tables', str(Path(__file__).parent.parent / 'src' / 'qins_lookup_tables.py'))
qins = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qins)

# Load fpins_converter similarly
spec2 = importlib.util.spec_from_file_location('fpins_converter', str(Path(__file__).parent.parent / 'src' / 'fpins_converter.py'))
fpins = importlib.util.module_from_spec(spec2)
# Provide qins functions into fpins module globals to avoid import errors
fpins.__dict__['qins_to_binary'] = qins.qins_to_binary
fpins.__dict__['binary_to_qins'] = qins.binary_to_qins
spec2.loader.exec_module(fpins)

# Now run self-test logic from fpins
if __name__ == '__main__':
    import numpy as np
    vals = np.array([0.5, 0.1, -0.25, 0.003, 1.0, 0.0001], dtype=np.float32)
    print('Original:', vals)
    enc = fpins.encode_tensor_fp32_to_fpins(vals, depth=2)
    print('Encoded:\n', enc)
    dec = fpins.decode_tensor_fpins_to_fp32(enc)
    print('Decoded:', dec)
    rel_err = np.abs((np.abs(vals) - dec) / (np.abs(vals) + 1e-12))
    print('Relative errors:', rel_err)
