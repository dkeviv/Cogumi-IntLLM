# Getting Started with IntLLM QINS Chat Demo

Quick start guide for setting up and running the interactive chat interface.

## 🎯 What You Have

A complete implementation of:
- ✅ Core QINS layer (inverse magnitude encoding)
- ✅ Model converter (FP32 → QINS)
- ✅ Compression pipeline (sparsity + Huffman)
- ✅ Model loader (with device auto-detection)
- ✅ Project structure and documentation

## 🚧 What's Next

The main chat interface (`examples/demo_chat.py`) and conversion script need to be created. These are the final pieces to make the demo fully functional.

## 📋 Step-by-Step Setup

### 1. Create Virtual Environment

```bash
cd /Users/vivekdurairaj/Projects/Cogumi-IntLLM

# Create venv
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# This will install:
# - torch (with MPS support for M4)
# - transformers (HuggingFace models)
# - gradio (web interface)
# - numpy, psutil, tqdm
# - pytest for testing
```

**Note on PyTorch**: The default installation includes MPS support for Apple Silicon. If you encounter issues:

```bash
# For CPU-only (smaller, faster install)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Verify Installation

```bash
# Quick test
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
python -c "import gradio; print(f'Gradio {gradio.__version__}')"
```

Expected output:
```
PyTorch 2.x.x
MPS available: True  (on M4 Mac)
Transformers 4.36.x
Gradio 4.x.x
```

## 🔨 Next Implementation Steps

### Step 1: Create the Chat Demo (Main Feature)

Create `examples/demo_chat.py` following the instructions in `.github/copilot-instructions.md`.

**Key components to implement:**

1. **QINSChatSystem class**:
   ```python
   - __init__(): Load model, initialize settings
   - _get_memory_gb(): Monitor memory usage
   - format_chat_history(): Phi-3.5 chat template
   - generate_streaming(): Token-by-token generation
   - chat(): Entry point for Gradio
   - get_stats(): System statistics
   ```

2. **create_gradio_interface() function**:
   ```python
   - Layout with chat + stats columns
   - Streaming chatbot interface
   - Temperature/top-p sliders
   - Example prompts
   - Memory display
   ```

3. **main() function**:
   ```python
   - Argument parsing (--model, --device, --hub)
   - Initialize QINSChatSystem
   - Launch Gradio interface
   ```

**Reference**: See `.github/copilot-instructions.md` lines 400-950 for complete template with TODOs.

### Step 2: Create Conversion Script

Create `examples/convert_phi35.py` to convert Phi-3.5-mini to QINS format.

**Key steps**:
1. Download Phi-3.5-mini from HuggingFace (~7.6 GB)
2. Convert using `convert_model_to_projective()`
3. Compress using `ProjectiveCompressor()`
4. Save using `save_compressed_model()`

**Reference**: See `.github/copilot-instructions.md` lines 950-1050.

### Step 3: Create Tests

Create test files in `tests/` directory:
- `tests/test_chat.py` - Chat system tests
- `tests/test_generation.py` - Generation quality tests

## 🧪 Testing What You Have

Even before creating the chat demo, you can test the core components:

### Test the Projective Layer

```bash
# Create a simple test script
cat > test_basics.py << 'EOF'
import torch
import torch.nn as nn
from src.projective_layer import ProjectiveLinear

# Create a standard Linear layer
linear = nn.Linear(64, 32)

# Convert to ProjectiveLinear
proj = ProjectiveLinear(64, 32)
proj.from_linear(linear)

# Test forward pass
x = torch.randn(10, 64)

with torch.no_grad():
    y_linear = linear(x)
    y_proj = proj(x)

# Compute error
error = torch.abs(y_linear - y_proj).mean()
print(f"Conversion error: {error:.6f}")
print(f"✓ ProjectiveLinear works!" if error < 0.1 else "✗ Error too high")
EOF

python test_basics.py
```

### Test Model Conversion

```python
# test_conversion_simple.py
import torch
import torch.nn as nn
from src.converter import convert_model_to_projective, get_model_statistics

# Create simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# Convert
model = SimpleModel()
stats_before = get_model_statistics(model)
print(f"Before: {stats_before['memory_fp32_gb']:.4f} GB")

model = convert_model_to_projective(model, verbose=True)
stats_after = get_model_statistics(model)
print(f"After: {stats_after['memory_int8_gb']:.4f} GB")
print(f"Compression: {stats_after['compression_ratio']:.2f}×")
```

### Test Compression

```python
# test_compression_simple.py
import numpy as np
from src.compression import ProjectiveCompressor

# Create sample weights
weights = {
    'layer1': np.random.randint(1, 256, (100, 50), dtype=np.uint8),
    'layer2': np.random.randint(1, 256, (50, 25), dtype=np.uint8)
}

# Compress
compressor = ProjectiveCompressor(phase=1)
compressed = compressor.compress(weights)

# Decompress
decompressed = compressor.decompress(compressed)

# Verify lossless
for name in weights:
    assert np.array_equal(weights[name], decompressed[name])
    print(f"✓ {name}: lossless verified")

# Show compression ratio
original_size = sum(w.nbytes for w in weights.values())
ratio = original_size / len(compressed)
print(f"\nCompression: {ratio:.2f}×")
```

## 📖 Documentation Reference

- **[README.md](README.md)** - Project overview and features
- **[TECHNICAL_SPEC.md](TECHNICAL_SPEC.md)** - Implementation details and algorithms
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - Complete implementation guide
- **[CHANGELOG.md](CHANGELOG.md)** - Version history

## 🎓 Understanding the Code

### Projective Number System

The key innovation is **inverse magnitude encoding**:

```python
# Traditional: stored_value ∝ weight
w = stored_value / scale  # Linear relationship

# QINS: stored_value ∝ 1/weight
w = scale / stored_value  # Inverse relationship!
```

**Why this works**:
- Small weights get high stored values (more precision bits)
- Large weights get low stored values (less precision needed)
- Natural allocation of quantization levels
- No special handling for zeros (min stored = 1)

### Forward Pass

```python
# Pre-computed once in __init__
lut[i] = scale / i  # For i in [1, 255]

# Every forward pass (fast!)
effective_weights = lut[stored]  # O(1) lookup
weights = sign.float() * effective_weights
output = F.linear(input, weights, bias)
```

**Performance**: LUT is 1 KB, fits in L1 cache → very fast lookups!

### Compression Pipeline

```
FP32 weights (7.6 GB)
     ↓ Conversion
QINS weights (1.9 GB) [4× reduction]
     ↓ Sparsity encoding (remove 40-60%)
Dense representation (~1 GB)
     ↓ Huffman coding (exploit distribution)
Compressed file (~400 MB) [19× total]
```

## 🚀 Running the Demo (Once Implemented)

### Quick Test (No Pre-compression)

```bash
python examples/demo_chat.py \
    --hub \
    --model microsoft/Phi-3.5-mini-instruct \
    --device mps
```

**First run**: Downloads 7.6 GB, converts on-the-fly (5-10 min)  
**Subsequent runs**: Reuses cached model

### Production Use (With Pre-compression)

```bash
# Step 1: Convert once (saves to disk)
python examples/convert_phi35.py \
    --output models/phi35-qins.compressed

# Step 2: Run chat (loads in <10 seconds)
python examples/demo_chat.py \
    --model models/phi35-qins.compressed \
    --device mps
```

### Open Browser

Navigate to: `http://localhost:7860`

You'll see:
- Chat interface with streaming responses
- Real-time memory monitoring
- Adjustable temperature/top-p
- Example prompts

## 🐛 Troubleshooting

### MPS Not Available

```python
import torch
print(torch.backends.mps.is_available())
```

If `False`, use `--device cpu` instead

### Out of Memory

If conversion fails with OOM:
- Close other applications
- Use `--device cpu` (uses system RAM)
- Consider smaller model (Phi-2)

### Model Download Issues

```bash
# Set HuggingFace cache
export HF_HOME=/path/to/large/disk

# Or use HF_HUB_CACHE
export HF_HUB_CACHE=/path/to/large/disk
```

### Import Errors

```bash
# Reinstall dependencies
pip uninstall torch transformers gradio
pip install -r requirements.txt
```

## 📊 Expected Performance

On M4 MacBook (24GB RAM):

| Metric | Value |
|--------|-------|
| Load time (compressed) | 5-10 seconds |
| Memory usage | 1.9 GB (model) + 0.5 GB (runtime) |
| First token latency | <2 seconds |
| Generation speed | 5-8 tokens/sec |
| Max conversation length | 512 tokens (configurable) |

## 💡 Tips for Development

1. **Start small**: Test with toy models before Phi-3.5
2. **Use verbose mode**: `convert_model_to_projective(model, verbose=True)`
3. **Monitor memory**: Use `psutil` to track actual usage
4. **Cache aggressively**: Save converted models to avoid recomputation
5. **Test incrementally**: Implement one function at a time, test before moving on

## 📚 Next Reading

1. Read `TECHNICAL_SPEC.md` for algorithm details
2. Read `.github/copilot-instructions.md` for implementation guide
3. Check HuggingFace Phi-3.5 model card for chat template details
4. Review Gradio documentation for interface customization

---

**Ready to implement?** Start with `examples/demo_chat.py` - it's the showcase feature!
