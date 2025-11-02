# Examples Directory

This directory contains example scripts demonstrating the QINS compression and chat system.

## 📁 Files

### `demo_chat.py` ⭐ (Main Demo)
Interactive Gradio chat interface for QINS-compressed Phi-3.5-mini.

**Features:**
- Real-time token streaming (ChatGPT-like display)
- Multi-turn conversation support
- Adjustable temperature and top-p sampling
- Memory usage monitoring
- Example prompts
- Works on CPU/MPS/CUDA

**Usage:**
```bash
# Option 1: Load from HuggingFace (downloads automatically)
python examples/demo_chat.py \
    --model microsoft/Phi-3.5-mini-instruct \
    --hub \
    --device mps

# Option 2: Load pre-compressed model
python examples/demo_chat.py \
    --model models/phi35-qins.compressed

# Option 3: Custom settings
python examples/demo_chat.py \
    --model microsoft/Phi-3.5-mini-instruct \
    --hub \
    --port 8080 \
    --share
```

**Arguments:**
- `--model`: Path to compressed model or HuggingFace model ID
- `--device`: cpu/mps/cuda (auto-detect if not specified)
- `--hub`: Load from HuggingFace Hub instead of local compressed file
- `--port`: Port for Gradio interface (default: 7860)
- `--share`: Create public Gradio link

**Expected Performance:**
- Load time: <10 seconds (compressed), 20-30 seconds (HuggingFace)
- Memory: ~1.9 GB (QINS uncompressed) or ~400 MB (compressed)
- Speed: 5-8 tokens/sec (CPU), 10-15 tokens/sec (MPS)

### `convert_phi35.py` (Model Conversion)
Convert Phi-3.5-mini from FP32 to compressed QINS format.

**Features:**
- Downloads Phi-3.5-mini from HuggingFace
- Converts to QINS (INT8 projective encoding)
- Validates conversion accuracy
- Compresses with sparsity + Huffman
- Saves compressed model

**Usage:**
```bash
# Standard conversion with compression
python examples/convert_phi35.py --output models/phi35-qins.compressed

# Skip compression (save QINS only)
python examples/convert_phi35.py \
    --output models/phi35-qins.pt \
    --skip-compression

# Different model
python examples/convert_phi35.py \
    --model-name microsoft/Phi-3-mini-instruct \
    --output models/phi3-qins.compressed
```

**Arguments:**
- `--output`: Where to save compressed model
- `--skip-compression`: Save QINS format without compression
- `--model-name`: HuggingFace model ID (default: Phi-3.5-mini-instruct)

**Process:**
1. Downloads model (~7.6 GB FP32)
2. Converts all Linear layers to ProjectiveLinear
3. Validates conversion (measures error)
4. Compresses with pipeline (sparsity + Huffman)
5. Saves to output path

**Expected Time:**
- Download: 5-10 minutes (depends on connection)
- Conversion: 2-3 minutes
- Compression: 1-2 minutes
- Total: ~10-15 minutes

**Output:**
- Compressed file: ~400 MB
- Compression ratio: ~19× vs original FP32
- Accuracy: <1% error vs FP32

## 🚀 Quick Start

### 1. Fastest Way (Load from HuggingFace)
```bash
python examples/demo_chat.py --hub --model microsoft/Phi-3.5-mini-instruct
```

### 2. Production Way (Pre-convert)
```bash
# One-time conversion
python examples/convert_phi35.py --output models/phi35-qins.compressed

# Fast loading afterwards
python examples/demo_chat.py --model models/phi35-qins.compressed
```

## 📊 Performance Comparison

| Method | First Load | Memory | Speed |
|--------|------------|--------|-------|
| FP32 (original) | 20-30s | 7.6 GB | 3-5 tok/s |
| QINS (uncompressed) | 15-20s | 1.9 GB | 5-8 tok/s |
| QINS (compressed) | 5-10s | 400 MB | 5-8 tok/s |

## 🎯 Example Workflows

### Workflow 1: Quick Test
```bash
# Test the system without pre-converting
python examples/demo_chat.py \
    --hub \
    --model microsoft/Phi-3.5-mini-instruct \
    --device mps
```

### Workflow 2: Production Deployment
```bash
# Step 1: Convert once
python examples/convert_phi35.py --output models/phi35-qins.compressed

# Step 2: Use repeatedly (fast load)
python examples/demo_chat.py --model models/phi35-qins.compressed
```

### Workflow 3: Benchmarking
```bash
# Convert and save both formats
python examples/convert_phi35.py \
    --output models/phi35-qins.compressed

python examples/convert_phi35.py \
    --output models/phi35-qins-uncompressed.pt \
    --skip-compression

# Compare performance
time python examples/demo_chat.py --model models/phi35-qins-uncompressed.pt
time python examples/demo_chat.py --model models/phi35-qins.compressed
```

## 🔧 Python API Usage

### Using Chat System Programmatically

```python
from examples.demo_chat import QINSChatSystem

# Initialize
chat = QINSChatSystem(
    "microsoft/Phi-3.5-mini-instruct",
    device="mps",
    load_from_hub=True
)

# Single turn conversation
history = []
for response in chat.generate_streaming("Hello, how are you?", history):
    print(response, end='\r')
print()  # Final newline

# Multi-turn conversation
history = [("Hello!", "Hi! How can I help you today?")]
for response in chat.generate_streaming("Tell me about quantum computing", history):
    print(response, end='\r')
print()

# Adjust settings
chat.temperature = 0.9
chat.top_p = 0.95
chat.max_new_tokens = 1024

# Generate with custom settings
for response in chat.generate_streaming("Write a poem", []):
    print(response, end='\r')
```

### Converting Models Programmatically

```python
import torch
from transformers import AutoModelForCausalLM
from src.converter import convert_model_to_projective
from src.compression import ProjectiveCompressor
from src.model_loader import save_compressed_model

# Load FP32 model
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    torch_dtype=torch.float32
)

# Convert to QINS
qins_model = convert_model_to_projective(model, scale=256, verbose=True)

# Compress and save
compressor = ProjectiveCompressor(phase=1)
save_compressed_model(qins_model, "my_model.compressed", compressor)

# Load later
from src.model_loader import QINSModelLoader
loader = QINSModelLoader()
loaded_model, tokenizer = loader.load("my_model.compressed")
```

## 📚 Additional Resources

- **Quick Start:** See [../QUICKSTART.md](../QUICKSTART.md)
- **Setup Guide:** See [../GETTING_STARTED.md](../GETTING_STARTED.md)
- **Technical Details:** See [../TECHNICAL_SPEC.md](../TECHNICAL_SPEC.md)
- **Project Overview:** See [../PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md)

## 🐛 Troubleshooting

### "Model not found" error
- Ensure model path is correct
- For HuggingFace models, use `--hub` flag
- Check internet connection for downloads

### "Device not available" error
- Use `--device cpu` to force CPU
- Check MPS availability: `python -c "import torch; print(torch.backends.mps.is_available())"`

### Slow performance
- Use MPS on M4: `--device mps`
- Use compressed model for faster loading
- Lower `max_new_tokens` setting

### Memory issues
- Use compressed model (~400 MB vs 1.9 GB)
- Close other applications
- Use CPU instead of MPS if issues persist

---

**Happy chatting! 🚀**
