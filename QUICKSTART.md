# QINS Chat Demo - Quick Start Guide

This guide will help you set up and run the QINS interactive chat demo with Phi-3.5-mini.

## 🚀 Quick Start (5 minutes)

### 1. Install Dependencies

```bash
cd /Users/vivekdurairaj/Projects/Cogumi-IntLLM

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Run Chat Demo (Option A: Direct from HuggingFace)

**Fastest way to test** - downloads and converts on-the-fly:

```bash
python examples/demo_chat.py \
    --model microsoft/Phi-3.5-mini-instruct \
    --hub \
    --device mps
```

This will:
- Download Phi-3.5-mini (~7.6 GB FP32)
- Convert to QINS format (~1.9 GB)
- Launch Gradio interface at http://localhost:7860

**Note:** First run downloads the model (may take 5-10 minutes depending on connection).

### 3. Run Chat Demo (Option B: Pre-converted Model)

**For production use** - pre-convert and compress the model:

```bash
# Step 1: Convert and compress (one-time, ~10 minutes)
python examples/convert_phi35.py --output models/phi35-qins.compressed

# Step 2: Run chat with compressed model (<10 seconds to load)
python examples/demo_chat.py --model models/phi35-qins.compressed
```

## 📊 Expected Results

### Memory Usage

| Stage | Memory | Size |
|-------|--------|------|
| FP32 (original) | ~7.6 GB | - |
| QINS (converted) | ~1.9 GB | 4× reduction |
| Compressed | ~400 MB | 19× reduction |

### Performance (M4 MacBook)

- **Load time:** <10 seconds (compressed model)
- **First token:** <2 seconds
- **Throughput:** 5-8 tokens/second on CPU
- **Throughput:** 10-15 tokens/second on MPS

## 🎨 Using the Chat Interface

### Main Features

1. **Chat History** - View conversation with streaming responses
2. **Memory Monitor** - Real-time memory usage display
3. **Generation Settings** - Adjust temperature, top-p, max tokens
4. **Example Prompts** - Quick start with pre-made prompts

### Generation Settings

- **Temperature** (0.1 - 2.0)
  - Low (0.1-0.5): More focused, deterministic
  - Medium (0.7-1.0): Balanced creativity
  - High (1.5-2.0): More creative, diverse

- **Top-p** (0.1 - 1.0)
  - Low (0.1-0.5): Conservative word choices
  - Medium (0.7-0.9): Balanced selection
  - High (0.95-1.0): More variety

- **Max Tokens** (50 - 2048)
  - Controls maximum response length

### Try These Prompts

```
1. "Explain the QINS compression method in simple terms"
2. "Write a Python function to implement binary search"
3. "What are the benefits of inverse magnitude encoding?"
4. "Compare CPU and GPU inference for LLMs"
```

## 🔧 Advanced Usage

### Custom Model Path

```bash
python examples/demo_chat.py \
    --model /path/to/your/model.compressed \
    --device cpu \
    --port 7860
```

### Create Public Link

```bash
python examples/demo_chat.py \
    --model microsoft/Phi-3.5-mini-instruct \
    --hub \
    --share
```

This creates a public URL you can share for 72 hours.

### Different Devices

```bash
# Use CPU
python examples/demo_chat.py --model models/phi35-qins.compressed --device cpu

# Use MPS (Apple Silicon)
python examples/demo_chat.py --model models/phi35-qins.compressed --device mps

# Use CUDA (NVIDIA GPU)
python examples/demo_chat.py --model models/phi35-qins.compressed --device cuda

# Auto-detect (default)
python examples/demo_chat.py --model models/phi35-qins.compressed
```

## 🧪 Testing

### Run Unit Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_chat.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Quick Verification

```bash
# Test imports
python -c "from src.projective_layer import ProjectiveLinear; print('✓ Core layer')"
python -c "from src.model_loader import QINSModelLoader; print('✓ Model loader')"
python -c "from examples.demo_chat import QINSChatSystem; print('✓ Chat system')"

# Test conversion (without downloading model)
python -c "from src.converter import convert_model_to_projective; print('✓ Converter')"
```

## 📝 Python API Usage

### Load and Use Chat System

```python
from examples.demo_chat import QINSChatSystem

# Initialize
chat = QINSChatSystem(
    "microsoft/Phi-3.5-mini-instruct",
    device="mps",
    load_from_hub=True
)

# Single turn
history = []
response = list(chat.generate_streaming("Hello!", history))
print(response[-1])

# Multi-turn
history = [("Hello!", "Hi! How can I help?")]
response = list(chat.generate_streaming("What's the weather?", history))
print(response[-1])
```

### Convert Model Programmatically

```python
from src.model_loader import QINSModelLoader
from src.converter import convert_model_to_projective
from transformers import AutoModelForCausalLM

# Load FP32 model
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    torch_dtype=torch.float32
)

# Convert to QINS
qins_model = convert_model_to_projective(model, scale=256)

# Save compressed
from src.model_loader import save_compressed_model
from src.compression import ProjectiveCompressor

compressor = ProjectiveCompressor(phase=1)
save_compressed_model(qins_model, "my_model.compressed", compressor)

# Load later
loader = QINSModelLoader()
loaded_model, tokenizer = loader.load("my_model.compressed")
```

## 🐛 Troubleshooting

### Import Errors

```bash
# Error: "Import torch could not be resolved"
pip install torch>=2.0.0

# Error: "Import transformers could not be resolved"
pip install transformers>=4.36.0

# Error: "Import gradio could not be resolved"
pip install gradio>=4.0.0
```

### Memory Issues

**Problem:** Out of memory during model loading

**Solutions:**
1. Use compressed model instead of `--hub`
2. Close other applications
3. Use smaller batch size
4. Use CPU instead of MPS if MPS has issues

### MPS Device Issues

**Problem:** MPS errors on M4

**Solutions:**
```bash
# Force CPU
python examples/demo_chat.py --model MODEL --device cpu

# Update PyTorch
pip install --upgrade torch torchvision torchaudio
```

### Slow Generation

**Problem:** <1 token/second

**Possible causes:**
1. Using FP32 instead of QINS
2. Not using MPS on Apple Silicon
3. Large max_tokens setting
4. System resource constraints

**Solutions:**
```bash
# Verify device
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"

# Use MPS explicitly
python examples/demo_chat.py --model MODEL --device mps
```

### Gradio Interface Not Opening

**Problem:** Interface doesn't open in browser

**Solutions:**
1. Manually open: http://localhost:7860
2. Try different port: `--port 8080`
3. Check firewall settings
4. Use `--share` for public link

## 📚 Next Steps

1. **Experiment with Settings**
   - Try different temperature values
   - Test various prompts
   - Monitor memory usage

2. **Benchmark Performance**
   ```bash
   python examples/benchmark_memory.py  # If created
   ```

3. **Explore the Code**
   - Read `src/projective_layer.py` - Core QINS implementation
   - Read `src/converter.py` - Model conversion logic
   - Read `TECHNICAL_SPEC.md` - Deep dive into algorithms

4. **Customize**
   - Modify system prompt in `demo_chat.py`
   - Adjust default generation settings
   - Add custom Gradio components

## 🎯 Success Criteria

Your chat demo is working correctly if:

- ✅ Model loads in <10 seconds (compressed)
- ✅ Memory usage ~1.9 GB (QINS) or ~400 MB (compressed)
- ✅ Responses stream token-by-token
- ✅ Multi-turn conversations work
- ✅ Generation speed >3 tokens/second on M4
- ✅ Responses are coherent and on-topic

## 📞 Support

For issues or questions:
1. Check `TECHNICAL_SPEC.md` for implementation details
2. Review `GETTING_STARTED.md` for setup help
3. Run tests to verify installation: `pytest tests/`
4. Check GitHub issues (if repository is public)

---

**Enjoy chatting with QINS! 🚀**
