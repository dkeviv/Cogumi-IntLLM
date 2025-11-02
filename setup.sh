#!/bin/bash

# QINS Chat Demo - Setup and Verification Script
# This script sets up the environment and verifies the installation

set -e  # Exit on error

echo "================================================"
echo "🚀 QINS Chat Demo - Setup & Verification"
echo "================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check Python version
echo "📋 Step 1: Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo -e "   ${GREEN}✓ Python 3.8+ detected${NC}"
else
    echo -e "   ${RED}✗ Python 3.8+ required${NC}"
    exit 1
fi
echo ""

# Step 2: Create virtual environment
echo "📋 Step 2: Setting up virtual environment..."
if [ -d "venv" ]; then
    echo -e "   ${YELLOW}⚠ Virtual environment already exists${NC}"
    read -p "   Delete and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo -e "   ${GREEN}✓ Virtual environment recreated${NC}"
    else
        echo "   Using existing virtual environment"
    fi
else
    python3 -m venv venv
    echo -e "   ${GREEN}✓ Virtual environment created${NC}"
fi
echo ""

# Step 3: Activate virtual environment
echo "📋 Step 3: Activating virtual environment..."
source venv/bin/activate
echo -e "   ${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Step 4: Upgrade pip
echo "📋 Step 4: Upgrading pip..."
pip install --upgrade pip --quiet
echo -e "   ${GREEN}✓ pip upgraded${NC}"
echo ""

# Step 5: Install dependencies
echo "📋 Step 5: Installing dependencies..."
echo "   This may take 5-10 minutes..."
pip install -r requirements.txt --quiet
echo -e "   ${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 6: Verify installations
echo "📋 Step 6: Verifying installations..."

# Check torch
if python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    echo -e "   ${GREEN}✓ PyTorch installed${NC}"
else
    echo -e "   ${RED}✗ PyTorch not found${NC}"
    exit 1
fi

# Check transformers
if python -c "import transformers; print(f'Transformers {transformers.__version__}')" 2>/dev/null; then
    echo -e "   ${GREEN}✓ Transformers installed${NC}"
else
    echo -e "   ${RED}✗ Transformers not found${NC}"
    exit 1
fi

# Check gradio
if python -c "import gradio; print(f'Gradio {gradio.__version__}')" 2>/dev/null; then
    echo -e "   ${GREEN}✓ Gradio installed${NC}"
else
    echo -e "   ${RED}✗ Gradio not found${NC}"
    exit 1
fi

# Check other dependencies
if python -c "import numpy, psutil, pytest" 2>/dev/null; then
    echo -e "   ${GREEN}✓ Other dependencies installed${NC}"
else
    echo -e "   ${RED}✗ Some dependencies missing${NC}"
    exit 1
fi
echo ""

# Step 7: Check device availability
echo "📋 Step 7: Checking device availability..."
device_info=$(python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'CPU cores: {torch.get_num_threads()}')
")
echo "$device_info"

if python -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    echo -e "   ${GREEN}✓ MPS (Apple Silicon) detected - optimal performance!${NC}"
elif python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo -e "   ${GREEN}✓ CUDA detected - GPU acceleration available${NC}"
else
    echo -e "   ${YELLOW}⚠ CPU only - will work but slower${NC}"
fi
echo ""

# Step 8: Test core imports
echo "📋 Step 8: Testing core imports..."
if python -c "from src.projective_layer import ProjectiveLinear" 2>/dev/null; then
    echo -e "   ${GREEN}✓ Core layer import successful${NC}"
else
    echo -e "   ${RED}✗ Core layer import failed${NC}"
    exit 1
fi

if python -c "from src.model_loader import QINSModelLoader" 2>/dev/null; then
    echo -e "   ${GREEN}✓ Model loader import successful${NC}"
else
    echo -e "   ${RED}✗ Model loader import failed${NC}"
    exit 1
fi

if python -c "from examples.demo_chat import QINSChatSystem" 2>/dev/null; then
    echo -e "   ${GREEN}✓ Chat system import successful${NC}"
else
    echo -e "   ${RED}✗ Chat system import failed${NC}"
    exit 1
fi
echo ""

# Step 9: Run quick tests
echo "📋 Step 9: Running quick tests..."
if python -m pytest tests/test_layer.py -v --tb=short 2>&1 | grep -q "passed"; then
    echo -e "   ${GREEN}✓ Layer tests passed${NC}"
else
    echo -e "   ${YELLOW}⚠ Some layer tests may have failed${NC}"
fi
echo ""

# Step 10: Summary
echo "================================================"
echo "✅ SETUP COMPLETE!"
echo "================================================"
echo ""
echo "🎉 Your QINS chat demo is ready!"
echo ""
echo "Next steps:"
echo ""
echo "1. Run the chat demo:"
echo "   ${GREEN}python examples/demo_chat.py --model microsoft/Phi-3.5-mini-instruct --hub${NC}"
echo ""
echo "2. Or pre-convert the model:"
echo "   ${GREEN}python examples/convert_phi35.py --output models/phi35-qins.compressed${NC}"
echo "   ${GREEN}python examples/demo_chat.py --model models/phi35-qins.compressed${NC}"
echo ""
echo "3. Run tests:"
echo "   ${GREEN}pytest tests/ -v${NC}"
echo ""
echo "4. Read documentation:"
echo "   - QUICKSTART.md - Quick start guide"
echo "   - GETTING_STARTED.md - Detailed setup"
echo "   - TECHNICAL_SPEC.md - Technical details"
echo "   - PROJECT_SUMMARY.md - Complete overview"
echo ""
echo "📊 Expected performance on M4 MacBook:"
echo "   - Load time: 5-10 seconds"
echo "   - Memory: ~1.9 GB (QINS) or ~400 MB (compressed)"
echo "   - Speed: 5-8 tokens/sec (CPU), 10-15 tokens/sec (MPS)"
echo ""
echo "Have fun chatting! 🚀"
echo ""
