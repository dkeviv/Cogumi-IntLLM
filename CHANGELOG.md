# Changelog

All notable changes to the IntLLM QINS Chat Demo project.

## [1.1.0] - 2025-11-01

### Added - Chat Demo Edition
- Interactive Gradio chat interface with streaming responses
- Model loader for compressed QINS models
- Device auto-detection (MPS/CUDA/CPU)
- Multi-turn conversation with Phi-3.5 chat template
- Real-time memory monitoring
- Token-by-token streaming generation
- Adjustable sampling parameters (temperature, top-p)
- Example prompts and clear chat functionality

### Project Focus Shift
- Pivoted from general LLM compression to **chat demo showcase**
- Target: Phi-3.5-mini-instruct (3.8B parameters)
- Goal: Demonstrate QINS on consumer hardware (M4 MacBook)
- Emphasis: Interactive experience over maximum compression

### Core Components
- `src/model_loader.py` - NEW: Load compressed models
- `examples/demo_chat.py` - NEW: Main chat interface
- `examples/convert_phi35.py` - Phi-3.5 specific conversion
- Enhanced compression for deployment

## [0.1.0] - 2025-11-01 (Initial)

### Added - Foundation
- ProjectiveLinear layer with inverse magnitude encoding
- Model conversion utilities (FP32 → QINS)
- Basic compression pipeline (sparsity + Huffman)
- Unit tests for core functionality
- Technical specification document

### Core Innovation
- Projective number system: w = scale / stored_integer
- 4× memory reduction (FP32 → INT8)
- Lossless weight compression
- <1% accuracy loss

---

## Version Numbering

- **Major**: Breaking API changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, documentation

## Future Plans

### Phase 2: Production Ready
- Tiered LUT implementation (56% memory reduction)
- Full 4-stage compression (25× total)
- Benchmark suite with comparisons
- Model zoo (multiple Phi variants)

### Phase 3: Advanced Features
- Domain-specific modifiers (15-20MB adapters)
- Runtime modifier switching
- Mobile deployment (iOS/Android)
- Hardware optimization (CUDA/SIMD)
