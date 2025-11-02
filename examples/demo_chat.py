"""
FILE: demo_chat.py
PURPOSE: Interactive Gradio chat interface for QINS Phi-3.5
DEPENDENCIES: gradio, torch, transformers, psutil

CRITICAL CONCEPTS:
- Multi-turn conversation with history formatting
- Token-by-token streaming for real-time display
- Memory monitoring (show QINS advantage)
- M4-optimized inference (MPS device)
- No HuggingFace .generate() - custom loop

CHAT TEMPLATE (Phi-3.5):
<|system|>
You are a helpful AI assistant.<|end|>
<|user|>
User message here<|end|>
<|assistant|>
Assistant response here<|end|>

GENERATION ALGORITHM:
1. Format conversation with chat template
2. Tokenize to input_ids
3. Loop until EOS or max_tokens:
   - Forward pass through model
   - Sample next token (temperature + top-p)
   - Decode and yield token
   - Append to input_ids
4. Return full response
"""

import gradio as gr
import torch
import torch.nn.functional as F
import psutil
import os
from typing import Iterator, List, Tuple
import time
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader import QINSModelLoader


class QINSChatSystem:
    """
    Interactive chat system for QINS models.
    
    Features:
    - Multi-turn conversation management
    - Streaming token generation (ChatGPT-like)
    - Memory usage monitoring
    - Temperature and top-p sampling
    - M4/MPS optimization
    
    Usage:
        chat = QINSChatSystem("models/phi35-qins.compressed")
        for token in chat.generate_streaming("Hello!", []):
            print(token, end='', flush=True)
    """
    
    def __init__(
        self, 
        model_path: str,
        device: str = None,
        load_from_hub: bool = False
    ):
        """
        Initialize chat system.
        
        Args:
            model_path: Path to compressed model OR HuggingFace model ID
            device: "cpu", "mps", "cuda", or None (auto-detect)
            load_from_hub: If True, download from HuggingFace
        """
        print("=" * 60)
        print("🚀 Initializing QINS Chat System")
        print("=" * 60)
        
        start_time = time.time()
        mem_before = self._get_memory_gb()
        
        # Load model
        loader = QINSModelLoader(device=device)
        
        if load_from_hub:
            self.model, self.tokenizer = loader.load_from_pretrained(model_path)
        else:
            self.model, self.tokenizer = loader.load(
                model_path,
                "microsoft/Phi-3.5-mini-instruct"
            )
        
        self.device = loader.device
        
        # Load stats
        load_time = time.time() - start_time
        mem_after = self._get_memory_gb()
        self.model_memory = mem_after - mem_before
        
        print(f"✓ Loaded in {load_time:.1f}s")
        print(f"✓ Model memory: {self.model_memory:.2f} GB")
        print(f"✓ Device: {self.device}")
        print("=" * 60)
        
        # Generation settings (can be adjusted)
        self.max_new_tokens = 512
        self.temperature = 0.7
        self.top_p = 0.9
        self.system_prompt = "You are a helpful AI assistant."
    
    def _get_memory_gb(self) -> float:
        """Get current process memory in GB."""
        process = psutil.Process(os.getpid())
        mem_bytes = process.memory_info().rss
        return mem_bytes / (1024 ** 3)
    
    def format_chat_history(
        self, 
        message: str, 
        history: List[Tuple[str, str]]
    ) -> str:
        """
        Format conversation for Phi-3.5 chat template.
        
        Args:
            message: Current user message
            history: List of (user_msg, assistant_msg) tuples
        
        Returns:
            Formatted prompt string with special tokens
        
        Phi-3.5 Format:
        <|system|>
        {system_prompt}<|end|>
        <|user|>
        {user_message}<|end|>
        <|assistant|>
        {assistant_response}<|end|>
        ...
        <|user|>
        {current_message}<|end|>
        <|assistant|>
        
        Algorithm:
        1. Start with system prompt
        2. Add each turn from history
        3. Add current message
        4. End with assistant tag (model continues here)
        """
        prompt = f"<|system|>\n{self.system_prompt}<|end|>\n"
        
        for user_msg, assistant_msg in history:
            prompt += f"<|user|>\n{user_msg}<|end|>\n"
            prompt += f"<|assistant|>\n{assistant_msg}<|end|>\n"
        
        prompt += f"<|user|>\n{message}<|end|>\n"
        prompt += "<|assistant|>\n"
        
        return prompt
    
    def generate_streaming(
        self,
        message: str,
        history: List[Tuple[str, str]],
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None
    ) -> Iterator[str]:
        """
        Generate response with token streaming.
        
        Args:
            message: User's message
            history: Conversation history
            max_new_tokens: Override default max tokens
            temperature: Override default temperature
            top_p: Override default top-p
        
        Yields:
            Token strings for streaming display
        
        Algorithm:
        1. Format prompt with history
        2. Tokenize: input_ids = tokenizer(prompt)
        3. Move to device
        4. Generation loop:
           a. Forward pass: logits = model(input_ids).logits
           b. Get last position: logits = logits[:, -1, :]
           c. Apply temperature: logits = logits / temperature
           d. Top-p filtering (nucleus sampling)
           e. Sample: next_token = multinomial(softmax(logits))
           f. Check EOS: break if next_token == eos_token_id
           g. Decode: token_text = tokenizer.decode(next_token)
           h. Yield token_text
           i. Append: input_ids = cat([input_ids, next_token])
        5. Done when EOS or max_tokens reached
        
        CRITICAL: Use torch.no_grad() to save memory
        CRITICAL: Use torch.multinomial() not argmax (for diversity)
        """
        # Use defaults if not specified
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        
        # Format prompt
        prompt = self.format_chat_history(message, history)
        
        # Tokenize
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)
        
        # Generation loop
        accumulated_text = ""
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass (use_cache=False to avoid cache issues)
                outputs = self.model(input_ids, use_cache=False)
                logits = outputs.logits[:, -1, :]  # Shape: (1, vocab_size)
                
                # Temperature scaling
                logits = logits / temperature
                
                # Top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), 
                    dim=-1
                )
                
                # Remove tokens with cumulative prob > top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 0] = False  # Keep at least one
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, indices_to_remove] = float('-inf')
                
                # Sample from filtered distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for end of sequence
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Decode token
                token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                
                # Yield for streaming
                accumulated_text += token_text
                yield accumulated_text
                
                # Append for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=-1)
    
    def chat(
        self, 
        message: str, 
        history: List[Tuple[str, str]]
    ) -> Iterator[str]:
        """
        Main chat function called by Gradio.
        
        This is the entry point for the Gradio ChatInterface.
        
        Args:
            message: User's message
            history: Gradio chat history format
        
        Yields:
            Accumulated response text for streaming
        """
        yield from self.generate_streaming(message, history)
    
    def get_stats(self) -> dict:
        """
        Get current system statistics.
        
        Returns:
            Dictionary with memory, device, model info
        """
        return {
            "memory_gb": self._get_memory_gb(),
            "model_memory_gb": self.model_memory,
            "device": self.device,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }


def create_gradio_interface(chat_system: QINSChatSystem):
    """
    Create Gradio web interface using ChatInterface (simpler and more compatible).
    
    Args:
        chat_system: Initialized QINSChatSystem
    
    Returns:
        Gradio Interface
    """
    def chat_fn(message, history):
        """Chat function for Gradio ChatInterface."""
        # Convert history to our format
        formatted_history = [(h[0], h[1]) for h in history] if history else []
        
        # Generate response with streaming
        full_response = ""
        for partial in chat_system.generate_streaming(message, formatted_history):
            full_response = partial
        
        return full_response
    
    # Create simple ChatInterface
    demo = gr.ChatInterface(
        fn=chat_fn,
        title="🚀 QINS Chat - Phi-3.5-mini on CPU",
        description=f"""
        **Quantum Integer Numerical System** - 4× compression, full quality
        
        **Device:** {chat_system.device} | **Memory:** {chat_system.model_memory:.2f} GB | **Compression:** 4× vs FP32
        """,
        examples=[
            "Explain quantum computing in simple terms",
            "Write a haiku about artificial intelligence",
            "What is the QINS compression method?",
            "Compare Python and Rust for systems programming"
        ],
        theme=gr.themes.Soft()
    )
    
    return demo


def create_gradio_interface_old(chat_system: QINSChatSystem) -> gr.Blocks:
    """
    DEPRECATED: Original complex interface - has compatibility issues.
    Kept for reference only.
    """
    with gr.Blocks(
        title="QINS Chat Demo",
        theme=gr.themes.Soft()
    ) as demo:
        
        # Header
        gr.Markdown("""
        # 🚀 QINS Chat - Phi-3.5-mini on CPU
        
        **Quantum Integer Numerical System** - 4× compression, full quality
        
        ---
        
        **Model:** Phi-3.5-mini-instruct (3.8B parameters)  
        **Memory:** ~1.9 GB (vs ~7.6 GB FP32)  
        **Compression:** 4× through inverse magnitude encoding  
        **Device:** CPU/MPS (M4 optimized)  
        """)
        
        with gr.Row():
            # Left column: Chat interface
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=500
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your message",
                        placeholder="Type your message here...",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send", scale=1, variant="primary")
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat")
                    
                # Example prompts
                gr.Examples(
                    examples=[
                        "Explain quantum computing in simple terms",
                        "Write a haiku about artificial intelligence",
                        "What is the QINS compression method?",
                        "Compare Python and Rust for systems programming"
                    ],
                    inputs=msg,
                    label="💡 Example Prompts"
                )
            
            # Right column: Stats and controls
            with gr.Column(scale=1):
                gr.Markdown("### 📊 System Stats")
                
                stats = chat_system.get_stats()
                
                memory_display = gr.Textbox(
                    label="Current Memory",
                    value=f"{stats['memory_gb']:.2f} GB",
                    interactive=False
                )
                
                gr.Markdown(f"""
                **Model Memory:** {stats['model_memory_gb']:.2f} GB  
                **Device:** {stats['device']}  
                **FP32 equivalent:** ~7.6 GB  
                **Compression:** 4×  
                
                ---
                
                ### ⚙️ Generation Settings
                """)
                
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=stats['temperature'],
                    step=0.1,
                    label="Temperature",
                    info="Higher = more creative"
                )
                
                top_p_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=stats['top_p'],
                    step=0.05,
                    label="Top-p",
                    info="Nucleus sampling threshold"
                )
                
                max_tokens_slider = gr.Slider(
                    minimum=50,
                    maximum=2048,
                    value=stats['max_tokens'],
                    step=50,
                    label="Max Tokens",
                    info="Maximum response length"
                )
                
                def update_stats():
                    stats = chat_system.get_stats()
                    return f"{stats['memory_gb']:.2f} GB"
                
                refresh_btn = gr.Button("🔄 Refresh Stats")
                refresh_btn.click(
                    fn=update_stats,
                    outputs=memory_display
                )
        
        # Footer info
        gr.Markdown("""
        ---
        
        ### 💡 About QINS
        
        **Quantum Integer Numerical System** uses inverse magnitude encoding:
        - Stored value 1 → highest magnitude (near infinity)
        - Stored value 255 → lowest magnitude (near zero)
        - Formula: `w_effective = scale / stored_integer`
        
        This enables:
        - ✅ 4× memory reduction (INT8 vs FP32)
        - ✅ Faster CPU inference
        - ✅ <1% accuracy loss
        - ✅ Natural precision allocation (more bits for small values)
        
        **[Learn more about QINS](https://github.com/your-repo/qins)**
        """)
        
        # Chat interaction logic
        def respond(message, chat_history, temperature, top_p, max_tokens):
            """Handle chat interaction with streaming."""
            if not message:
                return chat_history
                
            # Update settings
            chat_system.temperature = temperature
            chat_system.top_p = top_p
            chat_system.max_new_tokens = max_tokens
            
            # Generate response
            bot_message = ""
            for partial_response in chat_system.generate_streaming(message, chat_history):
                bot_message = partial_response
                yield chat_history + [[message, bot_message]]
        
        # Wire up interactions
        msg.submit(
            fn=respond,
            inputs=[msg, chatbot, temperature_slider, top_p_slider, max_tokens_slider],
            outputs=chatbot
        ).then(
            fn=lambda: "",  # Clear input box
            outputs=msg
        )
        
        send_btn.click(
            fn=respond,
            inputs=[msg, chatbot, temperature_slider, top_p_slider, max_tokens_slider],
            outputs=chatbot
        ).then(
            fn=lambda: "",
            outputs=msg
        )
        
        clear_btn.click(
            fn=lambda: None,
            outputs=chatbot
        )
    
    return demo


def main():
    """
    Launch QINS chat demo.
    
    Usage:
        python examples/demo_chat.py [--model MODEL_PATH] [--device DEVICE] [--hub]
    
    Arguments:
        --model: Path to compressed model or HuggingFace model ID
        --device: cpu/mps/cuda (auto-detect if not specified)
        --hub: Load from HuggingFace instead of compressed file
    
    Example:
        # From compressed file (recommended)
        python examples/demo_chat.py --model models/phi35-qins.compressed
        
        # From HuggingFace (downloads FP32 then converts)
        python examples/demo_chat.py --model microsoft/Phi-3.5-mini-instruct --hub
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="QINS Chat Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="models/phi35-qins.compressed",
        help="Path to compressed model or HuggingFace ID"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu/mps/cuda (auto if not specified)"
    )
    parser.add_argument(
        "--hub",
        action="store_true",
        help="Load from HuggingFace Hub"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for Gradio interface"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public link"
    )
    
    args = parser.parse_args()
    
    # Initialize chat system
    chat_system = QINSChatSystem(
        model_path=args.model,
        device=args.device,
        load_from_hub=args.hub
    )
    
    # Create and launch interface
    demo = create_gradio_interface(chat_system)
    
    print("\n" + "=" * 60)
    print("🌐 Launching Gradio interface...")
    print("=" * 60)
    
    # Try to launch with share=True to avoid localhost issues
    try:
        demo.launch(
            server_port=args.port,
            share=True,  # Always use share link to avoid localhost issues
            inbrowser=True
        )
    except Exception as e:
        print(f"Error launching interface: {e}")
        print("\nTrying alternative launch method...")
        demo.launch(
            share=True,
            inbrowser=False
        )


if __name__ == "__main__":
    main()
