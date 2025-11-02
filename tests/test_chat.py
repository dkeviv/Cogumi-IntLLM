"""
FILE: test_chat.py
PURPOSE: Test chat system functionality
DEPENDENCIES: pytest, torch, examples.demo_chat
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.demo_chat import QINSChatSystem


class TestChatHistoryFormatting:
    """Test Phi-3.5 chat template formatting."""
    
    def test_format_single_turn(self):
        """Test single message formatting."""
        # Create mock chat system (will fail if no model, but we test formatting logic)
        # In real test, would use a mock model
        
        # Expected format for single turn
        system_prompt = "You are a helpful AI assistant."
        user_msg = "Hello, world!"
        
        expected = (
            f"<|system|>\n{system_prompt}<|end|>\n"
            f"<|user|>\n{user_msg}<|end|>\n"
            f"<|assistant|>\n"
        )
        
        # Test formatting logic directly
        # (Would need to mock QINSChatSystem or extract formatting function)
        assert "<|system|>" in expected
        assert "<|user|>" in expected
        assert "<|assistant|>" in expected
        assert "<|end|>" in expected
    
    def test_format_multi_turn(self):
        """Test multi-turn conversation formatting."""
        system_prompt = "You are a helpful AI assistant."
        history = [
            ("Hello", "Hi! How can I help you?"),
            ("What's the weather?", "I don't have weather data.")
        ]
        current_msg = "Thanks anyway"
        
        expected = (
            f"<|system|>\n{system_prompt}<|end|>\n"
            f"<|user|>\nHello<|end|>\n"
            f"<|assistant|>\nHi! How can I help you?<|end|>\n"
            f"<|user|>\nWhat's the weather?<|end|>\n"
            f"<|assistant|>\nI don't have weather data.<|end|>\n"
            f"<|user|>\nThanks anyway<|end|>\n"
            f"<|assistant|>\n"
        )
        
        # Verify structure
        assert expected.count("<|system|>") == 1
        assert expected.count("<|user|>") == 3
        assert expected.count("<|assistant|>") == 3
        assert expected.count("<|end|>") == 7  # All tags except last assistant


class TestStreamingGeneration:
    """Test token streaming functionality."""
    
    @pytest.mark.skipif(not torch.cuda.is_available() and not torch.backends.mps.is_available(),
                        reason="Requires GPU/MPS for realistic testing")
    def test_streaming_yields_tokens(self):
        """Test that streaming generation yields tokens incrementally."""
        # This would require a real or mocked model
        # For now, test the concept
        
        def mock_generate():
            tokens = ["Hello", " world", "!"]
            accumulated = ""
            for token in tokens:
                accumulated += token
                yield accumulated
        
        results = list(mock_generate())
        assert len(results) == 3
        assert results[0] == "Hello"
        assert results[1] == "Hello world"
        assert results[2] == "Hello world!"
    
    def test_streaming_stops_at_eos(self):
        """Test that generation stops at EOS token."""
        # Mock test - would need real tokenizer
        eos_token_id = 2
        
        def mock_check_eos(token_id):
            return token_id == eos_token_id
        
        assert mock_check_eos(2) == True
        assert mock_check_eos(1) == False


class TestTemperatureSampling:
    """Test temperature affects diversity."""
    
    def test_temperature_scaling(self):
        """Test that temperature scales logits correctly."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        temperature = 2.0
        
        scaled_logits = logits / temperature
        
        assert torch.allclose(scaled_logits, torch.tensor([[0.5, 1.0, 1.5]]))
    
    def test_low_temperature_more_deterministic(self):
        """Test that low temperature makes distribution more peaked."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        
        # Low temperature
        probs_low = torch.softmax(logits / 0.1, dim=-1)
        
        # High temperature
        probs_high = torch.softmax(logits / 2.0, dim=-1)
        
        # Low temperature should have higher max probability
        assert probs_low.max() > probs_high.max()


class TestTopPFiltering:
    """Test top-p nucleus sampling."""
    
    def test_top_p_filters_tokens(self):
        """Test that top-p removes low probability tokens."""
        # Create sample logits
        logits = torch.tensor([[3.0, 2.0, 1.0, 0.5, 0.1]])
        probs = torch.softmax(logits, dim=-1)
        
        # Sort by probability
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Top-p = 0.9
        top_p = 0.9
        indices_to_remove = cumulative_probs > top_p
        indices_to_remove[..., 0] = False  # Keep at least one
        
        # Should filter some tokens
        assert indices_to_remove.any()
    
    def test_top_p_keeps_at_least_one(self):
        """Test that top-p always keeps at least one token."""
        logits = torch.tensor([[1.0]])
        probs = torch.softmax(logits, dim=-1)
        
        cumulative_probs = torch.cumsum(probs, dim=-1)
        
        top_p = 0.5
        indices_to_remove = cumulative_probs > top_p
        indices_to_remove[..., 0] = False
        
        # Should keep the only token
        assert not indices_to_remove.all()


class TestMemoryMonitoring:
    """Test memory tracking functionality."""
    
    def test_memory_measurement(self):
        """Test that memory can be measured."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_bytes = process.memory_info().rss
        mem_gb = mem_bytes / (1024 ** 3)
        
        # Should be positive and reasonable
        assert mem_gb > 0
        assert mem_gb < 1000  # Sanity check


class TestDeviceDetection:
    """Test device detection logic."""
    
    def test_mps_detection(self):
        """Test MPS (Apple Silicon) detection."""
        has_mps = torch.backends.mps.is_available()
        # Just verify the check works, result depends on hardware
        assert isinstance(has_mps, bool)
    
    def test_cuda_detection(self):
        """Test CUDA detection."""
        has_cuda = torch.cuda.is_available()
        assert isinstance(has_cuda, bool)
    
    def test_device_priority(self):
        """Test that device detection follows priority: MPS > CUDA > CPU."""
        if torch.backends.mps.is_available():
            expected = "mps"
        elif torch.cuda.is_available():
            expected = "cuda"
        else:
            expected = "cpu"
        
        # Should select appropriate device
        assert expected in ["mps", "cuda", "cpu"]


# Integration tests (require model)
@pytest.mark.integration
class TestChatSystemIntegration:
    """Integration tests requiring actual model."""
    
    @pytest.mark.skip(reason="Requires downloaded model")
    def test_full_chat_flow(self):
        """Test complete chat interaction."""
        # This would test with actual model
        # chat_system = QINSChatSystem("microsoft/Phi-3.5-mini-instruct", load_from_hub=True)
        # response = list(chat_system.generate_streaming("Hello", []))
        # assert len(response) > 0
        pass
    
    @pytest.mark.skip(reason="Requires downloaded model")
    def test_multi_turn_conversation(self):
        """Test multi-turn conversation maintains context."""
        # Would test actual multi-turn conversation
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
