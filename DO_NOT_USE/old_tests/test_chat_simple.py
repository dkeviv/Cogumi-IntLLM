"""
Simple test script to verify chat generation works
Uses the already-loaded model from cache
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from examples.demo_chat import QINSChatSystem

print("Testing QINS chat system...")
print("This will load the model (may take 1-2 minutes)...")
print()

# Initialize (will load from cache)
chat = QINSChatSystem(
    "microsoft/Phi-3.5-mini-instruct",
    device="mps",
    load_from_hub=True
)

print("\n" + "="*60)
print("✅ Model loaded successfully!")
print("="*60)
print()

# Test generation
test_message = "Hello! Can you count to 5?"
history = []

print(f"User: {test_message}")
print("Assistant: ", end="", flush=True)

try:
    for response in chat.generate_streaming(test_message, history):
        print(f"\r{' ' * 80}\rAssistant: {response}", end="", flush=True)
    print()  # Final newline
    print("\n" + "="*60)
    print("✅ Generation test PASSED!")
    print("   The chat system is working correctly.")
    print("="*60)
except Exception as e:
    print(f"\n❌ Generation test FAILED: {e}")
    import traceback
    traceback.print_exc()
