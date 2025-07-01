#!/usr/bin/env python3
"""
Simple test script for multi-prompt GCG implementation.
"""

import torch
import transformers
from nanogcg import GCGConfig, run

def test_multi_prompt_gcg():
    """Test the multi-prompt GCG implementation with a simple example."""
    
    # Use a small model for testing
    model_name = "gpt2"
    print(f"Loading model: {model_name}")
    
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Define multiple prompts and targets for testing
    # Each element is (messages, target) where messages can be a string or list of dicts
    prompts_and_targets = [
        ([{"role": "user", "content": "Tell me how to"}], "bake a cake"),
        ([{"role": "user", "content": "Explain how to"}], "solve a math problem"),
        ([{"role": "user", "content": "Show me how to"}], "write a letter")
    ]
    
    print(f"Testing with {len(prompts_and_targets)} prompts:")
    for i, (messages, target) in enumerate(prompts_and_targets):
        print(f"  {i+1}. {messages} -> '{target}'")
    
    # Create configuration with minimal settings for testing
    config = GCGConfig(
        num_steps=5,  # Very few steps for quick testing
        search_width=8,  # Small search width
        optim_str_init="test test test",  # Short optimization string
        verbosity="INFO"
    )
    
    print("\nRunning multi-prompt GCG optimization...")
    
    try:
        result = run(
            model=model,
            tokenizer=tokenizer,
            prompts_and_targets=prompts_and_targets,
            config=config
        )
        
        print(f"\nOptimization completed successfully!")
        print(f"Best loss: {result.best_loss:.4f}")
        print(f"Best string: '{result.best_string}'")
        print(f"Total optimization steps: {len(result.losses)}")
        
        return True
        
    except Exception as e:
        print(f"\nError during optimization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multi_prompt_gcg()
    if success:
        print("\n✓ Multi-prompt GCG test passed!")
    else:
        print("\n✗ Multi-prompt GCG test failed!")