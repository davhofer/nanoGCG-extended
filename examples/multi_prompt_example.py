#!/usr/bin/env python3
"""
Multi-Prompt GCG Example

This example demonstrates how to use the multi-prompt variant of nanoGCG
to optimize adversarial strings across multiple different prompts simultaneously.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import nanogcg
from nanogcg import GCGConfig

# Model configuration
MODEL_NAME = "gpt2"  # Very small model for testing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer():
    """Load the model and tokenizer."""
    print(f"Loading model {MODEL_NAME} on {DEVICE}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def basic_multi_prompt_example():
    """Basic multi-prompt optimization example."""
    print("\n=== Basic Multi-Prompt Example ===")
    
    model, tokenizer = load_model_and_tokenizer()
    
    # Define multiple prompts with different targets
    messages_and_targets = [
        # Conversation format
        ([{"role": "user", "content": "Tell me how to"}], "bake a delicious cake"),
        ([{"role": "user", "content": "Explain how to"}], "solve quadratic equations"),
        ([{"role": "user", "content": "Show me how to"}], "write a professional email"),
    ]
    
    # Configure GCG
    config = GCGConfig(
        num_steps=5,            # Very small for testing
        search_width=16,        # Very small for testing
        topk=8,                # Very small for testing
        batch_size=4,          # Small batch for testing
        verbosity="INFO"
    )
    
    print(f"Optimizing across {len(messages_and_targets)} prompts...")
    
    # Run multi-prompt optimization
    result = nanogcg.run(model, tokenizer, messages_and_targets, config)
    
    print(f"Optimization completed!")
    print(f"Best adversarial string: '{result.best_string}'")
    print(f"Best loss: {result.best_loss:.4f}")
    
    return result

def custom_placement_example():
    """Example with custom adversarial string placement."""
    print("\n=== Custom Placement Example ===")
    
    model, tokenizer = load_model_and_tokenizer()
    
    # Use {optim_str} placeholder for custom placement
    messages_and_targets = [
        ([{"role": "user", "content": "Please {optim_str} help me understand"}], "calculus concepts"),
        ([{"role": "user", "content": "Can you {optim_str} explain to me"}], "machine learning basics"),
        ([{"role": "user", "content": "I need {optim_str} assistance with"}], "programming in Python"),
    ]
    
    config = GCGConfig(
        num_steps=3,
        search_width=8,
        topk=4,
        batch_size=2,
        verbosity="INFO"
    )
    
    print(f"Optimizing with custom placement across {len(messages_and_targets)} prompts...")
    
    result = nanogcg.run(model, tokenizer, messages_and_targets, config)
    
    print(f"Best adversarial string: '{result.best_string}'")
    print(f"Best loss: {result.best_loss:.4f}")
    
    return result

def probe_sampling_example():
    """Example using probe sampling for acceleration."""
    print("\n=== Probe Sampling Example ===")
    
    model, tokenizer = load_model_and_tokenizer()
    
    # Multiple prompts for probe sampling
    messages_and_targets = [
        ([{"role": "user", "content": "Generate"}], "creative writing content"),
        ([{"role": "user", "content": "Create"}], "educational material"),
        ([{"role": "user", "content": "Produce"}], "technical documentation"),
        ([{"role": "user", "content": "Develop"}], "marketing content"),
    ]
    
    # Load a smaller draft model for probe sampling
    try:
        draft_model = AutoModelForCausalLM.from_pretrained(
            "distilgpt2",  # Very small draft model
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None
        )
        
        draft_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        if draft_tokenizer.pad_token is None:
            draft_tokenizer.pad_token = draft_tokenizer.eos_token
        
        # Configure with probe sampling
        from nanogcg import ProbeSamplingConfig
        
        probe_config = ProbeSamplingConfig(
            draft_model=draft_model,
            draft_tokenizer=draft_tokenizer,
            r=4,  # Reduced for testing  
            sampling_factor=8  # Reduced for testing
        )
        
        config = GCGConfig(
            num_steps=3,
            search_width=8,
            topk=4,
            batch_size=2,
            probe_sampling_config=probe_config,
            verbosity="INFO"
        )
        
        print(f"Optimizing with probe sampling across {len(messages_and_targets)} prompts...")
        
        result = nanogcg.run(model, tokenizer, messages_and_targets, config)
        
        print(f"Best adversarial string: '{result.best_string}'")
        print(f"Best loss: {result.best_loss:.4f}")
        
        return result
        
    except Exception as e:
        print(f"Probe sampling example failed: {e}")
        print("Falling back to basic optimization...")
        
        config = GCGConfig(
            num_steps=3,
            search_width=8,
            topk=4,
            batch_size=2,
            verbosity="INFO"
        )
        
        result = nanogcg.run(model, tokenizer, messages_and_targets, config)
        print(f"Best adversarial string: '{result.best_string}'")
        return result

def mixed_format_example():
    """Example mixing string and conversation formats."""
    print("\n=== Mixed Format Example ===")
    
    model, tokenizer = load_model_and_tokenizer()
    
    # Mix string and conversation formats
    messages_and_targets = [
        # String format
        ("Complete this sentence:", "The sky is blue"),
        # Conversation format  
        ([{"role": "user", "content": "What color is"}], "the ocean"),
        # Multi-turn conversation
        ([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Tell me about"}
        ], "interesting facts"),
    ]
    
    config = GCGConfig(
        num_steps=3,
        search_width=8,
        topk=4,
        batch_size=2,
        verbosity="INFO"
    )
    
    print(f"Optimizing mixed formats across {len(messages_and_targets)} prompts...")
    
    result = nanogcg.run(model, tokenizer, messages_and_targets, config)
    
    print(f"Best adversarial string: '{result.best_string}'")
    print(f"Best loss: {result.best_loss:.4f}")
    
    return result

def main():
    """Run all multi-prompt examples."""
    print("🚀 Multi-Prompt GCG Examples")
    print("=" * 50)
    
    try:
        # Run basic example
        basic_result = basic_multi_prompt_example()
        
        # Run custom placement example
        custom_result = custom_placement_example()
        
        # Run probe sampling example
        probe_result = probe_sampling_example()
        
        # Run mixed format example
        mixed_result = mixed_format_example()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        print("\nSummary of results:")
        print(f"Basic: '{basic_result.best_string}' (loss: {basic_result.best_loss:.4f})")
        print(f"Custom: '{custom_result.best_string}' (loss: {custom_result.best_loss:.4f})")
        print(f"Probe: '{probe_result.best_string}' (loss: {probe_result.best_loss:.4f})")
        print(f"Mixed: '{mixed_result.best_string}' (loss: {mixed_result.best_loss:.4f})")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()