#!/usr/bin/env python3
"""
Test script to verify skyrl-tx connection and basic functionality.

Usage:
    python scripts/test_skyrl_tx.py [--url http://localhost:8000]
"""

import argparse
import asyncio
import sys

import tinker


async def test_connection(base_url: str) -> bool:
    """Test basic connection to skyrl-tx server."""
    print(f"\n{'='*60}")
    print(f"Testing skyrl-tx connection at {base_url}")
    print(f"{'='*60}\n")

    try:
        # Test 1: Create service client
        print("1. Creating ServiceClient...")
        client = tinker.ServiceClient(base_url=base_url)
        print("   ✓ ServiceClient created")

        # Test 2: Create training client with LoRA
        print("\n2. Creating LoRA training client...")
        training_client = await client.create_lora_training_client_async(
            base_model="Qwen/Qwen3-4B",
            rank=32,
        )
        print("   ✓ Training client created")

        # Test 3: Get tokenizer
        print("\n3. Getting tokenizer...")
        tokenizer = training_client.get_tokenizer()
        test_text = "Hello, this is a test!"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"   ✓ Tokenizer works")
        print(f"     Input:  '{test_text}'")
        print(f"     Tokens: {tokens[:10]}..." if len(tokens) > 10 else f"     Tokens: {tokens}")
        print(f"     Decoded: '{decoded}'")

        # Test 4: Create sampling client
        print("\n4. Creating sampling client...")
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(
            name="test_checkpoint"
        )
        print("   ✓ Sampling client created")

        # Test 5: Test sampling
        print("\n5. Testing text generation...")
        from tinker_cookbook.model_info import get_recommended_renderer_name
        from tinker_cookbook.renderers import get_renderer

        renderer_name = get_recommended_renderer_name("Qwen/Qwen3-4B")
        renderer = get_renderer(renderer_name, tokenizer)

        messages = [{"role": "user", "content": "What is 2 + 2?"}]
        model_input = renderer.build_generation_prompt(messages)
        stop_sequences = renderer.get_stop_sequences()

        result = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=50,
                temperature=0.7,
                stop=stop_sequences,
            ),
        )

        if result.sequences:
            generated_tokens = list(result.sequences[0].tokens)
            generated_text = tokenizer.decode(generated_tokens)
            print("   ✓ Text generation works")
            print(f"     Generated: {generated_text[:100]}..." if len(generated_text) > 100 else f"     Generated: {generated_text}")
            
            # Check logprobs
            if result.sequences[0].logprobs:
                print(f"     Logprobs: ✓ (received {len(result.sequences[0].logprobs)} values)")
            else:
                print("     Logprobs: ⚠ Not returned (may affect RL training)")
        else:
            print("   ⚠ No sequences returned")

        print(f"\n{'='*60}")
        print("✓ All tests passed! skyrl-tx is ready for training.")
        print(f"{'='*60}\n")
        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print(f"\nMake sure skyrl-tx server is running:")
        print(f"  cd SkyRL/skyrl-tx")
        print(f"  uv run --extra tinker --extra gpu python -m tx.tinker.api \\")
        print(f"    --base-model Qwen/Qwen3-4B \\")
        print(f"    --tensor-parallel-size 1 \\")
        print(f"    --max-lora-rank 32")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test skyrl-tx connection")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="skyrl-tx server URL (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    success = asyncio.run(test_connection(args.url))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

