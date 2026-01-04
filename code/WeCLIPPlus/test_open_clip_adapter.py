"""
Test script for open_clip adapter
This script tests basic functionality of the adapter to ensure it works correctly.
"""

import sys
import os
import torch

# Add the parent directory to path so we can import clip
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clip

def test_basic_loading():
    """Test that we can load a model"""
    print("=" * 60)
    print("Test 1: Basic Model Loading")
    print("=" * 60)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Try to load ViT-B/16 model (commonly used in WeCLIP+)
        print("\nLoading ViT-B/16 model...")
        model, preprocess = clip.load("ViT-B/16", device=device)
        print("✓ Model loaded successfully!")

        # Check if model has expected attributes
        print("\nChecking model attributes...")
        assert hasattr(model, 'visual'), "Model missing 'visual' attribute"
        assert hasattr(model, 'encode_image'), "Model missing 'encode_image' method"
        assert hasattr(model, 'encode_text'), "Model missing 'encode_text' method"
        print("✓ All expected attributes present")

        return model, preprocess, device

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_tokenization():
    """Test text tokenization"""
    print("\n" + "=" * 60)
    print("Test 2: Text Tokenization")
    print("=" * 60)

    try:
        text = ["a photo of a cat", "a photo of a dog"]
        print(f"\nTokenizing: {text}")
        tokens = clip.tokenize(text)
        print(f"✓ Tokenization successful!")
        print(f"  Token shape: {tokens.shape}")
        print(f"  Expected shape: [2, 77]")
        assert tokens.shape == (2, 77), f"Unexpected token shape: {tokens.shape}"
        print("✓ Token shape correct")

        return tokens

    except Exception as e:
        print(f"✗ Error in tokenization: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_text_encoding(model, device):
    """Test text encoding"""
    print("\n" + "=" * 60)
    print("Test 3: Text Encoding")
    print("=" * 60)

    if model is None:
        print("✗ Skipping (model not loaded)")
        return None

    try:
        text = ["a photo of a cat", "a photo of a dog"]
        print(f"\nEncoding text: {text}")
        tokens = clip.tokenize(text).to(device)

        with torch.no_grad():
            text_features = model.encode_text(tokens)

        print(f"✓ Text encoding successful!")
        print(f"  Text features shape: {text_features.shape}")
        print(f"  Expected shape: [2, 512] or [2, 768]")

        return text_features

    except Exception as e:
        print(f"✗ Error in text encoding: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_image_encoding(model, device):
    """Test image encoding with custom dimensions"""
    print("\n" + "=" * 60)
    print("Test 4: Image Encoding")
    print("=" * 60)

    if model is None:
        print("✗ Skipping (model not loaded)")
        return None

    try:
        # Create a dummy image tensor
        H, W = 224, 224
        dummy_image = torch.randn(1, 3, H, W).to(device)
        print(f"\nEncoding image of shape: {dummy_image.shape}")

        with torch.no_grad():
            # Test with the WeCLIP+ interface
            image_features, attn_weights = model.encode_image(
                dummy_image, H, W, require_all_fts=False, clip_flag=16
            )

        print(f"✓ Image encoding successful!")
        print(f"  Image features type: {type(image_features)}")
        print(f"  Attention weights type: {type(attn_weights)}")

        return image_features, attn_weights

    except Exception as e:
        print(f"✗ Error in image encoding: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_visual_transformer_attributes(model):
    """Test that visual transformer has expected attributes"""
    print("\n" + "=" * 60)
    print("Test 5: Visual Transformer Attributes")
    print("=" * 60)

    if model is None:
        print("✗ Skipping (model not loaded)")
        return

    try:
        print("\nChecking visual.transformer attributes...")
        assert hasattr(model.visual, 'transformer'), "Missing 'transformer' attribute"
        assert hasattr(model.visual.transformer, 'resblocks'), "Missing 'resblocks' attribute"
        assert hasattr(model.visual.transformer, 'layers'), "Missing 'layers' attribute"

        num_layers = model.visual.transformer.layers
        print(f"✓ Visual transformer has {num_layers} layers")

        # Check that resblocks is indexable
        last_block = model.visual.transformer.resblocks[-1]
        print(f"✓ Can access transformer blocks (last block: {type(last_block).__name__})")

        return True

    except Exception as e:
        print(f"✗ Error checking visual transformer: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("OpenCLIP Adapter Test Suite")
    print("=" * 60)

    # Run tests
    model, preprocess, device = test_basic_loading()
    test_tokenization()
    test_text_encoding(model, device)
    test_image_encoding(model, device)
    test_visual_transformer_attributes(model)

    print("\n" + "=" * 60)
    print("Test Suite Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
