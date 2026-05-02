#!/usr/bin/env python3
"""
Quick verification script for border smoothing implementation.

Run this to check if everything is installed and working correctly.
"""

import sys
import os

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...")
    
    dependencies = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (MISSING)")
            missing.append(name)
    
    return len(missing) == 0


def check_files():
    """Check if all required files exist."""
    print("\nChecking required files...")
    
    files = [
        'attention_border_blending.py',
        'border_smoothing_inference.py',
        'integration_example.py',
        'JITTER_REDUCTION_GUIDE.md',
        'PARAMETER_TUNING.py',
    ]
    
    all_exist = True
    for filename in files:
        exists = os.path.exists(filename)
        status = "✓" if exists else "✗"
        print(f"  {status} {filename}")
        if not exists:
            all_exist = False
    
    return all_exist


def test_imports():
    """Test if modules can be imported."""
    print("\nTesting module imports...")
    
    try:
        from attention_border_blending import BorderAttentionBlender
        print("  ✓ BorderAttentionBlender imported")
    except Exception as e:
        print(f"  ✗ Failed to import BorderAttentionBlender: {e}")
        return False
    
    try:
        from border_smoothing_inference import load_video_frames
        print("  ✓ border_smoothing_inference imported")
    except Exception as e:
        print(f"  ✗ Failed to import border_smoothing_inference: {e}")
        return False
    
    return True


def test_border_blender():
    """Test BorderAttentionBlender functionality."""
    print("\nTesting BorderAttentionBlender...")
    
    try:
        import torch
        from attention_border_blending import BorderAttentionBlender
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        blender = BorderAttentionBlender(device=device)
        
        # Create test mask
        mask = torch.zeros(480, 832, device=device)
        mask[100:400, 200:700] = 1.0
        
        # Test border mask creation
        border_mask = blender.create_border_mask(mask)
        
        if border_mask.shape == mask.shape:
            print(f"  ✓ Border mask creation works (shape: {border_mask.shape})")
        else:
            print(f"  ✗ Border mask shape mismatch: expected {mask.shape}, got {border_mask.shape}")
            return False
        
        if 0 <= border_mask.min() <= 1 and 0 <= border_mask.max() <= 1:
            print(f"  ✓ Border mask values in valid range: [{border_mask.min():.3f}, {border_mask.max():.3f}]")
        else:
            print(f"  ✗ Border mask values out of range: [{border_mask.min():.3f}, {border_mask.max():.3f}]")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ BorderAttentionBlender test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_smooth_mask():
    """Test mask smoothing."""
    print("\nTesting mask smoothing...")
    
    try:
        import torch
        from attention_border_blending import BorderAttentionBlender
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        blender = BorderAttentionBlender(device=device)
        
        # Create test mask
        mask = torch.zeros(480, 832, device=device)
        mask[100:400, 200:700] = 1.0
        
        # Test smoothing
        smoothed = blender.smooth_mask_boundary(mask, sigma=1.5)
        
        # Check that smoothed mask has values between 0 and 1
        if 0 <= smoothed.min() <= 1 and 0 <= smoothed.max() <= 1:
            print(f"  ✓ Mask smoothing works: values in [{smoothed.min():.3f}, {smoothed.max():.3f}]")
            return True
        else:
            print(f"  ✗ Smoothed mask values out of range: [{smoothed.min():.3f}, {smoothed.max():.3f}]")
            return False
        
    except Exception as e:
        print(f"  ✗ Mask smoothing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results):
    """Print summary of all tests."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:.<40} {status}")
    
    print("="*60)
    
    if all_passed:
        print("✓ All tests passed! Ready to use.")
        return 0
    else:
        print("✗ Some tests failed. Please check errors above.")
        return 1


def print_quick_start():
    """Print quick start instructions."""
    print("\n" + "="*60)
    print("QUICK START")
    print("="*60)
    
    print("""
After inference generates edited_video.mp4, run:

python border_smoothing_inference.py \\
  --edited_video output/edited_video.mp4 \\
  --original_video input/original_video.mp4 \\
  --mask_video input/mask.mp4 \\
  --output output/edited_video_smooth.mp4 \\
  --blend_strength 0.6

For help with parameters:

python PARAMETER_TUNING.py interactive

For more information, see:

- README_BORDER_SMOOTHING.md (overview)
- JITTER_REDUCTION_GUIDE.md (detailed guide)
- PARAMETER_TUNING.py (parameter help)
""")


def main():
    """Run all verification tests."""
    print("="*60)
    print("BORDER SMOOTHING VERIFICATION")
    print("="*60)
    
    results = {
        "Dependencies": check_dependencies(),
        "Required Files": check_files(),
        "Module Imports": test_imports(),
        "BorderAttentionBlender": test_border_blender(),
        "Mask Smoothing": test_smooth_mask(),
    }
    
    exit_code = print_summary(results)
    
    if exit_code == 0:
        print_quick_start()
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
