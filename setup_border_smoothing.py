"""
Setup script for border smoothing implementation.

Run this to install missing dependencies.
"""

import subprocess
import sys


def install_opencv():
    """Install OpenCV."""
    print("Installing OpenCV...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python", "-q"])
        print("✓ OpenCV installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install OpenCV")
        return False


def check_and_install_dependencies():
    """Check and install required dependencies."""
    print("Checking and installing dependencies...\n")
    
    required = {
        'torch': 'PyTorch (should already be installed)',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name} is already installed")
        except ImportError:
            print(f"✗ {name} is missing, will install...")
            missing.append((module, name))
    
    if not missing:
        print("\n✓ All dependencies are installed!")
        return True
    
    print(f"\nInstalling {len(missing)} missing packages...\n")
    
    # Install OpenCV
    if 'cv2' in [m[0] for m in missing]:
        if not install_opencv():
            return False
    
    # Install other packages via pip
    pip_packages = {
        'PIL': 'Pillow',
        'numpy': 'numpy',
    }
    
    for module, name in missing:
        if module in pip_packages:
            pkg_name = pip_packages[module]
            print(f"Installing {name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name, "-q"])
                print(f"✓ {name} installed successfully")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {name}")
                return False
    
    print("\n✓ All dependencies installed successfully!")
    return True


def verify_installation():
    """Verify that everything works."""
    print("\nVerifying installation...\n")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
        
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        
        from PIL import Image
        print(f"✓ Pillow installed")
        
        # Try importing our modules
        from attention_border_blending import BorderAttentionBlender
        print("✓ attention_border_blending module works")
        
        from border_smoothing_inference import apply_border_smoothing_to_video
        print("✓ border_smoothing_inference module works")
        
        print("\n✓✓✓ All systems ready! ✓✓✓")
        return True
        
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        return False


if __name__ == "__main__":
    import os
    
    print("="*60)
    print("BORDER SMOOTHING SETUP")
    print("="*60)
    print()
    
    # Check if we're in the right directory
    if not os.path.exists('attention_border_blending.py'):
        print("ERROR: Not in LoRAEdit directory!")
        print("Please run this script from the LoRAEdit root directory.")
        sys.exit(1)
    
    # Install dependencies
    if not check_and_install_dependencies():
        print("\n✗ Setup failed. Please install dependencies manually:")
        print("  pip install opencv-python torch pillow numpy")
        sys.exit(1)
    
    # Verify
    if not verify_installation():
        print("\n✗ Verification failed. Please check your installation.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nYou can now use border smoothing:")
    print("\nQuick start:")
    print("  python border_smoothing_inference.py --help")
    print("\nFor interactive parameter selection:")
    print("  python PARAMETER_TUNING.py interactive")
    print("\nFor detailed guide:")
    print("  cat README_BORDER_SMOOTHING.md")
    print()
