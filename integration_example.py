"""
Quick integration example for border smoothing in your inference pipeline.

Add this code to inference_without_train.py after the main video generation.
"""

# ============================================================================
# ADD THIS TO YOUR inference_without_train.py (after main video generation)
# ============================================================================

def apply_border_smoothing_postprocess(
    edited_video_path: str,
    original_video_path: str, 
    mask_video_path: str,
    output_video_path: str,
    blend_strength: float = 0.6,
    enable: bool = True
):
    """
    Apply border smoothing as post-processing step after inference.
    
    USAGE IN MAIN CODE:
    ```
    # After existing inference code that creates edited_video.mp4
    output_path = os.path.join(data_dir, "edited_video.mp4")
    save_video(video, output_path, fps=30, quality=5)
    
    # Add this:
    if os.path.exists(mask_video_path):
        smoothed_path = os.path.join(data_dir, "edited_video_smooth.mp4")
        apply_border_smoothing_postprocess(
            edited_video_path=output_path,
            original_video_path=os.path.join(data_dir, "input_video.mp4"),
            mask_video_path=mask_video_path,
            output_video_path=smoothed_path,
            blend_strength=0.6,
            enable=True
        )
    ```
    """
    if not enable:
        print("Border smoothing disabled, skipping...")
        return
    
    try:
        from border_smoothing_inference import apply_border_smoothing_to_video
        
        print("\n" + "="*60)
        print("APPLYING BORDER SMOOTHING FOR JITTER REDUCTION")
        print("="*60)
        print(f"Edited video: {edited_video_path}")
        print(f"Original video: {original_video_path}")
        print(f"Mask video: {mask_video_path}")
        print(f"Output: {output_video_path}")
        print(f"Blend strength: {blend_strength}")
        print("="*60)
        
        apply_border_smoothing_to_video(
            edited_video_path=edited_video_path,
            original_video_path=original_video_path,
            mask_video_path=mask_video_path,
            output_path=output_video_path,
            blend_strength=blend_strength,
            border_width=5,
            blur_sigma=1.5
        )
        
        print("✓ Border smoothing completed successfully!")
        print(f"✓ Output saved to: {output_video_path}")
        
    except Exception as e:
        print(f"⚠ Warning: Border smoothing failed - {e}")
        print("Continuing with original video...")


# ============================================================================
# ALTERNATIVE: Inline implementation (if you prefer no additional imports)
# ============================================================================

def apply_border_smoothing_inline(
    edited_video_frames,
    original_video_frames,
    mask_video_frames,
    blend_strength: float = 0.6,
    border_width: int = 5,
    blur_sigma: float = 1.5
):
    """
    Apply border smoothing directly to frame lists.
    
    Args:
        edited_video_frames: List of PIL Images
        original_video_frames: List of PIL Images  
        mask_video_frames: List of PIL Images (grayscale masks)
        blend_strength: Blending strength (0-1)
        border_width: Width of border in pixels
        blur_sigma: Gaussian blur sigma
    
    Returns:
        List of blended PIL Images
    """
    import cv2
    import numpy as np
    from PIL import Image
    
    output_frames = []
    
    for i, (edited, original, mask) in enumerate(zip(edited_video_frames, original_video_frames, mask_video_frames)):
        if (i + 1) % max(1, len(edited_video_frames) // 10) == 0:
            print(f"Processing frame {i+1}/{len(edited_video_frames)}")
        
        # Convert to numpy
        edited_np = np.array(edited, dtype=np.float32) / 255.0
        original_np = np.array(original, dtype=np.float32) / 255.0
        mask_np = np.array(mask.convert('L'), dtype=np.float32) / 255.0
        
        # Binary mask
        mask_binary = (mask_np > 0.5).astype(np.uint8)
        
        # Create smooth blend mask
        kernel_size = int(4 * blur_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Gaussian blur on mask
        smoothed_mask = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), blur_sigma)
        
        # Distance transform for border detection
        dist = cv2.distanceTransform(mask_binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        max_dist = max(dist.max(), 1.0)
        
        # Blend mask: high at border, low inside
        blend_mask = (1.0 - dist / max_dist) * blend_strength
        blend_mask = np.clip(blend_mask, 0, 1)
        
        # Apply blending
        blended_np = (
            edited_np * (1 - blend_mask[:,:,np.newaxis]) + 
            original_np * blend_mask[:,:,np.newaxis]
        )
        
        # Convert back to PIL
        blended_np = np.clip(blended_np * 255, 0, 255).astype(np.uint8)
        output_frames.append(Image.fromarray(blended_np))
    
    return output_frames


# ============================================================================
# TESTING / DEBUGGING
# ============================================================================

def test_border_smoothing():
    """
    Test border smoothing with sample data.
    Run this to verify everything works.
    """
    print("Testing border smoothing implementation...")
    
    try:
        from border_smoothing_inference import BorderAttentionBlender
        
        # Create simple test tensors
        device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
        blender = BorderAttentionBlender(device=device)
        
        # Create test mask
        import torch
        mask = torch.zeros(480, 832, device=device)
        mask[100:400, 200:700] = 1.0
        
        # Create border mask
        border_mask = blender.create_border_mask(mask)
        
        print(f"✓ Test mask shape: {mask.shape}")
        print(f"✓ Border mask shape: {border_mask.shape}")
        print(f"✓ Border mask values (min, max): {border_mask.min():.3f}, {border_mask.max():.3f}")
        print("✓ Border smoothing test PASSED!")
        
    except Exception as e:
        print(f"✗ Border smoothing test FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test the implementation
    test_border_smoothing()
