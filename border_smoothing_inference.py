"""
Enhanced inference with border attention blending for jitter reduction.

This module wraps the existing inference pipeline and applies border-aware
attention blending to reduce jitter at mask boundaries.
"""

import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Optional
from attention_border_blending import BorderAttentionBlender
import torch.nn.functional as F


def load_video_frames(video_path: str, target_height: Optional[int] = None, target_width: Optional[int] = None) -> tuple:
    """
    Load video frames from file or folder.
    
    Args:
        video_path: Path to video file OR folder containing image frames
        target_height: Target height (optional)
        target_width: Target width (optional)
    
    Returns:
        (frames_list, original_height, original_width, fps)
    """
    import os
    from pathlib import Path
    
    frames = []
    fps = 30  # Default FPS
    
    # Check if input is a folder or video file
    if os.path.isdir(video_path):
        # Load frames from folder
        print(f"Loading frames from folder: {video_path}")
        
        # Get all image files (sorted)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = sorted([
            os.path.join(video_path, f) for f in os.listdir(video_path)
            if os.path.splitext(f)[1].lower() in image_extensions
        ])
        
        if not image_files:
            raise ValueError(f"No image files found in folder: {video_path}")
        
        print(f"Found {len(image_files)} frames")
        
        # Load frames
        for img_path in image_files:
            try:
                frame = cv2.imread(img_path)
                if frame is None:
                    print(f"Warning: Failed to read {img_path}, skipping")
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if target_height and target_width:
                    frame = cv2.resize(frame, (target_width, target_height))
                
                frames.append(Image.fromarray(frame))
            except Exception as e:
                print(f"Warning: Error reading {img_path}: {e}")
                continue
        
        # Get dimensions from first frame
        if frames:
            first_frame = cv2.imread(image_files[0])
            original_height, original_width = first_frame.shape[:2]
        else:
            raise ValueError("No valid frames could be loaded")
        
    else:
        # Load from video file
        print(f"Loading frames from video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if target_height and target_width:
                frame = cv2.resize(frame, (target_width, target_height))
            
            frames.append(Image.fromarray(frame))
        
        cap.release()
    
    print(f"Loaded {len(frames)} frames (size: {original_width}x{original_height}, fps: {fps})")
    return frames, original_height, original_width, fps


def apply_border_smoothing_to_video(
    edited_video_path: str,
    original_video_path: str,
    mask_video_path: str,
    output_path: str,
    blend_strength: float = 0.6,
    border_width: int = 5,
    blur_sigma: float = 1.5
) -> None:
    """
    Apply border smoothing to reduce jitter at mask edges.
    
    This function:
    1. Loads edited video, original video, and mask
    2. Creates soft border mask using morphological operations
    3. Blends edited video with original at borders
    4. Saves result
    
    Args:
        edited_video_path: Path to edited video
        original_video_path: Path to original video
        mask_video_path: Path to mask video (white = foreground, black = background)
        output_path: Output video path
        blend_strength: How much to blend at borders (0-1)
        border_width: Width of border region to blend
        blur_sigma: Sigma for Gaussian blur on mask boundary
    """
    import os
    from diffsynth import save_video
    
    # Load videos
    print(f"Loading videos...")
    edited_frames, h, w, fps = load_video_frames(edited_video_path)
    original_frames, _, _, _ = load_video_frames(original_video_path, h, w)
    mask_frames, _, _, _ = load_video_frames(mask_video_path, h, w)
    
    if len(edited_frames) != len(original_frames):
        raise ValueError(f"Frame count mismatch: edited {len(edited_frames)} vs original {len(original_frames)}")
    
    if len(mask_frames) != len(edited_frames):
        raise ValueError(f"Frame count mismatch: mask {len(mask_frames)} vs edited {len(edited_frames)}")
    
    print(f"Processing {len(edited_frames)} frames...")
    
    # Initialize blender
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blender = BorderAttentionBlender(device=device, blend_width=border_width)
    
    # Process each frame
    output_frames = []
    for i, (edited_frame, original_frame, mask_frame) in enumerate(zip(edited_frames, original_frames, mask_frames)):
        if (i + 1) % 10 == 0:
            print(f"  Processing frame {i+1}/{len(edited_frames)}")
        
        # Convert to numpy
        edited_np = np.array(edited_frame, dtype=np.float32) / 255.0
        original_np = np.array(original_frame, dtype=np.float32) / 255.0
        
        # Convert mask to binary (white=1, black=0)
        mask_np = np.array(mask_frame.convert('L'), dtype=np.float32) / 255.0
        mask_np = (mask_np > 0.5).astype(np.float32)
        
        # Smooth mask boundary using Gaussian blur
        kernel_size = int(4 * blur_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed_mask = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), blur_sigma)
        
        # Create dilated/eroded border mask for extra smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_width * 2 + 1, border_width * 2 + 1))
        dilated = cv2.dilate(mask_np, kernel, iterations=1)
        eroded = cv2.erode(mask_np, kernel, iterations=1)
        border_region = (dilated != eroded).astype(np.float32)
        
        # Create blend mask: soft transition at borders
        distance_transform = cv2.distanceTransform((mask_np * 255).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        max_dist = max(distance_transform.max(), 1)
        blend_mask = distance_transform / max_dist
        blend_mask = 1.0 - blend_mask  # Invert: high at border, low inside
        blend_mask = np.clip(blend_mask * blend_strength, 0, 1) * border_region
        
        # Blend: use original at borders, edited inside
        blended_np = edited_np * (1 - blend_mask[:, :, np.newaxis]) + original_np * blend_mask[:, :, np.newaxis]
        
        # Convert back to PIL Image
        blended_np = np.clip(blended_np * 255, 0, 255).astype(np.uint8)
        output_frames.append(Image.fromarray(blended_np))
    
    print(f"Saving output video to {output_path}...")
    save_video(output_frames, output_path, fps=fps, quality=5)
    print("Done!")


def create_smoothed_mask_video(
    mask_video_path: str,
    output_path: str,
    blur_sigma: float = 2.0
) -> None:
    """
    Create a smoothed version of mask video for visualization.
    
    Args:
        mask_video_path: Input mask video path
        output_path: Output smoothed mask video path
        blur_sigma: Gaussian blur sigma
    """
    from diffsynth import save_video
    
    print(f"Creating smoothed mask video...")
    mask_frames, h, w, fps = load_video_frames(mask_video_path)
    
    smoothed_frames = []
    kernel_size = int(4 * blur_sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    for mask_frame in mask_frames:
        mask_np = np.array(mask_frame.convert('L'), dtype=np.float32) / 255.0
        smoothed_np = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), blur_sigma)
        smoothed_np = np.clip(smoothed_np * 255, 0, 255).astype(np.uint8)
        smoothed_frames.append(Image.fromarray(smoothed_np).convert('RGB'))
    
    save_video(smoothed_frames, output_path, fps=fps, quality=5)
    print(f"Smoothed mask saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply border smoothing to reduce jitter")
    parser.add_argument("--edited_video", required=True, help="Path to edited video")
    parser.add_argument("--original_video", required=True, help="Path to original video")
    parser.add_argument("--mask_video", required=True, help="Path to mask video")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--blend_strength", type=float, default=0.6, help="Blend strength (0-1)")
    parser.add_argument("--border_width", type=int, default=5, help="Border width in pixels")
    parser.add_argument("--blur_sigma", type=float, default=1.5, help="Gaussian blur sigma")
    
    args = parser.parse_args()
    
    apply_border_smoothing_to_video(
        args.edited_video,
        args.original_video,
        args.mask_video,
        args.output,
        blend_strength=args.blend_strength,
        border_width=args.border_width,
        blur_sigma=args.blur_sigma
    )
