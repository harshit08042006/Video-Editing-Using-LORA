"""
Border-Aware Attention Blending for Jitter Reduction

This module provides attention map blending at mask borders to reduce jitter
while preserving the background completely and edited foreground motion.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple


class BorderAttentionBlender:
    """
    Blends attention maps at mask borders to reduce jitter while preserving
    background quality and edited foreground motion.
    """
    
    def __init__(self, device="cuda", dilation_kernel_size: int = 5, blend_width: int = 3):
        """
        Args:
            device: torch device
            dilation_kernel_size: Size of kernel for dilating masks (larger = wider blend zone)
            blend_width: Width of blend zone in pixels
        """
        self.device = device
        self.dilation_kernel_size = dilation_kernel_size
        self.blend_width = blend_width
        self.dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (dilation_kernel_size, dilation_kernel_size)
        )
    
    
    def create_border_mask(self, mask: torch.Tensor, dilate_kernel: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Create a soft border mask from a binary mask.
        
        Args:
            mask: Binary mask tensor of shape (B, 1, T, H, W) or (H, W) with values 0 or 1
            dilate_kernel: Optional cv2 kernel for dilation
        
        Returns:
            Soft border mask with smooth transitions (0 to 1)
        """
        if dilate_kernel is None:
            dilate_kernel = self.dilation_kernel
        
        # Handle different input shapes
        if mask.dim() == 5:  # (B, 1, T, H, W)
            B, C, T, H, W = mask.shape
            mask_np = mask[0, 0].cpu().numpy().astype(np.uint8) * 255
        elif mask.dim() == 4:  # (B, T, H, W)
            B, T, H, W = mask.shape
            mask_np = mask[0].cpu().numpy().astype(np.uint8) * 255
        elif mask.dim() == 2:  # (H, W)
            H, W = mask.shape
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")
        
        # Dilate mask to find border region
        dilated = cv2.dilate(mask_np, dilate_kernel, iterations=1)
        eroded = cv2.erode(mask_np, dilate_kernel, iterations=1)
        
        # Border is where dilated != eroded
        border = (dilated != eroded).astype(np.float32)
        
        # Create soft blend using distance transform
        dist_transform = cv2.distanceTransform(dilated, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        
        # Normalize distance to create smooth blend (0 at edges, 1 at center)
        max_dist = dist_transform.max()
        if max_dist > 0:
            normalized_dist = dist_transform / max_dist
        else:
            normalized_dist = np.zeros_like(dist_transform)
        
        # Create blend mask: higher values at border, lower inside
        blend_mask = 1.0 - normalized_dist
        blend_mask = np.clip(blend_mask, 0, 1)
        
        # Apply border to limit blend region
        blend_mask = blend_mask * border
        
        return torch.from_numpy(blend_mask).float().to(self.device)
    
    
    def blend_attention_at_border(
        self, 
        attention_edited: torch.Tensor,
        attention_original: torch.Tensor,
        mask: torch.Tensor,
        blend_strength: float = 1.0
    ) -> torch.Tensor:
        """
        Blend edited and original attention maps at mask borders.
        
        Args:
            attention_edited: Edited attention maps, shape matching mask
            attention_original: Original attention maps, same shape
            mask: Binary mask indicating foreground (1) and background (0)
            blend_strength: Strength of blending (0=all edited, 1=full blend at border)
        
        Returns:
            Blended attention maps
        """
        # Create border blend mask
        border_blend = self.create_border_mask(mask)
        
        # Ensure all tensors have same device and shape
        border_blend = border_blend.to(attention_edited.device)
        
        # Reshape border_blend to match attention shape if needed
        if border_blend.dim() != attention_edited.dim():
            # Expand dimensions to match attention
            while border_blend.dim() < attention_edited.dim():
                if border_blend.dim() == 2 and attention_edited.dim() == 5:
                    # Insert T, C dimensions
                    border_blend = border_blend.unsqueeze(0).unsqueeze(0)
                else:
                    border_blend = border_blend.unsqueeze(-1)
        
        # Interpolate border_blend to match attention spatial dimensions if needed
        if border_blend.shape[-2:] != attention_edited.shape[-2:]:
            border_blend = F.interpolate(
                border_blend.unsqueeze(0).unsqueeze(0),
                size=attention_edited.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
        
        # Blend: Use original attention at borders, edited attention inside foreground
        blend_factor = border_blend * blend_strength
        blended = attention_edited * (1 - blend_factor) + attention_original * blend_factor
        
        return blended
    
    
    def extract_border_region(
        self, 
        mask: torch.Tensor,
        border_width: int = 5
    ) -> torch.Tensor:
        """
        Extract only the border region (useful for visualization/debugging).
        
        Args:
            mask: Binary mask
            border_width: Width of border region in pixels
        
        Returns:
            Border region mask
        """
        if mask.dim() > 2:
            mask = mask.squeeze()
        
        mask_np = mask.cpu().numpy().astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_width * 2 + 1, border_width * 2 + 1))
        
        dilated = cv2.dilate(mask_np, kernel, iterations=1)
        eroded = cv2.erode(mask_np, kernel, iterations=1)
        border = (dilated != eroded).astype(np.float32)
        
        return torch.from_numpy(border).float().to(self.device)
    
    
    def smooth_mask_boundary(
        self, 
        mask: torch.Tensor, 
        sigma: float = 1.0
    ) -> torch.Tensor:
        """
        Smooth mask boundaries using Gaussian blur to avoid hard transitions.
        
        Args:
            mask: Binary mask
            sigma: Gaussian blur sigma
        
        Returns:
            Smoothed mask
        """
        if mask.dim() == 2:
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask.squeeze().cpu().numpy()
        
        # Apply Gaussian blur using cv2
        kernel_size = int(4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        smoothed_np = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), sigma)
        smoothed = torch.from_numpy(smoothed_np).float().to(self.device)
        
        return smoothed


# Utility function for post-processing
def apply_border_attention_blending(
    video_frames: torch.Tensor,
    mask_video: torch.Tensor,
    original_video: torch.Tensor,
    blend_strength: float = 0.5,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Apply border attention blending to entire video.
    
    Args:
        video_frames: Edited video frames (B, C, T, H, W)
        mask_video: Mask video (T, H, W) or (1, T, H, W)
        original_video: Original video frames (B, C, T, H, W)
        blend_strength: Blending strength
        device: torch device
    
    Returns:
        Blended video frames
    """
    blender = BorderAttentionBlender(device=device)
    
    # Ensure mask has correct shape
    if mask_video.dim() == 3:
        mask_video = mask_video.unsqueeze(0)  # Add batch dim
    
    # Process each frame
    blended_video = video_frames.clone()
    
    for t in range(video_frames.shape[2]):  # Iterate over time
        frame_mask = mask_video[0, t] if mask_video.dim() == 4 else mask_video[t]
        
        # Blend using Gaussian smoothing on mask boundary
        smoothed_mask = blender.smooth_mask_boundary(frame_mask, sigma=1.5)
        
        # Apply smoothed mask to blend original background
        blended_video[:, :, t] = (
            video_frames[:, :, t] * smoothed_mask +
            original_video[:, :, t] * (1 - smoothed_mask)
        )
    
    return blended_video
