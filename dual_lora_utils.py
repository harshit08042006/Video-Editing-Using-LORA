"""
Dual-LoRA Manager for Mask-Aware Attention Blending

This module enables loading two LoRA weight sets (WAN video + original video)
and blending their contributions per-token using a spatial mask at inference time.

Key idea:
  - Inside the foreground mask → use WAN LoRA (learned motion)
  - Outside the foreground mask → use original LoRA (stable background)
  - At the mask border → smooth Gaussian blend between both

This preserves foreground motion fidelity while eliminating border jitter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from safetensors import safe_open
from typing import Optional, Dict, Tuple, List


class DualLoRAManager:
    """
    Manages two sets of LoRA weights and blends their outputs per-token
    using a spatial mask during inference.
    
    The mask is 1 in the foreground (WAN LoRA) and 0 in the background
    (original LoRA), with smooth Gaussian transitions at the border.
    """

    def __init__(
        self,
        dit_model: nn.Module,
        wan_lora_path: str,
        orig_lora_path: str,
        lora_alpha: float = 1.0,
        border_sigma: float = 3.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            dit_model: The WanModel (DiT transformer) to attach hooks to.
                       Must be called AFTER enable_vram_management().
            wan_lora_path: Path to the WAN video LoRA safetensors file.
            orig_lora_path: Path to the original video LoRA safetensors file.
            lora_alpha: LoRA scaling factor (matches DiffSynth's load_lora alpha).
            border_sigma: Gaussian blur sigma for soft mask borders (in patch-grid pixels).
            device: Device to store LoRA matrices on.
            dtype: Data type for LoRA matrices.
        """
        self.lora_alpha = lora_alpha
        self.border_sigma = border_sigma
        self.device = device
        self.dtype = dtype
        self.hooks: List[torch.utils.hooks.RemovableHook] = []
        
        # Token-space mask: (1, seq_len, 1) where 1=foreground(WAN), 0=background(orig)
        self.token_mask: Optional[torch.Tensor] = None

        # Load both LoRA weight dictionaries
        print(f"Loading WAN LoRA from: {wan_lora_path}")
        wan_weights = self._load_lora_weights(wan_lora_path)
        print(f"Loading original LoRA from: {orig_lora_path}")
        orig_weights = self._load_lora_weights(orig_lora_path)

        # Register forward hooks on all targeted modules
        num_hooks = self._register_hooks(dit_model, wan_weights, orig_weights)
        print(f"Dual-LoRA: Registered {num_hooks} hooks on transformer layers")
        print(f"  Border sigma: {border_sigma} (patch-grid pixels)")
        print(f"  Alpha: {lora_alpha}")

    def _load_lora_weights(self, path: str) -> Dict[str, torch.Tensor]:
        """Load LoRA weights from a safetensors file, keeping them on the target device."""
        weights = {}
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key).to(self.device, self.dtype)
        print(f"  Loaded {len(weights)} weight tensors")
        return weights

    def _register_hooks(
        self,
        model: nn.Module,
        wan_weights: Dict[str, torch.Tensor],
        orig_weights: Dict[str, torch.Tensor],
    ) -> int:
        """
        Register forward hooks on each Linear layer that has LoRA weights.
        
        Returns the number of hooks registered.
        """
        # Build a dict of module_path -> module for the model
        module_dict = {}
        for name, mod in model.named_modules():
            module_dict[name] = mod

        # Group LoRA weights by target module
        # Key format: "diffusion_model.blocks.0.self_attn.q.lora_B.weight"
        # Module path in model: "blocks.0.self_attn.q"
        # Target param in model: "blocks.0.self_attn.q.weight"
        module_lora_pairs = {}
        for key in wan_weights:
            if ".lora_A.weight" not in key:
                continue
            # Extract module path: strip "diffusion_model." prefix and ".lora_A.weight" suffix
            module_path = key.replace("diffusion_model.", "").replace(".lora_A.weight", "")
            lora_b_key = key.replace("lora_A", "lora_B")

            if module_path not in module_dict:
                print(f"  WARNING: Module '{module_path}' not found in model, skipping")
                continue
            if key not in orig_weights or lora_b_key not in orig_weights:
                print(f"  WARNING: Original LoRA missing key for '{module_path}', skipping")
                continue

            module_lora_pairs[module_path] = {
                "wan_A": wan_weights[key],           # (rank, in_dim)
                "wan_B": wan_weights[lora_b_key],    # (out_dim, rank)
                "orig_A": orig_weights[key],         # (rank, in_dim)
                "orig_B": orig_weights[lora_b_key],  # (out_dim, rank)
            }

        # Register hooks
        hook_count = 0
        for module_path, lora_weights in module_lora_pairs.items():
            module = module_dict[module_path]

            # Determine if this layer operates on context tokens (not spatial)
            # cross_attn.k and cross_attn.v operate on text/CLIP context, not spatial tokens
            is_context_layer = "cross_attn.k" in module_path or "cross_attn.v" in module_path

            hook = self._create_hook(lora_weights, is_context_layer, module_path)
            handle = module.register_forward_hook(hook)
            self.hooks.append(handle)
            hook_count += 1

        return hook_count

    def _create_hook(
        self,
        lora_weights: Dict[str, torch.Tensor],
        is_context_layer: bool,
        module_path: str,
    ):
        """
        Create a forward hook that computes blended LoRA output.
        
        For spatial layers:
            delta = mask * wan_delta + (1 - mask) * orig_delta
        For context layers (cross_attn k, v):
            delta = wan_delta  (no spatial blending needed)
        """
        wan_A = lora_weights["wan_A"]    # (rank, in_dim)
        wan_B = lora_weights["wan_B"]    # (out_dim, rank)
        orig_A = lora_weights["orig_A"]  # (rank, in_dim)
        orig_B = lora_weights["orig_B"]  # (out_dim, rank)
        alpha = self.lora_alpha
        manager = self  # Closure reference to access token_mask

        def hook(module, input, output):
            x = input[0]  # (batch, seq_len, dim) or (batch, dim) for some layers
            
            # Compute WAN LoRA delta: alpha * (x @ A^T) @ B^T
            # = alpha * x @ A^T @ B^T  (matrix chain)
            wan_delta = alpha * (x @ wan_A.T @ wan_B.T)

            if is_context_layer or manager.token_mask is None:
                # For context layers or when no mask is set: use pure WAN LoRA
                return output + wan_delta

            # Compute original LoRA delta
            orig_delta = alpha * (x @ orig_A.T @ orig_B.T)

            # Get the stored token mask
            mask = manager.token_mask  # (1, seq_len, 1)

            # Handle sequence length mismatches (e.g., due to padding)
            if x.dim() == 3 and mask.dim() == 3:
                if mask.shape[1] != x.shape[1]:
                    if mask.shape[1] > x.shape[1]:
                        mask = mask[:, :x.shape[1], :]
                    else:
                        # Pad with 0 → background → use original LoRA for padding tokens
                        pad_size = x.shape[1] - mask.shape[1]
                        pad = torch.zeros(
                            mask.shape[0], pad_size, 1,
                            device=mask.device, dtype=mask.dtype,
                        )
                        mask = torch.cat([mask, pad], dim=1)

            # Blend: foreground (mask=1) → WAN, background (mask=0) → original
            blended_delta = mask * wan_delta + (1 - mask) * orig_delta
            return output + blended_delta

        return hook

    def set_mask_from_video(
        self,
        mask_video_path: str,
        num_frames: int,
        height: int,
        width: int,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        sigma: Optional[float] = None,
    ):
        """
        Read the mask video and convert it to a token-space mask with soft borders.
        
        The mask video convention:
          - White pixels = foreground (area being edited) → use WAN LoRA
          - Black pixels = background (preserved area) → use original LoRA
        
        Args:
            mask_video_path: Path to the mask video file.
            num_frames: Total number of video frames.
            height: Video height in pixels.
            width: Video width in pixels.
            patch_size: Transformer patch size (temporal, spatial_h, spatial_w).
            sigma: Gaussian blur sigma for border softening. Uses self.border_sigma if None.
        """
        if sigma is None:
            sigma = self.border_sigma

        # Read mask video frames
        cap = cv2.VideoCapture(mask_video_path)
        mask_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # (H, W), 0-255
            mask_frames.append(gray.astype(np.float32) / 255.0)  # Normalize to [0, 1]
        cap.release()

        if len(mask_frames) == 0:
            raise ValueError(f"Could not read any frames from mask video: {mask_video_path}")

        print(f"Dual-LoRA mask: Read {len(mask_frames)} frames from mask video")

        # Stack into (T, H, W) array
        mask_np = np.stack(mask_frames, axis=0)  # (T, H, W), values in [0, 1]

        # Apply Gaussian blur per frame for soft borders (before downsampling)
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        for t in range(mask_np.shape[0]):
            mask_np[t] = cv2.GaussianBlur(mask_np[t], (kernel_size, kernel_size), sigma)

        # Compute patch-grid dimensions
        # Latent: f_latent = (num_frames - 1) // 4 + 1, h_latent = height // 8, w_latent = width // 8
        # After patchify with patch_size: f = f_latent // pf, h = h_latent // ph, w = w_latent // pw
        pf, ph, pw = patch_size
        f_latent = (num_frames - 1) // 4 + 1
        h_latent = height // 8
        w_latent = width // 8
        f_grid = f_latent // pf
        h_grid = h_latent // ph
        w_grid = w_latent // pw

        print(f"  Patch grid: ({f_grid}, {h_grid}, {w_grid}), total tokens: {f_grid * h_grid * w_grid}")

        # Downsample the mask to patch-grid resolution using bilinear interpolation
        mask_tensor = torch.from_numpy(mask_np).float()  # (T, H, W)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, T, H, W)
        mask_downsampled = F.interpolate(
            mask_tensor,
            size=(f_grid, h_grid, w_grid),
            mode="trilinear",
            align_corners=False,
        )
        mask_downsampled = mask_downsampled.squeeze(0).squeeze(0)  # (f_grid, h_grid, w_grid)

        # Flatten to token sequence: (f_grid * h_grid * w_grid)
        token_mask = mask_downsampled.reshape(1, -1, 1)  # (1, seq_len, 1)

        # Ensure values are in [0, 1]
        token_mask = token_mask.clamp(0.0, 1.0)

        # Move to device
        self.token_mask = token_mask.to(self.device, self.dtype)

        # Print mask statistics
        fg_ratio = token_mask.mean().item() * 100
        border_tokens = ((token_mask > 0.05) & (token_mask < 0.95)).sum().item()
        print(f"  Foreground coverage: {fg_ratio:.1f}%")
        print(f"  Border tokens (0.05 < mask < 0.95): {border_tokens}")
        print(f"  Token mask shape: {self.token_mask.shape}")

    def remove_hooks(self):
        """Remove all registered hooks from the model."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        self.token_mask = None
        print("Dual-LoRA: All hooks removed")


def get_patch_size_from_model(dit_model: nn.Module) -> Tuple[int, int, int]:
    """
    Extract patch size from the WanModel.
    Falls back to (1, 2, 2) for WAN I2V 14B if not found.
    """
    if hasattr(dit_model, "patch_size"):
        ps = dit_model.patch_size
        if isinstance(ps, (list, tuple)):
            return tuple(ps)
        return (ps, ps, ps)
    # Default for WAN I2V 14B 480P
    return (1, 2, 2)
