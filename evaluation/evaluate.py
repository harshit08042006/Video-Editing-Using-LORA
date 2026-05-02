"""
Video Editing Evaluation Script
================================
Evaluates the quality of motion-edited videos using 5 metrics:

1. Pose Distance    — How closely the edited video follows the target motion
                      (lower is better, 0 = perfect match)
2. CLIP Similarity  — How well the edited video preserves original appearance
                      (higher is better, 1.0 = identical semantics)
3. SSIM             — How well the background is preserved between original
                      and edited video (higher is better, 1.0 = identical)
4. tLPIPS           — Temporal consistency / flicker detection
                      (lower is better, 0 = perfectly consistent)
5. FID              — Fréchet Inception Distance between frame distributions
                      (lower is better, 0 = identical distributions)

Usage:
    # Just run it — all videos are expected next to this script:
    python evaluate.py

    # Override any path if needed:
    python evaluate.py --original /other/path/original.mp4

Default file layout (same directory as evaluate.py):
    evaluation/
    ├── evaluate.py
    ├── original.mp4
    ├── generated.mp4
    ├── edited.mp4
    ├── inference_mask.mp4   (optional, for background-only SSIM)
    └── eval_results.json    (auto-generated output)
"""

import argparse
import json
import os
import warnings
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Frame extraction utilities
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """Extract frames from a video file as a list of BGR numpy arrays."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"No frames extracted from: {video_path}")

    print(f"  Extracted {len(frames)} frames from {os.path.basename(video_path)} "
          f"({frames[0].shape[1]}x{frames[0].shape[0]})")
    return frames


def resize_frames_to_match(frames: List[np.ndarray],
                           target_h: int, target_w: int) -> List[np.ndarray]:
    """Resize all frames to (target_h, target_w) if they don't already match."""
    resized = []
    for f in frames:
        if f.shape[0] != target_h or f.shape[1] != target_w:
            f = cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        resized.append(f)
    return resized


def align_frame_counts(*frame_lists: List[np.ndarray]) -> Tuple[List[np.ndarray], ...]:
    """Truncate all frame lists to the length of the shortest one."""
    min_len = min(len(fl) for fl in frame_lists)
    return tuple(fl[:min_len] for fl in frame_lists)


# ---------------------------------------------------------------------------
# 1. Pose Distance  (MediaPipe Pose)
# ---------------------------------------------------------------------------

def compute_pose_distance(generated_frames: List[np.ndarray],
                          edited_frames: List[np.ndarray]) -> Dict:
    """
    Compare pose keypoints between the generated (motion reference) video
    and the edited output video using MediaPipe Pose.

    Returns mean normalized L2 distance across all frames & keypoints.
    Lower is better (0 = poses match perfectly).
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError:
        raise ImportError(
            "mediapipe is required for Pose Distance. "
            "Install it with: pip install mediapipe"
        )
    
    import urllib.request
    model_path = os.path.join(os.path.dirname(__file__), "pose_landmarker_heavy.task")
    if not os.path.exists(model_path):
        print("  Downloading MediaPipe Pose Model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
            model_path
        )

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE
    )

    distances = []
    frames_with_both_poses = 0
    frames_with_no_pose = 0

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for i, (gen_frame, edit_frame) in enumerate(
            tqdm(zip(generated_frames, edited_frames),
                 total=len(generated_frames), desc="  Pose Distance")
        ):
            # MediaPipe expects RGB
            gen_rgb = cv2.cvtColor(gen_frame, cv2.COLOR_BGR2RGB)
            edit_rgb = cv2.cvtColor(edit_frame, cv2.COLOR_BGR2RGB)

            gen_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=gen_rgb)
            edit_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=edit_rgb)
            
            gen_result = landmarker.detect(gen_mp_image)
            edit_result = landmarker.detect(edit_mp_image)

            if gen_result.pose_landmarks and edit_result.pose_landmarks:
                frames_with_both_poses += 1
                gen_kps = np.array(
                    [[lm.x, lm.y] for lm in gen_result.pose_landmarks[0]]
                )
                edit_kps = np.array(
                    [[lm.x, lm.y] for lm in edit_result.pose_landmarks[0]]
                )
                # Normalized L2 distance per keypoint, then average
                per_kp_dist = np.linalg.norm(gen_kps - edit_kps, axis=1)
                distances.append(np.mean(per_kp_dist))
            else:
                frames_with_no_pose += 1

    if len(distances) == 0:
        print("  ⚠ No frames had detectable poses in both videos.")
        return {
            "pose_distance_mean": float("nan"),
            "pose_distance_std": float("nan"),
            "frames_with_both_poses": 0,
            "frames_without_pose": frames_with_no_pose,
            "total_frames": len(generated_frames),
        }

    return {
        "pose_distance_mean": float(np.mean(distances)),
        "pose_distance_std": float(np.std(distances)),
        "frames_with_both_poses": frames_with_both_poses,
        "frames_without_pose": frames_with_no_pose,
        "total_frames": len(generated_frames),
    }


# ---------------------------------------------------------------------------
# 2. CLIP Similarity
# ---------------------------------------------------------------------------

def compute_clip_similarity(original_frames: List[np.ndarray],
                            edited_frames: List[np.ndarray],
                            batch_size: int = 16) -> Dict:
    """
    Measure semantic similarity between original and edited frames using
    CLIP (ViT-B/32). Evaluates how well the edited video preserves the
    original appearance / identity.

    Returns mean cosine similarity (higher is better, 1.0 = identical).
    """
    try:
        import clip  # openai clip
    except ImportError:
        try:
            import open_clip
            return _compute_clip_similarity_open_clip(original_frames, edited_frames, batch_size)
        except ImportError:
            raise ImportError(
                "Either 'clip' (pip install git+https://github.com/openai/CLIP.git) "
                "or 'open_clip' (pip install open-clip-torch) is required for CLIP Similarity."
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    similarities = []

    for start in tqdm(range(0, len(original_frames), batch_size),
                      desc="  CLIP Similarity"):
        end = min(start + batch_size, len(original_frames))

        orig_batch = []
        edit_batch = []
        for i in range(start, end):
            orig_pil = Image.fromarray(cv2.cvtColor(original_frames[i], cv2.COLOR_BGR2RGB))
            edit_pil = Image.fromarray(cv2.cvtColor(edited_frames[i], cv2.COLOR_BGR2RGB))
            orig_batch.append(preprocess(orig_pil))
            edit_batch.append(preprocess(edit_pil))

        orig_tensor = torch.stack(orig_batch).to(device)
        edit_tensor = torch.stack(edit_batch).to(device)

        with torch.no_grad():
            orig_features = model.encode_image(orig_tensor)
            edit_features = model.encode_image(edit_tensor)

            # Normalize
            orig_features = orig_features / orig_features.norm(dim=-1, keepdim=True)
            edit_features = edit_features / edit_features.norm(dim=-1, keepdim=True)

            # Cosine similarity per pair
            cos_sim = (orig_features * edit_features).sum(dim=-1)
            similarities.extend(cos_sim.cpu().numpy().tolist())

    return {
        "clip_similarity_mean": float(np.mean(similarities)),
        "clip_similarity_std": float(np.std(similarities)),
        "clip_similarity_min": float(np.min(similarities)),
        "clip_similarity_max": float(np.max(similarities)),
        "total_frames": len(similarities),
    }


def _compute_clip_similarity_open_clip(original_frames: List[np.ndarray],
                                       edited_frames: List[np.ndarray],
                                       batch_size: int = 16) -> Dict:
    """Fallback CLIP similarity using open_clip."""
    import open_clip
    from torchvision import transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()

    similarities = []

    for start in tqdm(range(0, len(original_frames), batch_size),
                      desc="  CLIP Similarity (open_clip)"):
        end = min(start + batch_size, len(original_frames))

        orig_batch = []
        edit_batch = []
        for i in range(start, end):
            orig_pil = Image.fromarray(cv2.cvtColor(original_frames[i], cv2.COLOR_BGR2RGB))
            edit_pil = Image.fromarray(cv2.cvtColor(edited_frames[i], cv2.COLOR_BGR2RGB))
            orig_batch.append(preprocess(orig_pil))
            edit_batch.append(preprocess(edit_pil))

        orig_tensor = torch.stack(orig_batch).to(device)
        edit_tensor = torch.stack(edit_batch).to(device)

        with torch.no_grad():
            orig_features = model.encode_image(orig_tensor)
            edit_features = model.encode_image(edit_tensor)

            orig_features = orig_features / orig_features.norm(dim=-1, keepdim=True)
            edit_features = edit_features / edit_features.norm(dim=-1, keepdim=True)

            cos_sim = (orig_features * edit_features).sum(dim=-1)
            similarities.extend(cos_sim.cpu().numpy().tolist())

    return {
        "clip_similarity_mean": float(np.mean(similarities)),
        "clip_similarity_std": float(np.std(similarities)),
        "clip_similarity_min": float(np.min(similarities)),
        "clip_similarity_max": float(np.max(similarities)),
        "total_frames": len(similarities),
    }


# ---------------------------------------------------------------------------
# 3. SSIM (Background preservation)
# ---------------------------------------------------------------------------

def compute_ssim(original_frames: List[np.ndarray],
                 edited_frames: List[np.ndarray],
                 mask_frames: Optional[List[np.ndarray]] = None) -> Dict:
    """
    Compute Structural Similarity Index between original and edited frames.

    If mask_frames is provided, SSIM is computed ONLY on the background region
    (where mask is black/zero), giving a focused measure of how well the
    background is preserved despite the foreground edit.

    Returns mean SSIM (higher is better, 1.0 = identical structure).
    """
    ssim_scores = []

    for i in tqdm(range(len(original_frames)), desc="  SSIM"):
        orig_gray = cv2.cvtColor(original_frames[i], cv2.COLOR_BGR2GRAY)
        edit_gray = cv2.cvtColor(edited_frames[i], cv2.COLOR_BGR2GRAY)

        if mask_frames is not None:
            # Create binary background mask (invert: background=1, foreground=0)
            mask = mask_frames[i]
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # Threshold to binary (foreground is white in mask)
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            bg_mask = (mask_bin == 0).astype(np.uint8)  # background = 1

            # If background is too small, skip this frame
            if bg_mask.sum() < 100:
                continue

            # Zero out the foreground region in both images
            orig_masked = orig_gray * bg_mask
            edit_masked = edit_gray * bg_mask

            # Compute SSIM with full=True to get the per-pixel map
            score, ssim_map = ssim_fn(
                orig_masked, edit_masked,
                data_range=255, full=True
            )
            # Compute mean SSIM only over background pixels
            ssim_bg = ssim_map[bg_mask == 1].mean()
            ssim_scores.append(float(ssim_bg))
        else:
            score = ssim_fn(orig_gray, edit_gray, data_range=255)
            ssim_scores.append(float(score))

    if len(ssim_scores) == 0:
        return {
            "ssim_mean": float("nan"),
            "ssim_std": float("nan"),
            "total_frames": 0,
            "note": "No valid frames for SSIM computation",
        }

    return {
        "ssim_mean": float(np.mean(ssim_scores)),
        "ssim_std": float(np.std(ssim_scores)),
        "ssim_min": float(np.min(ssim_scores)),
        "ssim_max": float(np.max(ssim_scores)),
        "total_frames": len(ssim_scores),
        "mask_used": mask_frames is not None,
    }


# ---------------------------------------------------------------------------
# 4. tLPIPS (Temporal LPIPS — temporal consistency / flicker)
# ---------------------------------------------------------------------------

def compute_tlpips(original_frames: List[np.ndarray],
                   edited_frames: List[np.ndarray]) -> Dict:
    """
    Compute temporal LPIPS (tLPIPS).

    tLPIPS = mean | LPIPS(edit_t, edit_{t-1}) - LPIPS(orig_t, orig_{t-1}) |

    This measures whether the edited video preserves the same temporal
    dynamics as the original. Large values indicate flickering or unnatural
    motion in the edit.

    Lower is better (0 = perfectly matching temporal dynamics).
    """
    try:
        import lpips
    except ImportError:
        raise ImportError(
            "lpips is required for tLPIPS. Install it with: pip install lpips"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = lpips.LPIPS(net="alex").to(device)
    loss_fn.eval()

    def frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
        """Convert BGR uint8 frame to normalized [-1, 1] tensor."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        t = t * 2.0 - 1.0  # normalize to [-1, 1]
        return t.unsqueeze(0)

    # Ensure consistent resolution for LPIPS (resize to 256x256 for speed)
    def prepare_frame(frame: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
        return frame_to_tensor(resized).to(device)

    temporal_diffs = []

    for i in tqdm(range(1, len(original_frames)), desc="  tLPIPS"):
        with torch.no_grad():
            # Original consecutive pair
            orig_prev = prepare_frame(original_frames[i - 1])
            orig_curr = prepare_frame(original_frames[i])
            lpips_orig = loss_fn(orig_prev, orig_curr).item()

            # Edited consecutive pair
            edit_prev = prepare_frame(edited_frames[i - 1])
            edit_curr = prepare_frame(edited_frames[i])
            lpips_edit = loss_fn(edit_prev, edit_curr).item()

            temporal_diffs.append(abs(lpips_edit - lpips_orig))

    if len(temporal_diffs) == 0:
        return {
            "tlpips_mean": float("nan"),
            "tlpips_std": float("nan"),
            "total_frame_pairs": 0,
        }

    return {
        "tlpips_mean": float(np.mean(temporal_diffs)),
        "tlpips_std": float(np.std(temporal_diffs)),
        "tlpips_min": float(np.min(temporal_diffs)),
        "tlpips_max": float(np.max(temporal_diffs)),
        "total_frame_pairs": len(temporal_diffs),
    }

# ---------------------------------------------------------------------------
# 5. FID (Fréchet Inception Distance)
# ---------------------------------------------------------------------------

def compute_fid(original_frames: List[np.ndarray],
                edited_frames: List[np.ndarray],
                batch_size: int = 16) -> Dict:
    """
    Compute Fréchet Inception Distance (FID) between original and edited
    video frames.

    Treats each frame as a sample in the distribution and computes FID
    using InceptionV3 features. This is the standard single-pair adaptation
    of FVD used in video editing papers — it measures overall visual quality
    drift between the original and edited videos.

    Lower is better (0 = distributions are identical).
    """
    from torchvision import models, transforms
    from scipy import linalg

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load InceptionV3 and strip the final classification layer
    inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT,
                                    transform_input=False)
    inception.fc = torch.nn.Identity()  # output is now 2048-d features
    inception = inception.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def extract_features(frames: List[np.ndarray]) -> np.ndarray:
        """Extract InceptionV3 features for all frames."""
        all_features = []
        for start in range(0, len(frames), batch_size):
            end = min(start + batch_size, len(frames))
            batch = []
            for i in range(start, end):
                pil_img = Image.fromarray(
                    cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
                )
                batch.append(preprocess(pil_img))
            batch_tensor = torch.stack(batch).to(device)
            with torch.no_grad():
                feats = inception(batch_tensor)
            all_features.append(feats.cpu().numpy())
        return np.concatenate(all_features, axis=0)

    print("    Extracting original frame features...")
    orig_feats = extract_features(original_frames)
    print("    Extracting edited frame features...")
    edit_feats = extract_features(edited_frames)

    # Compute mean and covariance for both sets
    mu_orig = np.mean(orig_feats, axis=0)
    sigma_orig = np.cov(orig_feats, rowvar=False)
    mu_edit = np.mean(edit_feats, axis=0)
    sigma_edit = np.cov(edit_feats, rowvar=False)

    # Fréchet distance: ||mu1 - mu2||^2 + Tr(S1 + S2 - 2*sqrt(S1 @ S2))
    diff = mu_orig - mu_edit
    covmean, _ = linalg.sqrtm(sigma_orig @ sigma_edit, disp=False)

    # Numerical stability: discard imaginary components
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_score = float(
        diff @ diff + np.trace(sigma_orig + sigma_edit - 2.0 * covmean)
    )

    return {
        "fid_score": fid_score,
        "num_original_frames": len(original_frames),
        "num_edited_frames": len(edited_frames),
        "feature_dim": int(orig_feats.shape[1]),
    }


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(
    original_path: str,
    generated_path: str,
    edited_path: str,
    mask_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    output_json: Optional[str] = None,
) -> Dict:
    """
    Run the full evaluation pipeline on three input videos.

    Args:
        original_path:  Path to the original (unedited) video.
        generated_path: Path to the generated motion-reference video.
        edited_path:    Path to the final edited output video.
        mask_path:      Optional path to mask video (for background SSIM).
        max_frames:     Optional maximum number of frames to evaluate.
        output_json:    Optional path to save results as JSON.

    Returns:
        Dictionary containing all metric results.
    """
    print("=" * 70)
    print("  Video Editing Evaluation")
    print("=" * 70)
    print(f"  Original  : {original_path}")
    print(f"  Generated : {generated_path}")
    print(f"  Edited    : {edited_path}")
    if mask_path:
        print(f"  Mask      : {mask_path}")
    print("=" * 70)

    # --- Extract frames ---
    print("\n[1/6] Extracting frames...")
    original_frames = extract_frames(original_path, max_frames)
    generated_frames = extract_frames(generated_path, max_frames)
    edited_frames = extract_frames(edited_path, max_frames)

    mask_frames = None
    if mask_path:
        mask_frames = extract_frames(mask_path, max_frames)

    # --- Align frame counts ---
    print("\n[2/6] Aligning frame counts...")
    if mask_frames is not None:
        original_frames, generated_frames, edited_frames, mask_frames = \
            align_frame_counts(original_frames, generated_frames, edited_frames, mask_frames)
    else:
        original_frames, generated_frames, edited_frames = \
            align_frame_counts(original_frames, generated_frames, edited_frames)

    num_frames = len(original_frames)
    print(f"  Using {num_frames} frames for evaluation.")

    # --- Resize to match ---
    target_h, target_w = edited_frames[0].shape[:2]
    original_frames = resize_frames_to_match(original_frames, target_h, target_w)
    generated_frames = resize_frames_to_match(generated_frames, target_h, target_w)
    if mask_frames is not None:
        mask_frames = resize_frames_to_match(mask_frames, target_h, target_w)

    results = {"num_frames_evaluated": num_frames}

    # --- Metric 1: Pose Distance ---
    print("\n[3/6] Computing Pose Distance (generated ↔ edited)...")
    try:
        results["pose_distance"] = compute_pose_distance(generated_frames, edited_frames)
        pd = results["pose_distance"]["pose_distance_mean"]
        print(f"  ✓ Pose Distance: {pd:.6f} "
              f"(detected in {results['pose_distance']['frames_with_both_poses']}/{num_frames} frames)")
    except ImportError as e:
        print(f"  ⚠ Skipped — {e}")
        results["pose_distance"] = {"error": str(e)}
    except Exception as e:
        print(f"  ✗ Error — {e}")
        results["pose_distance"] = {"error": str(e)}

    # --- Metric 2: CLIP Similarity ---
    print("\n[4/6a] Computing CLIP Similarity (original ↔ edited)...")
    try:
        results["clip_similarity"] = compute_clip_similarity(original_frames, edited_frames)
        cs = results["clip_similarity"]["clip_similarity_mean"]
        print(f"  ✓ CLIP Similarity: {cs:.4f}")
    except ImportError as e:
        print(f"  ⚠ Skipped — {e}")
        results["clip_similarity"] = {"error": str(e)}
    except Exception as e:
        print(f"  ✗ Error — {e}")
        results["clip_similarity"] = {"error": str(e)}

    # --- Metric 3: SSIM ---
    print("\n[4/6b] Computing SSIM (original ↔ edited, background focus)...")
    try:
        results["ssim"] = compute_ssim(original_frames, edited_frames, mask_frames)
        s = results["ssim"]["ssim_mean"]
        mask_note = " (background only)" if mask_frames is not None else " (full frame)"
        print(f"  ✓ SSIM: {s:.4f}{mask_note}")
    except Exception as e:
        print(f"  ✗ Error — {e}")
        results["ssim"] = {"error": str(e)}

    # --- Metric 4: tLPIPS ---
    print("\n[5/6] Computing tLPIPS (temporal consistency)...")
    try:
        results["tlpips"] = compute_tlpips(original_frames, edited_frames)
        tl = results["tlpips"]["tlpips_mean"]
        print(f"  ✓ tLPIPS: {tl:.6f}")
    except ImportError as e:
        print(f"  ⚠ Skipped — {e}")
        results["tlpips"] = {"error": str(e)}
    except Exception as e:
        print(f"  ✗ Error — {e}")
        results["tlpips"] = {"error": str(e)}

    # --- Metric 5: FID ---
    print("\n[6/6] Computing FID (original ↔ edited)...")
    try:
        results["fid"] = compute_fid(original_frames, edited_frames)
        fid_val = results["fid"]["fid_score"]
        print(f"  ✓ FID: {fid_val:.4f}")
    except ImportError as e:
        print(f"  ⚠ Skipped — {e}")
        results["fid"] = {"error": str(e)}
    except Exception as e:
        print(f"  ✗ Error — {e}")
        results["fid"] = {"error": str(e)}

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  {'Metric':<25} {'Value':<15} {'Interpretation'}")
    print(f"  {'-'*25} {'-'*15} {'-'*30}")

    if "error" not in results.get("pose_distance", {}):
        pd_val = results["pose_distance"]["pose_distance_mean"]
        pd_interp = "Excellent" if pd_val < 0.02 else "Good" if pd_val < 0.05 else "Fair" if pd_val < 0.10 else "Poor"
        print(f"  {'Pose Distance':<25} {pd_val:<15.6f} {pd_interp} (↓ lower is better)")

    if "error" not in results.get("clip_similarity", {}):
        cs_val = results["clip_similarity"]["clip_similarity_mean"]
        cs_interp = "Excellent" if cs_val > 0.95 else "Good" if cs_val > 0.90 else "Fair" if cs_val > 0.80 else "Poor"
        print(f"  {'CLIP Similarity':<25} {cs_val:<15.4f} {cs_interp} (↑ higher is better)")

    if "error" not in results.get("ssim", {}):
        s_val = results["ssim"]["ssim_mean"]
        s_interp = "Excellent" if s_val > 0.95 else "Good" if s_val > 0.90 else "Fair" if s_val > 0.80 else "Poor"
        print(f"  {'SSIM':<25} {s_val:<15.4f} {s_interp} (↑ higher is better)")

    if "error" not in results.get("tlpips", {}):
        tl_val = results["tlpips"]["tlpips_mean"]
        tl_interp = "Excellent" if tl_val < 0.01 else "Good" if tl_val < 0.03 else "Fair" if tl_val < 0.06 else "Poor"
        print(f"  {'tLPIPS':<25} {tl_val:<15.6f} {tl_interp} (↓ lower is better)")

    if "error" not in results.get("fid", {}):
        fid_v = results["fid"]["fid_score"]
        fid_interp = "Excellent" if fid_v < 10 else "Good" if fid_v < 30 else "Fair" if fid_v < 60 else "Poor"
        print(f"  {'FID':<25} {fid_v:<15.4f} {fid_interp} (↓ lower is better)")

    print("=" * 70)

    # --- Save to JSON ---
    if output_json:
        os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else ".", exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {output_json}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    # Resolve the directory where this script lives (for default paths)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Default paths — all relative to the script's own directory
    DEFAULT_ORIGINAL  = os.path.join(SCRIPT_DIR, "original.mp4")
    DEFAULT_GENERATED = os.path.join(SCRIPT_DIR, "generated.mp4")
    DEFAULT_EDITED    = os.path.join(SCRIPT_DIR, "edited.mp4")
    DEFAULT_MASK      = os.path.join(SCRIPT_DIR, "inference_mask.mp4")
    DEFAULT_OUTPUT    = os.path.join(SCRIPT_DIR, "eval_results.json")

    # Auto-detect mask: use it if the file exists, otherwise None
    default_mask = DEFAULT_MASK if os.path.exists(DEFAULT_MASK) else None

    parser = argparse.ArgumentParser(
        description="Evaluate quality of motion-edited videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Default usage (no arguments needed — videos sit next to this script):
    python evaluate.py

Override any path:
    python evaluate.py --original /other/original.mp4

Metrics:
    Pose Distance   — generated ↔ edited  (motion fidelity, ↓ better)
    CLIP Similarity — original  ↔ edited  (appearance preservation, ↑ better)
    SSIM            — original  ↔ edited  (background preservation, ↑ better)
    tLPIPS          — original  ↔ edited  (temporal consistency, ↓ better)
    FID             — original  ↔ edited  (visual quality drift, ↓ better)
        """
    )

    parser.add_argument("--original", default=DEFAULT_ORIGINAL,
                        help=f"Path to the original (unedited) video (default: {DEFAULT_ORIGINAL})")
    parser.add_argument("--generated", default=DEFAULT_GENERATED,
                        help=f"Path to the generated motion-reference video (default: {DEFAULT_GENERATED})")
    parser.add_argument("--edited", default=DEFAULT_EDITED,
                        help=f"Path to the final edited output video (default: {DEFAULT_EDITED})")
    parser.add_argument("--mask", default=default_mask,
                        help=f"Path to mask video for background-only SSIM (default: {DEFAULT_MASK} if it exists)")
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Maximum number of frames to evaluate (default: all)")
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT,
                        help=f"Path to save evaluation results as JSON (default: {DEFAULT_OUTPUT})")

    args = parser.parse_args()

    evaluate(
        original_path=args.original,
        generated_path=args.generated,
        edited_path=args.edited,
        mask_path=args.mask,
        max_frames=args.num_frames,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
