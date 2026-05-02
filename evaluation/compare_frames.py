"""
Video Frame Comparison Grid
============================
Extracts 6 evenly-spaced frames from the original, generated, and edited
videos and arranges them in a side-by-side comparison grid.

Output: comparison_grid.png (saved next to this script)

Usage:
    python compare_frames.py
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_ORIGINAL  = os.path.join(SCRIPT_DIR, "original.mp4")
DEFAULT_GENERATED = os.path.join(SCRIPT_DIR, "generated.mp4")
DEFAULT_EDITED    = os.path.join(SCRIPT_DIR, "edited.mp4")
OUTPUT_PATH       = os.path.join(SCRIPT_DIR, "comparison_grid.png")

NUM_FRAMES = 6


def extract_frames(video_path: str) -> list:
    """Extract all frames from a video."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames in {video_path}")
    return frames


def pick_frames(frames: list, n: int = 6) -> list:
    """Pick n evenly-spaced frames."""
    total = len(frames)
    if total <= n:
        return frames
    indices = np.linspace(0, total - 1, n, dtype=int)
    return [frames[i] for i in indices]


def frames_to_pil(frames: list, target_h: int, target_w: int) -> list:
    """Resize BGR frames and convert to PIL RGB."""
    pil_frames = []
    for f in frames:
        f = cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        pil_frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    return pil_frames


def get_font(size: int):
    """Try to load a nice font, fall back to default."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
    ]
    for p in font_paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def build_grid(original_path: str, generated_path: str, edited_path: str,
               output_path: str, num_frames: int = 6):
    """Build and save the comparison grid image."""

    print(f"Loading videos...")
    orig_all  = extract_frames(original_path)
    gen_all   = extract_frames(generated_path)
    edit_all  = extract_frames(edited_path)

    print(f"  Original : {len(orig_all)} frames")
    print(f"  Generated: {len(gen_all)} frames")
    print(f"  Edited   : {len(edit_all)} frames")

    orig_picked = pick_frames(orig_all, num_frames)
    gen_picked  = pick_frames(gen_all, num_frames)
    edit_picked = pick_frames(edit_all, num_frames)

    # Use a common resolution (reasonable display size)
    target_h, target_w = 360, 640
    orig_pil = frames_to_pil(orig_picked, target_h, target_w)
    gen_pil  = frames_to_pil(gen_picked, target_h, target_w)
    edit_pil = frames_to_pil(edit_picked, target_h, target_w)

    # --- Layout ---
    # 3 rows (Original / Generated / Edited) x num_frames columns
    # + row labels on the left + column headers on top

    label_w   = 280          # width of left label column
    pad       = 4            # padding between cells
    cell_w    = target_w
    cell_h    = target_h

    grid_w = label_w + num_frames * (cell_w + pad) + pad
    grid_h = 3 * (cell_h + pad) + pad

    canvas = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw   = ImageDraw.Draw(canvas)
    font_label  = get_font(48)

    rows = [
        ("Original",  orig_pil,  (30, 80, 140)),
        ("Generated", gen_pil,   (140, 90, 20)),
        ("Edited",    edit_pil,  (20, 120, 50)),
    ]

    # Draw rows
    for row_idx, (label, pil_frames, color) in enumerate(rows):
        y_top = pad + row_idx * (cell_h + pad)

        # Row label
        lx = label_w // 2
        ly = y_top + cell_h // 2
        draw.text((lx, ly), label, fill=color, font=font_label, anchor="mm")

        # Paste frames
        for col, frame_img in enumerate(pil_frames):
            x_left = label_w + pad + col * (cell_w + pad)
            canvas.paste(frame_img, (x_left, y_top))

    canvas.save(output_path, quality=95)
    print(f"\n✓ Comparison grid saved to: {output_path}")
    print(f"  Grid size: {grid_w} x {grid_h} px")


if __name__ == "__main__":
    build_grid(DEFAULT_ORIGINAL, DEFAULT_GENERATED, DEFAULT_EDITED,
               OUTPUT_PATH, NUM_FRAMES)
