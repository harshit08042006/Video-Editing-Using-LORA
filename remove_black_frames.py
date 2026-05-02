"""
Remove first black frame from video
"""

import cv2
import numpy as np
from PIL import Image
import os

def remove_first_black_frame(input_video, output_video):
    """
    Remove the first black frame from a video.
    
    Args:
        input_video: Path to input video
        output_video: Path to output video
    """
    print(f"Loading video: {input_video}")
    
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {frame_count} frames, {width}x{height}, {fps} fps")
    
    # Read all frames
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if frame is mostly black (first frame)
        if frame_idx == 0:
            # Convert to grayscale and check mean intensity
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray)
            print(f"Frame {frame_idx}: mean intensity = {mean_intensity:.2f}")
            
            if mean_intensity < 10:  # Very dark = black frame
                print(f"Frame {frame_idx} is BLACK, SKIPPING...")
                frame_idx += 1
                continue
        
        frames.append(frame)
        frame_idx += 1
    
    cap.release()
    
    print(f"Kept {len(frames)} frames (removed {frame_count - len(frames)} black frames)")
    
    if not frames:
        print("ERROR: All frames were black!")
        return
    
    # Write output video
    print(f"Writing output video: {output_video}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for i, frame in enumerate(frames):
        if (i + 1) % max(1, len(frames) // 10) == 0:
            print(f"  Writing frame {i+1}/{len(frames)}")
        out.write(frame)
    
    out.release()
    print(f"✓ Done! Saved to: {output_video}")


if __name__ == "__main__":
    input_video = "/home/harshit23236/LoRAEdit/processed_data/video_1776169719/inference_rgb.mp4"
    output_video = "/home/harshit23236/LoRAEdit/processed_data/video_1776169719/inference_rgb_fixed.mp4"
    
    if not os.path.exists(input_video):
        print(f"ERROR: Input video not found: {input_video}")
        exit(1)
    
    remove_first_black_frame(input_video, output_video)
