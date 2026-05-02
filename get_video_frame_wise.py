import cv2
import os
from pathlib import Path

# Video path
video_path = "/home/harshit23236/LoRAEdit/processed_data/video_1776655911/inference_rgb.mp4"

# Create output directory
video_dir = os.path.dirname(video_path)
output_dir = os.path.join(video_dir, "edited_video_frames")
os.makedirs(output_dir, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Save frame with zero-padded naming
    frame_name = f"{frame_count:05d}.png"
    frame_path = os.path.join(output_dir, frame_name)
    cv2.imwrite(frame_path, frame)
    
    print(f"Saved {frame_name}")
    frame_count += 1

cap.release()
print(f"\nExtraction complete! {frame_count} frames saved to {output_dir}")