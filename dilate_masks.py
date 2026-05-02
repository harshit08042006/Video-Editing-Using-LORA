
import cv2
import numpy as np
import os
from glob import glob
from PIL import Image
mask_dir = './processed_data/video_1771562690/source_masks/'
rgb_dir = './processed_data/video_1771562690/source_frames/'
traindata_dir = './processed_data/video_1771562690/traindata/'
output_dir = './processed_data/video_1771562690/'
MASK_THRESHOLD = 34936 # 10% of 832x480

def dilate_mask(mask, kernel_size=100):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


mask_paths = sorted(glob(os.path.join(mask_dir, '*.png')))
rgb_paths = sorted(glob(os.path.join(rgb_dir, '*.png')))

# Utility: apply grayscale to mask region (like predata_app.py)
def apply_grayscale_to_mask_region(rgb_path, mask_array):
    input_img = Image.open(rgb_path).convert('RGB')
    input_array = np.array(input_img)
    output_array = input_array.copy()
    white_pixels = mask_array > 200
    output_array[white_pixels] = 128
    return output_array

# Step 1: Dilate masks if small
for mask_path in mask_paths:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue
    mask_area = np.count_nonzero(mask)
    if mask_area < MASK_THRESHOLD:
        mask = dilate_mask(mask)
        cv2.imwrite(mask_path, mask)
        new_area = np.count_nonzero(mask)
        print(f"-> dilated, new area: {new_area}")
    else:
        print(f"-> not dilated, area: {mask_area}")

# Step 2: Regenerate inference_mask.mp4 and inference_rgb.mp4
import subprocess
import shutil

temp_mask_dir = os.path.join(output_dir, 'temp_mask')
temp_rgb_dir = os.path.join(output_dir, 'temp_rgb')
os.makedirs(temp_mask_dir, exist_ok=True)
os.makedirs(temp_rgb_dir, exist_ok=True)


# Save updated masks and grayed RGB frames
for i, (mask_path, rgb_path) in enumerate(zip(mask_paths, rgb_paths)):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(os.path.join(temp_mask_dir, f'frame_{i:04d}.png'), mask)
    grayed_rgb = apply_grayscale_to_mask_region(rgb_path, mask)
    cv2.imwrite(os.path.join(temp_rgb_dir, f'frame_{i:04d}.png'), cv2.cvtColor(grayed_rgb, cv2.COLOR_RGB2BGR))

# Generate inference_mask.mp4
mask_video_path = os.path.join(output_dir, 'inference_mask.mp4')
mask_ffmpeg_cmd = [
    'ffmpeg', '-y', '-framerate', '25', '-i', os.path.join(temp_mask_dir, 'frame_%04d.png'),
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '11', mask_video_path
]
subprocess.run(mask_ffmpeg_cmd, check=True)

# Generate inference_rgb.mp4
rgb_video_path = os.path.join(output_dir, 'inference_rgb.mp4')
rgb_ffmpeg_cmd = [
    'ffmpeg', '-y', '-framerate', '25', '-i', os.path.join(temp_rgb_dir, 'frame_%04d.png'),
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '11', rgb_video_path
]
subprocess.run(rgb_ffmpeg_cmd, check=True)


# Step 3: Regenerate sequence_all_frames_49.mp4 in traindata (original logic)
import imageio.v2 as iio

temp_seq_dir = os.path.join(output_dir, 'temp_sequence')
os.makedirs(temp_seq_dir, exist_ok=True)

# Read all RGB frames and masks
rgb_arrays = [np.array(Image.open(p).convert('RGB')) for p in rgb_paths]
mask_arrays = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in mask_paths]

frame_index = 0
# Stage 0: Place first frame separately
first_image = rgb_arrays[0]
iio.imwrite(os.path.join(temp_seq_dir, f'frame_{frame_index:03d}.jpg'), first_image)
frame_index += 1

# Stage 1: Copy first frame, then apply gray processing to all other frames
iio.imwrite(os.path.join(temp_seq_dir, f'frame_{frame_index:03d}.jpg'), first_image)
frame_index += 1
for i, (img, mask) in enumerate(zip(rgb_arrays[1:], mask_arrays[1:]), 1):
    grayed_img = apply_grayscale_to_mask_region(rgb_paths[i], mask)
    iio.imwrite(os.path.join(temp_seq_dir, f'frame_{frame_index:03d}.jpg'), grayed_img)
    frame_index += 1

# Stage 2: Directly copy all original frames
for img in rgb_arrays:
    iio.imwrite(os.path.join(temp_seq_dir, f'frame_{frame_index:03d}.jpg'), img)
    frame_index += 1

# Stage 3: Add mask frames (first frame is all black, other frames use corresponding mask)
black_mask = np.zeros_like(mask_arrays[0])
iio.imwrite(os.path.join(temp_seq_dir, f'frame_{frame_index:03d}.jpg'), black_mask)
frame_index += 1
for mask in mask_arrays[1:]:
    iio.imwrite(os.path.join(temp_seq_dir, f'frame_{frame_index:03d}.jpg'), mask)
    frame_index += 1

# Generate training video
sequence_video_path = os.path.join(traindata_dir, f'sequence_all_frames_{len(rgb_arrays)}.mp4')
sequence_ffmpeg_cmd = [
    'ffmpeg', '-y', '-framerate', '5', '-i', os.path.join(temp_seq_dir, 'frame_%03d.jpg'),
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '11', sequence_video_path
]
subprocess.run(sequence_ffmpeg_cmd, check=True)

# Clean up temporary directories
shutil.rmtree(temp_mask_dir)
shutil.rmtree(temp_rgb_dir)
shutil.rmtree(temp_seq_dir)