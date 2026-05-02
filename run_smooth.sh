#!/bin/bash
# Border smoothing script

python border_smoothing_inference.py \
  --edited_video /home/harshit23236/LoRAEdit/processed_data/video_1776166954/edited_video.mp4 \
  --original_video /home/harshit23236/LoRAEdit/processed_data/video_1776166484/source_frames \
  --mask_video /home/harshit23236/LoRAEdit/processed_data/video_1776166954/inference_mask.mp4 \
  --output output/smooth.mp4 \
  --blend_strength 0.6
