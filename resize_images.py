#!/usr/bin/env python3
# ...existing code...
import os
import subprocess
import sys
import tempfile
import shutil
from fractions import Fraction

input_vid = "/home/harshit23236/LoRAEdit/processed_data/video_1772431838/inference_rgb.mp4"
first_frame = "/home/harshit23236/LoRAEdit/processed_data/video_1772431838/source_frames/00000.png"
tmp_out = input_vid + ".tmp.mp4"

if not os.path.exists(input_vid):
    print("input video not found:", input_vid); sys.exit(1)
if not os.path.exists(first_frame):
    print("frame image not found:", first_frame); sys.exit(1)

def ffprobe_video_props(path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    out = subprocess.check_output(cmd).decode().strip().splitlines()
    if len(out) < 3:
        raise RuntimeError("ffprobe failed to get video properties")
    width = out[0].strip()
    height = out[1].strip()
    r_frame_rate = out[2].strip()
    try:
        fps = str(float(Fraction(r_frame_rate)))
    except Exception:
        fps = r_frame_rate
    return int(width), int(height), fps

tmpdir = tempfile.mkdtemp(prefix="replace_frame_")
try:
    width, height, fps = ffprobe_video_props(input_vid)

    # extract frames starting at 00000.png
    cmd_extract = [
        "ffmpeg", "-y",
        "-i", input_vid,
        "-start_number", "0",
        os.path.join(tmpdir, "%05d.png")
    ]
    subprocess.run(cmd_extract, check=True)

    # scale/pad provided first frame to match video resolution and write as 00000.png
    scaled_first = os.path.join(tmpdir, "00000.png")
    cmd_scale = [
        "ffmpeg", "-y",
        "-i", first_frame,
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        scaled_first
    ]
    subprocess.run(cmd_scale, check=True)

    # encode frames back to video (no audio)
    cmd_encode = [
        "ffmpeg", "-y",
        "-framerate", fps,
        "-start_number", "0",
        "-i", os.path.join(tmpdir, "%05d.png"),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        tmp_out
    ]
    subprocess.run(cmd_encode, check=True)

    os.replace(tmp_out, input_vid)
    print("Replaced first frame and saved to:", input_vid)
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
# ...existing code...