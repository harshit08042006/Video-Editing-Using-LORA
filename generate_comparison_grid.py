import os
import matplotlib.pyplot as plt
from PIL import Image

# frames required
frames = [1,5,10,15,20,25,30,35,40,45,49]

# change this for each video
video_folder = "processed_data/video_1773336792"

paths = {
    "Ground Truth": os.path.join(video_folder, "source_frames_original_video"),
    "Wan/VEO": os.path.join(video_folder, "wan_frames"),
    "LoRA Edit": os.path.join(video_folder, "original_edited_video_frames"),
    "Our Results": os.path.join(video_folder, "edited_video_frames")
}

rows = list(paths.keys())

fig, axes = plt.subplots(len(rows), len(frames), figsize=(22,8))

for r, method in enumerate(rows):
    folder = paths[method]

    for c, frame in enumerate(frames):

        frame_name = f"{frame:05d}.png"
        img_path = os.path.join(folder, frame_name)

        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[r,c].imshow(img)
        else:
            axes[r,c].text(0.5,0.5,"Missing",ha='center')

        axes[r,c].axis("off")

        if r == 0:
            axes[r,c].set_title(f"Frame {frame}", fontsize=10)

    axes[r,0].set_ylabel(method, fontsize=12)

plt.tight_layout()

save_path = os.path.join(video_folder, "comparison_grid.png")
plt.savefig(save_path, dpi=300)

print("Saved to:", save_path)