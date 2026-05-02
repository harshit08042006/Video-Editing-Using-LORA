"""
Dual-LoRA Inference Script

This script runs video inference with two LoRA weights simultaneously:
  - WAN LoRA (learned foreground motion from the generated WAN video)
  - Original LoRA (learned background appearance from the real source video)

At each transformer layer, the contributions are blended per-token using the
spatial mask: foreground uses WAN LoRA, background uses original LoRA,
and borders are smoothly blended via Gaussian blur.

Usage:
    python inference_dual_lora.py \
        --model_root_dir /path/to/Wan2.1-I2V-14B-480P \
        --wan_data_dir /path/to/processed_data/video_1776166954 \
        --orig_data_dir /path/to/processed_data/video_1776166484 \
        --border_sigma 3.0

This does NOT modify any existing files or weights.
"""

import torch
import os
import glob
import argparse
import re
import random
from PIL import Image
from diffsynth import ModelManager, save_video, VideoData
from custom_wan_pipe import WanVideoPipeline
from dual_lora_utils import DualLoRAManager, get_patch_size_from_model

# Florence model import - required dependency
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel

# Global variables to store Florence model
florence_model = None
florence_processor = None


def init_florence_model():
    """Initialize Florence model, only needs to be called once"""
    global florence_model, florence_processor
        
    if florence_model is not None and florence_processor is not None:
        return True  # Model already loaded, no need to reload
    
    print("Loading Florence model, please wait...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Some Florence checkpoints expect this flag on the base model
    if not hasattr(PreTrainedModel, "_supports_sdpa"):
        setattr(PreTrainedModel, "_supports_sdpa", False)

    # Try multiple model variants in order of preference
    model_variants = [
        "microsoft/Florence-2-large",
        "microsoft/Florence-2-base",
        "microsoft/Florence-2-large-ft",
    ]
    
    loaded = False
    for model_name in model_variants:
        try:
            print(f"Trying to load {model_name}...")
            florence_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                attn_implementation="eager",
            ).to(device)
            
            florence_processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
            print(f"Florence model loaded successfully from {model_name}")
            loaded = True
            break
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue
    
    if not loaded:
        raise RuntimeError("Failed to load any Florence model variant. Please check your transformers library version and internet connection.")
    
    return True


def generate_caption(image, concept_prefix=""):
    """Use Florence model to generate caption for image"""
    global florence_model, florence_processor
    
    if florence_model is None or florence_processor is None:
        raise RuntimeError("Florence model not initialized, please call init_florence_model() first")
    
    device = next(florence_model.parameters()).device
    torch_dtype = next(florence_model.parameters()).dtype
    
    # If input is a path, read image; if PIL Image object, use directly
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif hasattr(image, 'convert'):
        image = image.convert("RGB")
    else:
        # If it's a numpy array, convert to PIL Image
        if hasattr(image, 'shape'):
            image = Image.fromarray(image).convert("RGB")
        else:
            raise ValueError("Unsupported image format")
    
    prompt = "<DETAILED_CAPTION>"

    # Construct input
    inputs = florence_processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    # Florence language model does not return past_key_values reliably, disable cache to avoid None issues
    if hasattr(florence_model, "config"):
        florence_model.config.use_cache = False

    # Generate caption
    generated_ids = florence_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        use_cache=False,
    )
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # Post-processing
    parsed_answer = florence_processor.post_process_generation(
        generated_text, task=prompt, image_size=(image.width, image.height)
    )
    caption_text = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")
    
    # Add concept prefix
    if concept_prefix:
        caption_text = f"{concept_prefix} {caption_text}"

    return caption_text


def find_max_epoch_lora(data_dir, use_additional=False):
    """Find the lora file with maximum epoch"""
    lora_dir_name = "lora_additional" if use_additional else "lora"
    lora_base_dir = os.path.join(data_dir, lora_dir_name)
    if not os.path.exists(lora_base_dir):
        if use_additional:
            raise FileNotFoundError(f"Additional LoRA directory does not exist: {lora_base_dir}\n"
                                  f"Please train additional LoRA first using: python train.py --config {os.path.join(data_dir, 'configs', 'training_additional.toml')}")
        else:
            raise FileNotFoundError(f"LoRA directory does not exist: {lora_base_dir}")
    
    # Find all date directories
    date_dirs = [d for d in os.listdir(lora_base_dir) if os.path.isdir(os.path.join(lora_base_dir, d))]
    if not date_dirs:
        raise FileNotFoundError(f"No training directories found in LoRA directory: {lora_base_dir}")
    
    # Find the latest training directory (sorted by name, usually datetime format)
    latest_date_dir = sorted(date_dirs)[-1]
    date_dir_path = os.path.join(lora_base_dir, latest_date_dir)
    
    # Find epoch directories
    epoch_dirs = []
    for item in os.listdir(date_dir_path):
        item_path = os.path.join(date_dir_path, item)
        if os.path.isdir(item_path) and item.startswith("epoch"):
            # Extract epoch number
            match = re.search(r'epoch(\d+)', item)
            if match:
                epoch_num = int(match.group(1))
                epoch_dirs.append((epoch_num, item_path))
    
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch directories found in {date_dir_path}")
    
    # Find maximum epoch
    max_epoch_num, max_epoch_path = max(epoch_dirs, key=lambda x: x[0])
    lora_file_path = os.path.join(max_epoch_path, "adapter_model.safetensors")
    
    if not os.path.exists(lora_file_path):
        raise FileNotFoundError(f"LoRA file does not exist: {lora_file_path}")
    
    print(f"Found LoRA file with maximum epoch: epoch{max_epoch_num} - {lora_file_path}")
    return lora_file_path


def find_input_image(data_dir):
    """Find edited image, prioritize png, then jpg"""
    png_path = os.path.join(data_dir, "edited_image.png")
    jpg_path = os.path.join(data_dir, "edited_image.jpg")
    
    if os.path.exists(png_path):
        print(f"Found edited image: {png_path}")
        return png_path
    elif os.path.exists(jpg_path):
        print(f"Found edited image: {jpg_path}")
        return jpg_path
    else:
        raise FileNotFoundError(f"Edited image does not exist, checked the following paths:\n- {png_path}\n- {jpg_path}")


def validate_paths(model_root_dir, wan_data_dir, orig_data_dir):
    """Validate if paths exist"""
    if not os.path.exists(model_root_dir):
        raise FileNotFoundError(f"Model root directory does not exist: {model_root_dir}")
    
    if not os.path.exists(wan_data_dir):
        raise FileNotFoundError(f"WAN data directory does not exist: {wan_data_dir}")

    if not os.path.exists(orig_data_dir):
        raise FileNotFoundError(f"Original data directory does not exist: {orig_data_dir}")
    
    # Check required model files
    required_files = [
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "models_t5_umt5-xxl-enc-bf16.pth", 
        "Wan2.1_VAE.pth"
    ]
    
    for file_name in required_files:
        file_path = os.path.join(model_root_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required model file does not exist: {file_path}")
    
    # Check diffusion model files
    diffusion_model_pattern = os.path.join(model_root_dir, "diffusion_pytorch_model*.safetensors")
    diffusion_model_files = glob.glob(diffusion_model_pattern)
    if not diffusion_model_files:
        raise FileNotFoundError(f"No diffusion model files found, pattern: {diffusion_model_pattern}")


def main(model_root_dir, wan_data_dir, orig_data_dir, border_sigma=3.0, output_suffix="_dual_lora"):
    """Main function for dual-LoRA inference"""
    try:
        # Validate paths
        print("=" * 60)
        print("DUAL-LoRA INFERENCE")
        print("=" * 60)
        print(f"  WAN data dir (foreground motion): {wan_data_dir}")
        print(f"  Original data dir (background stability): {orig_data_dir}")
        print(f"  Border sigma: {border_sigma}")
        print()
        
        print("Validating paths...")
        validate_paths(model_root_dir, wan_data_dir, orig_data_dir)
        
        # Find LoRA paths for both
        print("\nFinding LoRA weights...")
        wan_lora_path = find_max_epoch_lora(wan_data_dir)
        orig_lora_path = find_max_epoch_lora(orig_data_dir)
        
        # Find input image and videos from WAN data dir
        input_image_path = find_input_image(wan_data_dir)
        pseudo_video_path = os.path.join(wan_data_dir, "inference_rgb.mp4")
        mask_video_path = os.path.join(wan_data_dir, "inference_mask.mp4")
        
        # Check if video files exist
        if not os.path.exists(pseudo_video_path):
            raise FileNotFoundError(f"Pseudo video file does not exist: {pseudo_video_path}")
        if not os.path.exists(mask_video_path):
            raise FileNotFoundError(f"Mask video file does not exist: {mask_video_path}")
        
        print(f"\nUsing paths:")
        print(f"  Model root directory: {model_root_dir}")
        print(f"  WAN LoRA: {wan_lora_path}")
        print(f"  Original LoRA: {orig_lora_path}")
        print(f"  Edited image: {input_image_path}")
        print(f"  Pseudo video: {pseudo_video_path}")
        print(f"  Mask video: {mask_video_path}")
        
        # Automatically find all safetensors files starting with diffusion_pytorch_model
        diffusion_model_pattern = os.path.join(model_root_dir, "diffusion_pytorch_model*.safetensors")
        diffusion_model_files = sorted(glob.glob(diffusion_model_pattern))

        # ============================================================
        # LOAD BASE MODEL WITHOUT ANY LoRA
        # ============================================================
        print("\nLoading base model (WITHOUT LoRA)...")
        model_manager = ModelManager(device="cpu")
        model_manager.load_models([
            os.path.join(model_root_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ], torch_dtype=torch.float32)
        model_manager.load_models([
            diffusion_model_files,
            os.path.join(model_root_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            os.path.join(model_root_dir, "Wan2.1_VAE.pth"),
        ], torch_dtype=torch.bfloat16)
        
        # NOTE: We intentionally do NOT call model_manager.load_lora() here!
        # The dual-LoRA hooks will handle both LoRAs manually.
        
        pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
        pipe.enable_vram_management(num_persistent_param_in_dit=0)

        # ============================================================
        # SETUP DUAL-LoRA HOOKS
        # ============================================================
        print("\nSetting up dual-LoRA hooks...")
        
        # Get the video dimensions from the pseudo video
        import cv2
        cap = cv2.VideoCapture(pseudo_video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        print(f"  Video: {num_frames} frames, {frame_width}x{frame_height}")

        # Get patch size from the model
        patch_size = get_patch_size_from_model(pipe.dit)
        print(f"  Patch size: {patch_size}")

        # Create the DualLoRAManager (registers hooks on all transformer layers)
        dual_lora = DualLoRAManager(
            dit_model=pipe.dit,
            wan_lora_path=wan_lora_path,
            orig_lora_path=orig_lora_path,
            lora_alpha=1.0,
            border_sigma=border_sigma,
            device="cuda",
            dtype=torch.bfloat16,
        )

        # Set up the token-space mask from the mask video
        dual_lora.set_mask_from_video(
            mask_video_path=mask_video_path,
            num_frames=num_frames,
            height=frame_height,
            width=frame_width,
            patch_size=patch_size,
        )

        # ============================================================
        # RUN INFERENCE (same as original, hooks handle the blending)
        # ============================================================
        # Initialize Florence model
        print("\nInitializing Florence model...")
        init_florence_model()

        # Load edited image
        input_image = Image.open(input_image_path)

        # Read concept prefix from prefix.txt
        prefix_file = os.path.join(wan_data_dir, 'prefix.txt')
        concept_prefix = ""
        if os.path.exists(prefix_file):
            try:
                with open(prefix_file, 'r', encoding='utf-8') as f:
                    concept_prefix = f.read().strip()
                print(f"Read concept prefix from {prefix_file}: {concept_prefix}")
            except Exception as e:
                print(f"Failed to read prefix.txt file: {e}")
                concept_prefix = "p3rs0n,"
        else:
            print(f"prefix.txt file not found: {prefix_file}, using default prefix")
            concept_prefix = "p3rs0n,"

        # Get the visual description from Florence
        print("Analyzing image...")
        visual_description = generate_caption(input_image, concept_prefix=concept_prefix)
        
        # Get action from prompt.txt
        prompt_file = os.path.join(wan_data_dir, 'prompt.txt')
        action_prompt = ""
        
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                action_prompt = f.read().strip()
            print(f"Found action prompt: {action_prompt}")

        # Combine prompts
        if action_prompt:
            final_prompt = f"{action_prompt}"
        else:
            final_prompt = visual_description

        print(f"Final merged prompt sent to model: {final_prompt}")

        print("\n" + "=" * 60)
        print("Starting dual-LoRA inference...")
        print("=" * 60)

        video = pipe(
            prompt=final_prompt,
            input_image=input_image,
            
            input_video=None,   
            pseudo_video_path=pseudo_video_path,
            mask_video_path=mask_video_path,            

            negative_prompt="Overexposure, static, blurred details, subtitles, paintings, pictures, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, mutilated, redundant fingers, poorly painted hands, poorly painted faces, deformed, disfigured, deformed limbs, fused fingers, cluttered background, three legs, a lot of people in the background, upside down",
            
            seed=42,
            num_frames=81,  
            num_inference_steps=30,
            denoising_strength=1.0,
            tea_cache_l1_thresh=None,
            tea_cache_model_id="Wan2.1-I2V-14B-480P",
        )
        
        # ============================================================
        # SAVE OUTPUT
        # ============================================================
        output_path = os.path.join(wan_data_dir, f"edited_video{output_suffix}.mp4")
        save_video(video, output_path, fps=30, quality=5)
        print(f"\nVideo saved to: {output_path}")
        
        # Cleanup hooks
        dual_lora.remove_hooks()
        print("Dual-LoRA inference complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-LoRA video generation inference script")
    parser.add_argument("--model_root_dir", required=True,
                       help="Model root directory path (e.g., Wan2.1-I2V-14B-480P)")
    parser.add_argument("--wan_data_dir", required=True,
                       help="WAN video data directory (e.g., processed_data/video_1776166954)")
    parser.add_argument("--orig_data_dir", required=True,
                       help="Original video data directory (e.g., processed_data/video_1776166484)")
    parser.add_argument("--border_sigma", type=float, default=3.0,
                       help="Gaussian blur sigma for soft mask borders (default: 3.0). "
                            "Higher = wider soft transition zone. Try 2.0-5.0.")
    parser.add_argument("--output_suffix", type=str, default="_dual_lora",
                       help="Suffix for the output video filename (default: '_dual_lora')")
    
    args = parser.parse_args()
    
    main(
        model_root_dir=args.model_root_dir,
        wan_data_dir=args.wan_data_dir,
        orig_data_dir=args.orig_data_dir,
        border_sigma=args.border_sigma,
        output_suffix=args.output_suffix,
    )
