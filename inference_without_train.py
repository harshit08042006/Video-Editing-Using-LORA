import torch
import os
import glob
import argparse
import random
from PIL import Image
from diffsynth import ModelManager, save_video, VideoData
from custom_wan_pipe import WanVideoPipeline

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

def validate_paths(model_root_dir, data_dir):
    """Validate if paths exist"""
    if not os.path.exists(model_root_dir):
        raise FileNotFoundError(f"Model root directory does not exist: {model_root_dir}")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    
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

def main(model_root_dir, data_dir):
    """Main function"""
    try:
        # Validate paths
        print("Validating paths...")
        validate_paths(model_root_dir, data_dir)
        
        # Infer various paths
        print("Inferring paths...")
        input_image_path = find_input_image(data_dir)
        pseudo_video_path = os.path.join(data_dir, "inference_rgb.mp4")
        mask_video_path = os.path.join(data_dir, "inference_mask.mp4")
        
        # Check if video files exist
        if not os.path.exists(pseudo_video_path):
            raise FileNotFoundError(f"Pseudo video file does not exist: {pseudo_video_path}")
        if not os.path.exists(mask_video_path):
            raise FileNotFoundError(f"Mask video file does not exist: {mask_video_path}")
        
        print(f"Using paths:")
        print(f"  Model root directory: {model_root_dir}")
        print(f"  Data directory: {data_dir}")
        print(f"  Edited image: {input_image_path}")
        print(f"  Pseudo video: {pseudo_video_path}")
        print(f"  Mask video: {mask_video_path}")
        
        # Automatically find all safetensors files starting with diffusion_pytorch_model
        diffusion_model_pattern = os.path.join(model_root_dir, "diffusion_pytorch_model*.safetensors")
        diffusion_model_files = sorted(glob.glob(diffusion_model_pattern))

        print("Loading models...")
        model_manager = ModelManager(device="cpu")
        model_manager.load_models([
            os.path.join(model_root_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ], torch_dtype=torch.float32)
        model_manager.load_models([
            diffusion_model_files,
            os.path.join(model_root_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            os.path.join(model_root_dir, "Wan2.1_VAE.pth"),
        ], torch_dtype=torch.bfloat16)
        pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
        pipe.enable_vram_management(num_persistent_param_in_dit=0)

        # Initialize Florence model
        print("Initializing Florence model...")
        init_florence_model()

        # Load edited image
        input_image = Image.open(input_image_path)

        # Read concept prefix from prefix.txt
        prefix_file = os.path.join(data_dir, 'prefix.txt')
        concept_prefix = ""
        if os.path.exists(prefix_file):
            try:
                with open(prefix_file, 'r', encoding='utf-8') as f:
                    concept_prefix = f.read().strip()
                print(f"Read concept prefix from {prefix_file}: {concept_prefix}")
            except Exception as e:
                print(f"Failed to read prefix.txt file: {e}")
                concept_prefix = "p3rs0n,"  # Use default value
        else:
            print(f"prefix.txt file not found: {prefix_file}, using default prefix")
            concept_prefix = "p3rs0n,"


        # 1. Get the visual description from Florence (The "What it looks like")
        print("Analyzing image...")
        visual_description = generate_caption(input_image, concept_prefix=concept_prefix)
        
        # 2. Get your action from prompt.txt (The "What it does")
        prompt_file = os.path.join(data_dir, 'prompt.txt')
        action_prompt = ""
        
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                action_prompt = f.read().strip()
            print(f"Found action prompt: {action_prompt}")

        # 3. Combine them into ONE prompt for the pipeline
        if action_prompt:
            # Combining visual + ac ion
            final_prompt = f"{action_prompt}"
        else:
            final_prompt = visual_description

        print(f"Final merged prompt sent to model: {final_prompt}")

      
        # Generate random seed
        random_seed = random.randint(0, 2**32 - 1)
        print(f"Using random seed: {random_seed}")

        print("Starting inference...")
        # video = pipe(
        #     prompt=visual_description,
        #     negative_prompt="Overexposure, static, blurred details, subtitles, paintings, pictures, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, mutilated, redundant fingers, poorly painted hands, poorly painted faces, deformed, disfigured, deformed limbs, fused fingers, cluttered background, three legs, a lot of people in the background, upside down",
        #     input_image=input_image,
        #     pseudo_video_path=pseudo_video_path,
        #     mask_video_path=mask_video_path,
        #     num_inference_steps=30,
        #     seed=random_seed, tiled=True,
        #     # TeaCache parameters
        #     tea_cache_l1_thresh=0.275, # The larger this value is, the faster the speed, but the worse the visual quality.
        #     tea_cache_model_id="Wan2.1-I2V-14B-480P", # Choose one in (Wan2.1-T2V-1.3B, Wan2.1-T2V-14B, Wan2.1-I2V-14B-480P, Wan2.1-I2V-14B-720P).
        # )

        video = pipe(
            prompt=final_prompt,              # <--- Use the combined prompt with visual description and action
            input_image=input_image,       # <--- Crucial: Only provide the image, not the video
            
            # CRITICAL: Set these to None to ensure you don't copy old motion
            input_video=None,   
            pseudo_video_path=pseudo_video_path,
            mask_video_path=mask_video_path,            

            negative_prompt="Overexposure, static, blurred details, subtitles, paintings, pictures, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, mutilated, redundant fingers, poorly painted hands, poorly painted faces, deformed, disfigured, deformed limbs, fused fingers, cluttered background, three legs, a lot of people in the background, upside down",
            
            # Motion Settings
            seed=42,                        # Change this to get different variations
            num_frames=81,  
            num_inference_steps=30,                # Duration
            # See explanation below (Higher = More motion)
            denoising_strength=1.0,          # Must be 1.0 so it ignores latent noise and follows prompt
            tea_cache_l1_thresh=0.275,
            tea_cache_model_id="Wan2.1-I2V-14B-480P"
        )
        
        output_path = os.path.join(data_dir, "edited_video.mp4")
        save_video(video, output_path, fps=30, quality=5)
        print(f"Video saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video generation inference script")
    parser.add_argument("--model_root_dir", required=True, help="Model root directory path")
    parser.add_argument("--data_dir", required=True, help="Data directory path")
    
    args = parser.parse_args()
    
    main(args.model_root_dir, args.data_dir)