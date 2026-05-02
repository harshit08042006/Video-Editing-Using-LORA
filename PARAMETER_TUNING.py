"""
Parameter Tuning Guide for Border Smoothing

This file helps you find the optimal parameters for your specific use case.
"""

# ============================================================================
# PARAMETER TUNING QUICK REFERENCE
# ============================================================================

PARAMETER_RECOMMENDATIONS = {
    "blend_strength": {
        "description": "How much to blend edited with original at borders",
        "range": [0.0, 1.0],
        "recommended": 0.6,
        "adjustments": {
            "increase": "If jitter is still visible",
            "decrease": "If edited motion is being lost",
        },
        "presets": {
            "minimal": 0.3,      # Preserve most edited motion
            "balanced": 0.6,     # Good balance (default)
            "aggressive": 0.9,   # Maximum smoothing
        }
    },
    
    "border_width": {
        "description": "Width of border region affected (in pixels)",
        "range": [1, 20],
        "recommended": 5,
        "adjustments": {
            "increase": "If jitter extends far from boundary",
            "decrease": "If you're losing too much edited motion",
        },
        "presets": {
            "small": 3,          # Tight border, preserve details
            "medium": 5,         # Standard (default)
            "large": 10,         # Wide border, aggressive smoothing
        }
    },
    
    "blur_sigma": {
        "description": "Gaussian blur on mask boundary",
        "range": [0.5, 3.0],
        "recommended": 1.5,
        "adjustments": {
            "increase": "For smoother, wider transitions",
            "decrease": "For sharper boundaries",
        },
        "presets": {
            "sharp": 0.8,        # Sharp transition
            "smooth": 1.5,       # Smooth transition (default)
            "very_smooth": 2.5,  # Very smooth, wide blend",
        }
    }
}


# ============================================================================
# PRESET CONFIGURATIONS FOR DIFFERENT SCENARIOS
# ============================================================================

SCENARIO_PRESETS = {
    "high_jitter": {
        "description": "High jitter visible at borders",
        "blend_strength": 0.7,
        "border_width": 7,
        "blur_sigma": 2.0,
    },
    
    "low_jitter": {
        "description": "Minimal jitter, just needs fine-tuning",
        "blend_strength": 0.4,
        "border_width": 3,
        "blur_sigma": 1.0,
    },
    
    "preserve_motion": {
        "description": "Want to preserve edited motion, minimal blending",
        "blend_strength": 0.3,
        "border_width": 3,
        "blur_sigma": 0.8,
    },
    
    "maximum_smoothing": {
        "description": "Want maximum smoothing regardless of motion loss",
        "blend_strength": 0.85,
        "border_width": 10,
        "blur_sigma": 2.5,
    },
    
    "balanced": {
        "description": "Default balanced preset",
        "blend_strength": 0.6,
        "border_width": 5,
        "blur_sigma": 1.5,
    },
    
    "fine_details": {
        "description": "Preserve fine details at boundaries",
        "blend_strength": 0.3,
        "border_width": 2,
        "blur_sigma": 0.5,
    },
}


# ============================================================================
# TUNING PROCEDURE
# ============================================================================

def get_tuning_sequence():
    """
    Recommended sequence for tuning parameters.
    """
    return [
        {
            "step": 1,
            "parameter": "blend_strength",
            "range": [0.3, 0.6, 0.9],
            "description": "Start with blend_strength. Try 0.3, 0.6, 0.9. Pick best.",
            "tip": "Smallest value that removes jitter."
        },
        {
            "step": 2,
            "parameter": "blur_sigma",
            "range": [0.8, 1.5, 2.5],
            "description": "Fine-tune blur for smoothness of transition.",
            "tip": "Increase if blend_strength changes are too abrupt."
        },
        {
            "step": 3,
            "parameter": "border_width",
            "range": [3, 5, 7],
            "description": "Adjust border width based on jitter extent.",
            "tip": "Increase if jitter extends far from mask edge."
        },
        {
            "step": 4,
            "parameter": "iterate",
            "description": "Fine-tune blend_strength again with optimal blur and width.",
            "tip": "Small adjustments (±0.1) usually sufficient."
        }
    ]


# ============================================================================
# DIAGNOSTIC COMMANDS
# ============================================================================

DIAGNOSTIC_COMMANDS = {
    "test_mild_jitter": """
# Test configuration for mild jitter
python border_smoothing_inference.py \\
    --edited_video output/edited_video.mp4 \\
    --original_video input/original_video.mp4 \\
    --mask_video input/mask.mp4 \\
    --output output/test_mild.mp4 \\
    --blend_strength 0.4 \\
    --border_width 3 \\
    --blur_sigma 1.0
""",
    
    "test_moderate_jitter": """
# Test configuration for moderate jitter (recommended starting point)
python border_smoothing_inference.py \\
    --edited_video output/edited_video.mp4 \\
    --original_video input/original_video.mp4 \\
    --mask_video input/mask.mp4 \\
    --output output/test_moderate.mp4 \\
    --blend_strength 0.6 \\
    --border_width 5 \\
    --blur_sigma 1.5
""",
    
    "test_severe_jitter": """
# Test configuration for severe jitter
python border_smoothing_inference.py \\
    --edited_video output/edited_video.mp4 \\
    --original_video input/original_video.mp4 \\
    --mask_video input/mask.mp4 \\
    --output output/test_severe.mp4 \\
    --blend_strength 0.8 \\
    --border_width 8 \\
    --blur_sigma 2.0
""",
    
    "test_preserve_motion": """
# Test configuration prioritizing motion preservation
python border_smoothing_inference.py \\
    --edited_video output/edited_video.mp4 \\
    --original_video input/original_video.mp4 \\
    --mask_video input/mask.mp4 \\
    --output output/test_preserve.mp4 \\
    --blend_strength 0.3 \\
    --border_width 2 \\
    --blur_sigma 0.8
""",
}


# ============================================================================
# EFFECTIVENESS EVALUATION
# ============================================================================

def evaluate_results(metric="visual"):
    """
    How to evaluate border smoothing effectiveness.
    """
    
    evaluation_methods = {
        "visual_inspection": [
            "1. Load result in video player",
            "2. Play at slow speed (0.5x)",
            "3. Look at mask boundaries frame-by-frame",
            "4. Check for: flickering, jitter, smooth transitions",
            "5. Verify edited motion is preserved inside mask",
        ],
        
        "specific_checks": [
            "✓ Are borders smooth across frames?",
            "✓ Is there less flickering than original?",
            "✓ Is edited motion still preserved?",
            "✓ Is background stable outside mask?",
            "✓ Are transitions natural looking?",
        ],
        
        "quantitative": [
            "Measure pixel-level variance at borders across frames",
            "Compare with original: should see 40-70% reduction",
            "Check SSIM (structural similarity) with original",
            "Higher SSIM at borders = better smoothing",
        ]
    }
    
    return evaluation_methods


# ============================================================================
# COMMON ISSUES & SOLUTIONS
# ============================================================================

TROUBLESHOOTING = {
    "still_seeing_jitter": {
        "possible_causes": [
            "blend_strength too low",
            "border_width too small",
            "blur_sigma too small",
        ],
        "solution": "Increase blend_strength first (try 0.7-0.8)",
        "alternative": "Increase border_width to 7-10",
    },
    
    "losing_edited_motion": {
        "possible_causes": [
            "blend_strength too high",
            "border_width too large",
            "blend_strength affecting too much area",
        ],
        "solution": "Decrease blend_strength to 0.3-0.4",
        "alternative": "Reduce border_width to 2-3",
    },
    
    "transitions_look_unnatural": {
        "possible_causes": [
            "blur_sigma mismatch with border_width",
            "Hard jump between blended and non-blended",
        ],
        "solution": "Adjust blur_sigma to 1.5-2.5",
        "alternative": "Fine-tune border_width",
    },
    
    "background_looks_wrong": {
        "possible_causes": [
            "Original video has different artifacts",
            "Mask is misaligned",
        ],
        "solution": "Check mask alignment with edited video",
        "alternative": "Reduce blend_strength if background looks wrong",
    },
    
    "too_much_processing_time": {
        "possible_causes": [
            "High resolution video",
            "Inefficient parameter tuning",
        ],
        "solution": "Reduce blur_sigma and border_width",
        "alternative": "Process shorter segment for testing",
    },
}


# ============================================================================
# BATCH TESTING SCRIPT
# ============================================================================

def generate_batch_test_script(video_paths: dict):
    """
    Generate script to test multiple parameter combinations.
    
    Args:
        video_paths: Dict with keys: edited_video, original_video, mask_video, output_dir
    """
    
    test_configs = [
        ("mild", 0.4, 3, 1.0),
        ("moderate", 0.6, 5, 1.5),
        ("strong", 0.8, 7, 2.0),
        ("preserve", 0.3, 2, 0.8),
    ]
    
    script = "#!/bin/bash\n"
    script += "# Batch testing script - run all configurations\n\n"
    
    for name, blend, border, blur in test_configs:
        script += f"""# Configuration: {name}
python border_smoothing_inference.py \\
    --edited_video {video_paths['edited_video']} \\
    --original_video {video_paths['original_video']} \\
    --mask_video {video_paths['mask_video']} \\
    --output {video_paths['output_dir']}/result_{name}.mp4 \\
    --blend_strength {blend} \\
    --border_width {border} \\
    --blur_sigma {blur}

echo "Generated: result_{name}.mp4"
"""
    
    return script


# ============================================================================
# INTERACTIVE TUNING (OPTIONAL)
# ============================================================================

def interactive_parameter_selector():
    """
    Interactive CLI for parameter selection.
    Run this to get guided tuning.
    """
    
    print("\n" + "="*60)
    print("INTERACTIVE PARAMETER TUNING")
    print("="*60)
    
    print("\nDescribe your jitter issue:")
    print("1. Mild jitter (barely noticeable)")
    print("2. Moderate jitter (visible but acceptable)")
    print("3. Severe jitter (very noticeable)")
    print("4. Extreme jitter (makes video unwatchable)")
    
    severity = input("Select (1-4): ").strip()
    
    severity_map = {
        "1": SCENARIO_PRESETS["low_jitter"],
        "2": SCENARIO_PRESETS["balanced"],
        "3": SCENARIO_PRESETS["high_jitter"],
        "4": SCENARIO_PRESETS["maximum_smoothing"],
    }
    
    config = severity_map.get(severity, SCENARIO_PRESETS["balanced"])
    
    print("\nIs preserving edited motion important?")
    print("1. Yes, very important")
    print("2. Moderate importance")
    print("3. Not important, smooth it all")
    
    motion_prio = input("Select (1-3): ").strip()
    
    if motion_prio == "1":
        config = SCENARIO_PRESETS["preserve_motion"]
    elif motion_prio == "3":
        config = SCENARIO_PRESETS["maximum_smoothing"]
    
    print("\n" + "="*60)
    print("RECOMMENDED PARAMETERS:")
    print("="*60)
    print(f"blend_strength: {config['blend_strength']}")
    print(f"border_width:   {config['border_width']}")
    print(f"blur_sigma:     {config['blur_sigma']}")
    print("="*60 + "\n")
    
    return config


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "interactive":
            config = interactive_parameter_selector()
            print("Use these parameters in your command:")
            print(f"--blend_strength {config['blend_strength']} \\")
            print(f"--border_width {config['border_width']} \\")
            print(f"--blur_sigma {config['blur_sigma']}")
        
        elif command == "help":
            print("Parameter Tuning Guide")
            print("-" * 60)
            for param, info in PARAMETER_RECOMMENDATIONS.items():
                print(f"\n{param}:")
                print(f"  {info['description']}")
                print(f"  Recommended: {info['recommended']}")
        
        else:
            print(f"Unknown command: {command}")
            print("Available: interactive, help")
    
    else:
        # Print all information
        print("Parameter Tuning Reference")
        print("="*60)
        print("\nDiagnostic Commands:")
        for name, cmd in DIAGNOSTIC_COMMANDS.items():
            print(f"\n# {name}")
            print(cmd)
