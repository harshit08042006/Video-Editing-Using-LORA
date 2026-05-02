# Border Jitter Reduction - Visual Implementation Guide

## Problem Visualization

```
BEFORE (With Jitter):
┌─────────────────────────────────────┐
│  Frame N                            │
│  ┌──────────────────────┐           │
│  │ ✓ Edited Motion      │           │
│  │ (Good)               │           │
│  ├──────────────────────┤           │ ← JITTER HERE!
│  │~~~~~~~~~~~~~~~~~~~~~│ ← Border   │ (Inconsistent across frames)
│  │ ~ Unstable Border ~ │ (BAD)      │
│  │~~~~~~~~~~~~~~~~~~~~~│           │
│  │ ✓ Original BG       │           │
│  │ (Preserved)         │           │
│  └──────────────────────┘           │
└─────────────────────────────────────┘

Frame N+1 (Similar but Different) → FLICKER!
```

```
AFTER (Smooth):
┌─────────────────────────────────────┐
│  Frame N                            │
│  ┌──────────────────────┐           │
│  │ ✓ Edited Motion      │           │
│  │ (Good)               │           │
│  ├══════════════════════┤           │
│  │ ░░░░░░░░░░░░░░░░░ │ ← Smooth   │
│  │ ░ Blended Border ░ │ (GOOD)    │
│  │ ░░░░░░░░░░░░░░░░░ │           │
│  │ ✓ Original BG       │           │
│  │ (Preserved)         │           │
│  └──────────────────────┘           │
└─────────────────────────────────────┘

Frame N+1 (Same blend strategy) → STABLE!
```

---

## Algorithm Flow

```
INPUT VIDEO FRAMES
    ↓
┌─────────────────────────────────────┐
│  Load Videos:                       │
│  - edited_video.mp4                 │
│  - original_video.mp4               │
│  - mask.mp4                         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  FOR each frame in sequence:        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  1. Load frame triplet:             │
│     edited[t], original[t], mask[t] │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  2. Create Border Mask:             │
│     Dilate mask → Erode mask        │
│     Border = Dilated - Eroded       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  3. Smooth Boundary:                │
│     Gaussian Blur on mask edges     │
│     (σ = 1.5)                       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  4. Blend:                          │
│     result = edited × (1 - blend)   │
│            + original × blend       │
│                                     │
│     blend = high at border,         │
│             low inside mask         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  5. Save frame                      │
└─────────────────────────────────────┘
    ↓
Output: SMOOTH VIDEO!
```

---

## Blend Mask Creation

```
BINARY MASK (Hard edges):
┌─────────────────────┐
│ 000000000000000000  │
│ 000111111111000000  │
│ 001111111111100000  │
│ 001111111111100000  │
│ 000111111111000000  │
│ 000000000000000000  │
└─────────────────────┘
     ↓ Dilate
┌─────────────────────┐
│ 000000000000000000  │
│ 000111111111000000  │
│ 001111111111100000  │
│ 001111111111100000  │
│ 000111111111000000  │
│ 000000000000000000  │
└─────────────────────┘

     ↓ Erode
┌─────────────────────┐
│ 000000000000000000  │
│ 000011111100000000  │
│ 000111111100000000  │
│ 000111111100000000  │
│ 000011111100000000  │
│ 000000000000000000  │
└─────────────────────┘

     ↓ Border = Dilated - Eroded
┌─────────────────────┐
│ 000000000000000000  │
│ 000100000001000000  │
│ 001000000001100000  │
│ 001000000001100000  │
│ 000100000001000000  │
│ 000000000000000000  │
└─────────────────────┘
        (Border region)

     ↓ Gaussian Blur
┌─────────────────────┐
│ 000000000000000000  │
│ 000.5...0.5000000   │
│ 00.7.......7.00000  │
│ 00.7.......7.00000  │
│ 000.5...0.5000000   │
│ 000000000000000000  │
└─────────────────────┘
   (Smooth blend mask)
```

---

## Implementation Architecture

```
USER
  ↓
  ├─→ border_smoothing_inference.py
  │   (Main user interface)
  │   ├─→ load_video_frames()
  │   ├─→ apply_border_smoothing_to_video()
  │   └─→ Command-line parser
  │
  ├─→ attention_border_blending.py
  │   (Core algorithms)
  │   ├─→ BorderAttentionBlender
  │   │   ├─→ create_border_mask()
  │   │   ├─→ blend_attention_at_border()
  │   │   ├─→ extract_border_region()
  │   │   └─→ smooth_mask_boundary()
  │   └─→ apply_border_attention_blending()
  │
  └─→ integration_example.py
      (Integration templates)
      ├─→ apply_border_smoothing_postprocess()
      ├─→ apply_border_smoothing_inline()
      └─→ test_border_smoothing()

OUTPUT: Smooth video with reduced jitter at borders
```

---

## Parameter Effects Visualization

### blend_strength = 0.3
```
Edited:     ████░░░░░░░░░░░░░░░░░░░░████
Original:   ░░░░████████████████████░░░░
Result:     ████░░░░░░░░░░░░░░░░░░░░████
            (Mostly edited, minimal blending)
```

### blend_strength = 0.6 (Recommended)
```
Edited:     ████░░░░░░░░░░░░░░░░░░░░████
Original:   ░░░░████████████████████░░░░
Result:     ████▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓████
            (Good balance at borders)
```

### blend_strength = 0.9
```
Edited:     ████░░░░░░░░░░░░░░░░░░░░████
Original:   ░░░░████████████████████░░░░
Result:     ████████████████████████████
            (Maximum blending, smooth but less motion)
```

---

## Processing Pipeline Example

```
Step 1: Load Video Segment
╔════════════════════════════════╗
║ Frame 1: edited_frame_1.jpg    ║
║ Frame 2: edited_frame_2.jpg    ║
║ Frame 3: edited_frame_3.jpg    ║
║ ... (81 frames total)          ║
╚════════════════════════════════╝

Step 2: Detect Border Regions
╔════════════════════════════════╗
║ Frame 1: Border pixels = 2,340 ║
║ Frame 2: Border pixels = 2,310 ║
║ Frame 3: Border pixels = 2,370 ║
║ (Same mask applied)            ║
╚════════════════════════════════╝

Step 3: Create Blend Masks
╔════════════════════════════════╗
║ Frame 1: Blend range [0.0-0.6] ║
║ Frame 2: Blend range [0.0-0.6] ║
║ Frame 3: Blend range [0.0-0.6] ║
║ (Consistent across frames)     ║
╚════════════════════════════════╝

Step 4: Blend Frames
╔════════════════════════════════╗
║ Frame 1: edited(inside) +      ║
║          original(border)      ║
║ Frame 2: edited(inside) +      ║
║          original(border)      ║
║ Frame 3: edited(inside) +      ║
║          original(border)      ║
║ (Temporal consistency!)        ║
╚════════════════════════════════╝

Output: Smooth video!
```

---

## Comparison: Different Strategies

```
STRATEGY 1: No Processing (Current)
├─ ✓ Fastest
├─ ✓ Preserves all edited motion
└─ ✗ Jitter visible at borders

STRATEGY 2: Gaussian Blur on Mask (Simple)
├─ ✓ Fast, simple
├─ ✓ Reduces some jitter
└─ ✗ May lose detail at borders

STRATEGY 3: Border Blending ← RECOMMENDED
├─ ✓ Smooth borders
├─ ✓ Preserves edited motion
├─ ✓ Good balance
└─ △ Adds ~30% processing time

STRATEGY 4: Full Video Blend (Aggressive)
├─ ✓ Very smooth
├─ ✗ Loses edited motion
└─ ✗ Slower

STRATEGY 5: Temporal Consistency (Training)
├─ ✓ Most effective
├─ ✓ Learning-based solution
└─ ✗ Requires retraining (Time/compute)
```

---

## Real-World Example

```
Project: Girl passes potted plant

INPUT MASK (White = Foreground):
    ┌──────────────────────────────┐
    │ Girl region (white):         │
    │ ┌────────────┐               │
    │ │           │                │
    │ │   GIRL    │                │
    │ │           │                │
    │ └────────────┘               │
    │ Background (black)           │
    └──────────────────────────────┘

AFTER BORDER BLENDING:
    ┌──────────────────────────────┐
    │ ✓ Girl motion (edited):      │
    │ ┌────────────┐               │
    │ │           │                │
    │ │   GIRL    │ ← Smooth       │
    │ │           │   Motion       │
    │ └════════════┘               │
    │ ░░░░░░░░░░░░░░░░░░░░░░░░░░  │ ← Smooth border
    │ ✓ Background (original):     │ ← Stable, no flicker
    └──────────────────────────────┘

RESULT: Girl moves smoothly without jitter at edges!
```

---

## When to Use Each Technique

```
Jitter Severity vs Recommended Approach:

┌────────────────────────────────────────────────────┐
│ Very Mild                                          │
│ (barely visible)                  → Skip (optional)│
├────────────────────────────────────────────────────┤
│ Mild                                               │
│ (visible on close inspection)  → Border Blending  │
│                                  (blend_str=0.4)  │
├────────────────────────────────────────────────────┤
│ Moderate                                           │
│ (clearly visible)            → Border Blending    │
│                                (blend_str=0.6)    │
├────────────────────────────────────────────────────┤
│ Severe                                             │
│ (very distracting)          → Border Blending     │
│                              + Mask Smoothing    │
│                              (blend_str=0.8)    │
├────────────────────────────────────────────────────┤
│ Extreme                                            │
│ (makes video unwatchable)   → Multiple techniques:│
│                              - Border Blending    │
│                              - Temporal Filtering │
│                              - Optical Flow       │
└────────────────────────────────────────────────────┘
```

---

## Performance Metrics

```
Processing Time vs Video Resolution:

480p (typical):  2-5 minutes for 1-minute video
720p:            5-10 minutes for 1-minute video
1080p:           10-20 minutes for 1-minute video

Memory Usage:    ~1-2GB for 1-minute video
GPU Memory:      Not significantly increased

Speedup possible with:
- Reduce blur_sigma (0.8 instead of 1.5)
- Reduce border_width (3 instead of 5)
- Process on CPU if GPU busy
```

---

## Checklist for Implementation

```
□ Copy attention_border_blending.py
□ Copy border_smoothing_inference.py
□ Run verify_border_smoothing.py
□ Test on 10-second sample video
□ Verify output has no jitter at borders
□ Verify edited motion is preserved inside mask
□ Tune parameters (blend_strength)
□ Process full video
□ Compare with original
□ Document results
□ Present to professor
```

---

## Next Steps After Implementation

```
Week 1: Test & Validate
├─ Run on sample videos
├─ Measure jitter reduction
└─ Optimize parameters

Week 2: Integration
├─ Add to main pipeline
├─ Test on full dataset
└─ Document results

Week 3: Refinement
├─ Try other techniques if needed
├─ Compare approaches
└─ Finalize implementation

Week 4: Presentation
├─ Show before/after videos
├─ Explain methodology
└─ Discuss results with professor
```

---

This visual guide helps understand what's happening. For detailed implementation, see the Python files! 🚀
