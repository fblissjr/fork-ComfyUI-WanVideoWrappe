# Apollo Paper Experimental Fork of [Kijai's ComfyUI wrapper nodes](https://github.com/kijai/ComfyUI-WanVideoWrapper) for [WanVideo](https://github.com/Wan-Video/Wan2.1)

We need to make sure fps sampling and context window for fps sampling and the fps sampler and interpolation are all working accurately. I don't think they are.
How does this change our context.pyand nodes.py nodes for WanVideoContextOptionsFPS, WanVideoFPSSampler, WanVideoLatentInterpolator, and if relevant, WanVideoSampler?
Are there better places to implement this in the ./wanvideo/ sections of the code base?
While you analyze that, let's also plan to fully implement Temporal Perception Emphasis, Perceiver Resampling, and tightly integrating it with the ./wanvideo/modules architecture, model, and potentially the solvers in ./wanvideo/utils
Output the full code changes required.

# FORK UPDATES

- Added fps sampling for context windows as two new nodes - you need to use `WanVideoContextOptionsFPS` connected to a new sampler, `WanVideoFPSSampler`. Experimental. Very experimental.
- Added `WanVideoLatentInterpolator` to interpolate frames in latent space, using with SLERP or Linear interpolation
- inspiration: Apollo paper: [Apollo: An Exploration of Video Understanding in Large Multimodal Models](https://arxiv.org/abs/2412.10360)

## Why FPS Sampling + Latent Interpolation works? (theoretically)

- **caveat**: first, it may not work well without the model having been trained on it specifically - i'm mostly trying to see how far it can be pushed via inference alone

- but tldr:

1. **FPS sampling** ensures consistent _semantic_ understanding of time in the diffusion model

   - The model learns proper motion understanding at a specific FPS
   - Apollo paper showed this improves temporal consistency significantly

2. **Latent interpolation** works in the model's "understanding space"

   - Between two semantically coherent keyframes, the latent space contains meaningful transitions
   - SLERP especially follows curved paths in latent space that often align with natural motion arcs

3. **Together, they create a "best of both worlds" scenario**:
   - Computationally efficient (generate fewer frames)
   - Temporally consistent (frames at meaningful intervals)
   - Smooth motion (intelligent interpolation)
   - Memory efficient (smaller batches during diffusion)

## Recommended Experiments

If you want to see the difference most clearly:

1. **Baseline**: Generate 24fps directly without interpolation
2. **Efficient**: Generate 4fps with 6× interpolation (same 24fps result)
3. **Artistic**: Generate 2fps with 12× interpolation using different easing methods

The efficient approach should be ~6× faster while maintaining quality, and the artistic approach lets you experiment with different motion styles.

For the most consistent results, this equation should help:

```
target_fps × interpolation_factor = video_combine_frame_rate
```

# Some guidance on the new nodes, what they do, and what to try

## FPS Sampling Context Window, `WanVideoContextOptionsFPS`, & FPS Sampler, `WanVideoFPSSampler`, Parameters

### `context_schedule` (uniform_standard_fps)

- `uniform_standard_fps` uses Apollo paper's approach of consistent timing

- **Effect**: Determines how the video is split into chunks for processing
- **When to adjust**: Change to "uniform_standard" if you notice temporal inconsistencies
- **Example**: "uniform_standard_fps" maintains consistent time intervals between frames, while "uniform_standard" distributes frames evenly regardless of timing

### `context_frames` (81)

- Number of frames handled in each context window (81 is good)

- **Effect**: The number of frames processed together in each window
- **When to adjust**: Higher values capture longer-term dependencies but use more VRAM
- **Example**: Setting to 121 might give more coherent motion but use more memory; setting to 41 uses less memory but might have more abrupt transitions

### `context_stride` (4)

- How far window advances between steps (4 is small, fine-grained)

- **Effect**: How far the window advances between steps
- **When to adjust**: Lower values give more overlap but increase computation
- **Example**: If context_frames=81 and context_stride=4, each step processes 77 new frames

### `context_overlap` (16)

- Frames overlapping between windows (16 provides good continuity)

- **Effect**: Number of frames that overlap between consecutive windows
- **When to adjust**: Higher values give smoother transitions between windows
- **Example**: Setting to 24 gives smoother transitions; setting to 8 may create more visible seams between sections

### `freenoise` (true)

- **Effect**: Shuffles noise to prevent repetitive patterns
- **When to adjust**: Turn off if you want more deterministic generation
- **Example**: With freenoise=true, longer videos have more variation; with false, patterns might repeat

### `target_fps` (2.0)

- The conceptual FPS the model should "think" in (2.0 is low, dreamy)

- **Effect**: The conceptual frames per second for temporal consistency
- **When to adjust**: This is what Apollo found crucial for motion understanding
- **Example**: 1.0 creates slower perceived motion; 4.0 creates faster perceived motion

### `original_fps` (30)

- Reference for "normal" speed (30 is standard)

- **Effect**: Reference value used in temporal calculations
- **When to adjust**: Rarely needs changing; it's a baseline for calculations
- **Example**: This mainly affects how the model internally calculates time relations

## `WanVideoLatentInterpolator` Paraameters

### interpolation_factor

- How many frames to create between keyframes (6 means 5 new frames between each original)

- **`method`**:

  - `linear`: Direct averaging, cleaner for small interpolations
  - `slerp`: Spherical interpolation, better maintains semantic coherence

- **`ease_method`**:
  - `linear`: Uniform interpolation
  - `ease_in`: Slow→Fast (acceleration effect)
  - `ease_out`: Fast→Slow (deceleration effect, most natural)
  - `ease_in_out`: Slow→Fast→Slow (gentle transitions)

## Recommended Values for Text-to-Video Generation

- `context_schedule`: "uniform_standard_fps"
- `context_frames`: 81
- `context_stride`: 4
- `context_overlap`: 16
- `freenoise`: true
- `target_fps`: 2.0
- `original_fps`: 30

For WanVideoFPSSampler:

- `sampling_mode`: "fps"
- `target_fps`: 2.0
- `original_fps`: 30

## Variations to Experiment With

### Motion Speed Variations

- **Slow motion**: target_fps=1.0, original_fps=30
- **Apollo recommended**: target_fps=2.0, original_fps=30
- **Faster motion**: target_fps=4.0, original_fps=30

### Window Size Variations

- **Low VRAM**: context_frames=41, context_overlap=8
- **Balanced**: context_frames=81, context_overlap=16
- **High quality**: context_frames=121, context_overlap=24

### Rules of Thumb

- **Effective new frames per window** = context_frames - context_overlap
- **Video playback duration** ≈ num_frames ÷ playback_fps
- **Optimal overlap ratio**: context_overlap ≈ 20% of context_frames
- **Generation duration** = num_frames ÷ target_fps

## additional ideas to try from the apollo paper

1. **changing the vision encoder**:

   - Apollo found combining SigLIP-SO400M with InternVideo2 gave best results

2. **Perceiver Resampling**:

   - add parameters to control resampling token count more precisely
   - pseudocode implementation:

   ```python
   # Add to WanVideoFPSSampler
   base_inputs["required"]["tokens_per_frame"] = ("INT", {"default": 16, "min": 4, "max": 64, "step": 4})
   ```

3. **Temporal Perception Emphasis**:
   - add a parameter that weights temporal vs. spatial information:

   ```python
   base_inputs["required"]["temporal_emphasis"] = ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1})
   ```

   - this would modify how the model interprets motion vs. static elements

# WORK IN PROGRESS

# Installation

1. Clone this repo into `custom_nodes` folder.
2. Install dependencies: `pip install -r requirements.txt`
   or if you use the portable install, run this in ComfyUI_windows_portable -folder:

`python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-WanVideoWrapper\requirements.txt`

## Models

<https://huggingface.co/Kijai/WanVideo_comfy/tree/main>

Text encoders to `ComfyUI/models/text_encoders`

Transformer to `ComfyUI/models/diffusion_models`

Vae to `ComfyUI/models/vae`

---

Context window test:

1025 frames using window size of 81 frames, with 16 overlap. With the 1.3B T2V model this used under 5GB VRAM and took 10 minutes to gen on a 5090:

<https://github.com/user-attachments/assets/89b393af-cf1b-49ae-aa29-23e57f65911e>

This very first test was 512x512x81

~16GB used with 20/40 blocks offloaded

<https://github.com/user-attachments/assets/fa6d0a4f-4a4d-4de5-84a4-877cc37b715f>

Vid2vid example:

with 14B T2V model:

<https://github.com/user-attachments/assets/ef228b8a-a13a-4327-8a1b-1eb343cf00d8>

with 1.3B T2V model

<https://github.com/user-attachments/assets/4f35ba84-da7a-4d5b-97ee-9641296f391e>
