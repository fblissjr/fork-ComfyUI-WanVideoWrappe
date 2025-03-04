# WanVideo FPS Sampling & Latent Interpolation User Guide

This guide explains how to use the FPS Sampling and Latent Interpolation features for generating smoother, more temporally consistent videos with WanVideo, while potentially speeding up generation time.

## How It Works (High Level)

This workflow uses two key optimizations:

1. **FPS-based sampling**: Generates frames at specific time intervals that match your target video's timing
2. **Latent interpolation**: Generates fewer keyframes through the diffusion process, then interpolates between them

The result is videos with more consistent timing and motion, generated faster than traditional methods.

## Node Configuration Guide

### WanVideo ContextOptions FPS

This node configures how frames are sampled in time and how the video is processed in chunks.

| Parameter | Example Value | Description |
|-----------|---------------|-------------|
| `context_schedule` | uniform_standard_fps | The algorithm used for dividing video into chunks. `uniform_standard_fps` ensures frames are sampled at consistent time intervals rather than evenly spaced regardless of duration. |
| `context_frames` | 81 | The number of frames to process in each "window" (chunk). Higher values capture more temporal context but use more VRAM. Range: 41 (low VRAM) to 121 (high quality). |
| `context_stride` | 4 | How many frames to advance between chunks. Lower values create more overlap and smoother transitions but increase computation time. |
| `context_overlap` | 16 | How many frames overlap between consecutive chunks. Higher values (16-24) create smoother transitions between chunks, lower values (4-8) may have visible seams. |
| `freenoise` | true | Shuffles noise patterns to prevent repetitive artifacts in longer videos. Recommended to leave on. |
| `target_fps` | 2.0 | The "conceptual" FPS the model should generate at. Lower values (1-2) create dreamy, slower motion; higher values (4-8) create more dynamic, faster motion. |
| `original_fps` | 30.0 | The reference frame rate. This is used for time calculations and rarely needs changing from 30. |
| `output_fps` | 24.0 | The desired final frame rate for the output video. Common values: 24 (cinematic), 30 (standard), 60 (smooth). |

### WanVideo FPS Config

This node centralizes FPS-related settings to ensure consistency across the workflow.

| Parameter | Example Value | Description |
|-----------|---------------|-------------|
| `generation_fps` | 2.0 | Same as `target_fps` - the rate at which frames are generated. Lower values (1-2) are more efficient but create fewer keyframes. |
| `output_fps` | 24.0 | The final frame rate after interpolation. This should match your video player's frame rate. |
| `source_fps` | 30.0 | The reference frame rate for time calculations (same as `original_fps`). |
| `context_frames` | 81 | Same as in ContextOptions - frames processed in each chunk. |
| `context_overlap` | 16 | Same as in ContextOptions - frames shared between chunks. |
| `enable_frame_blending` | true | Controls whether frames are blended at the boundaries between chunks. Recommended to leave on for smoother transitions. |

### WanVideo Latent Interpolator

This node creates intermediate frames between the generated keyframes.

| Parameter | Example Value | Description |
|-----------|---------------|-------------|
| `interpolation_factor` | 6 | How many frames to create between each keyframe. Should satisfy: `generation_fps × interpolation_factor = output_fps` (e.g., 4 × 6 = 24). |
| `method` | adaptive | The interpolation algorithm to use: <br>- `linear`: Simple averaging, good for subtle motions<br>- `slerp`: Spherical interpolation, better for complex motions<br>- `adaptive`: Automatically chooses between methods based on frame differences |
| `ease_method` | ease_out | The timing function for interpolation:<br>- `linear`: Uniform speed<br>- `ease_in`: Slow start, accelerates<br>- `ease_out`: Fast start, decelerates (most natural)<br>- `ease_in_out`: Slow start and end, faster in middle |
| `use_recommended_factor` | true | When enabled, uses the interpolation factor calculated from `generation_fps` and `output_fps`. |
| `motion_smoothness` | 0.50 | Controls temporal smoothing strength (0-1). Higher values reduce jitter but may lose some detail. |

### WanVideo FPS Sampler

This is the main sampling node that generates frames according to the FPS configuration.

| Parameter | Example Value | Description |
|-----------|---------------|-------------|
| `sampling_mode` | fps | The algorithm for frame sampling:<br>- `fps`: Time-consistent sampling (recommended)<br>- `uniform`: Evenly distributed frames regardless of duration |
| `generation_fps` | 2.0 | Same as in FPS Config - rate of keyframe generation. |
| `original_fps` | 30.0 | Same as in FPS Config - reference for time calculations. |
| `output_fps` | 24.0 | Same as in FPS Config - final frame rate. |
| `scheduler` | unipc | The diffusion sampling algorithm. `unipc` is generally fastest and most stable. |
| `steps` | 30 | Number of denoising steps. 20-30 is typical; higher values (50+) give slightly better quality at the cost of generation time. |
| `cfg` | 6.0 | Classifier-free guidance scale. Higher values (7-9) give stronger adherence to prompt; lower values (3-5) give more creative freedom. |
| `shift` | 5.0 | Controls sigma schedule for noise sampling. Rarely needs changing from 5.0. |
| `riflex_freq_index` | 0 | Advanced parameter for rotation frequencies. Leave at 0 unless creating looping animations. |

### Video Combine (After Decode)

| Parameter | Example Value | Description |
|-----------|---------------|-------------|
| `frame_rate` | 16 | The playback frame rate for the final video file. Should match or be divisible by `output_fps`. |
| `loop_count` | 0 | Number of times to loop the video (0 = no loop). |

## Parameter Relationships & Dependencies

The most important relationship to maintain is:

```
generation_fps × interpolation_factor = output_fps
```

For example:
- 2fps generation × 12 interpolation = 24fps output
- 4fps generation × 6 interpolation = 24fps output
- 8fps generation × 3 interpolation = 24fps output

Other important relationships:

1. **Context window size**:
   - Larger `context_frames` improves temporal consistency but uses more VRAM
   - A good balance: `context_frames` = 81, `context_overlap` = 16

2. **Interpolation quality**:
   - Lower `generation_fps` with higher `interpolation_factor` = faster generation but might lose some detail
   - Higher `generation_fps` with lower `interpolation_factor` = slower generation but better detail preservation

3. **Frame blending**:
   - `enable_frame_blending` should generally be kept on for seamless transitions between chunks
   - Can be disabled for very distinct scene changes

## Creative Presets for Different Video Styles

### Cinematic Film (24fps)
```
generation_fps: 3.0
interpolation_factor: 8
method: adaptive
ease_method: ease_out
motion_smoothness: 0.7
steps: 30
cfg: 5.5
```
*Creates film-like motion with natural motion blur and timing.*

### Slow Motion / Dreamy
```
generation_fps: 1.0
interpolation_factor: 24
method: slerp
ease_method: ease_in_out
motion_smoothness: 0.8
steps: 30
cfg: 6.0
```
*Ultra-slow, dreamy motion with smooth transitions, great for ethereal scenes.*

### Action Sequence / Fast Motion
```
generation_fps: 6.0
interpolation_factor: 4
method: adaptive
ease_method: linear
motion_smoothness: 0.3
steps: 25
cfg: 7.0
```
*Faster, more dynamic motion with less interpolation, good for action sequences.*

### Timelapse
```
generation_fps: 8.0
interpolation_factor: 3
method: linear
ease_method: linear
motion_smoothness: 0.2
steps: 25
cfg: 8.0
```
*Fast, linear progression with minimal smoothing, ideal for timelapses and rapid transitions.*

### Security Camera / Surveillance
```
generation_fps: 2.0
interpolation_factor: 6
method: linear
ease_method: linear
motion_smoothness: 0.0
steps: 20
cfg: 5.0
```
*Slightly choppy, surveillance-like footage with minimal interpolation smoothing.*

### Stop Motion Animation
```
generation_fps: 5.0
interpolation_factor: 4
method: linear
ease_method: linear
motion_smoothness: 0.1
steps: 30
cfg: 7.5
```
*Slightly jerky motion that mimics stop-motion animation techniques.*

### Smooth Product Showcase
```
generation_fps: 4.0
interpolation_factor: 6
method: slerp
ease_method: ease_in_out
motion_smoothness: 0.9
steps: 35
cfg: 6.5
```
*Ultra-smooth camera movements for product showcases and commercial-style motion.*

### Retro Video Game 
```
generation_fps: 6.0
interpolation_factor: 2
method: linear
ease_method: linear
motion_smoothness: 0.0
steps: 20
cfg: 8.0
```
*Lower effective framerate with more abrupt transitions for a retro game feel.*

## Tips for Best Results

1. **Match the generation FPS to the content type**:
   - Slow, gentle scenes: 1-2 FPS generation
   - Medium movement scenes: 3-4 FPS generation
   - Fast action scenes: 6-8 FPS generation

2. **Choose the right interpolation method based on content**:
   - `linear`: Best for subtle, simple motion or timelapses
   - `slerp`: Best for complex motion, camera movements, and organic scenes
   - `adaptive`: Best for general purpose, mixed content

3. **Adjust ease methods for stylistic purposes**:
   - `linear`: Technical/documentary feel, timelapses
   - `ease_out`: Natural human/animal motion (most scenes)
   - `ease_in`: Objects gaining momentum
   - `ease_in_out`: Camera pans and smooth transitions

4. **Batch size vs. video length**:
   - For longer videos, use smaller `context_frames` (41-61)
   - For shorter, higher quality videos, use larger `context_frames` (81-121)

5. **Memory optimization**:
   - If running out of VRAM, reduce `context_frames` first
   - Then try reducing `generation_fps` and increasing `interpolation_factor`

## Troubleshooting Common Issues

- **Stuttering motion**: Increase `motion_smoothness` and ensure `enable_frame_blending` is on
- **Blurry interpolation**: Decrease `interpolation_factor` and increase `generation_fps`
- **Visible "seams" between chunks**: Increase `context_overlap` (16→24) 
- **Out of memory errors**: Reduce `context_frames` (81→61→41)
- **Strange looping artifacts**: Ensure `context_schedule` is set to `uniform_standard_fps`
