# fps sampling nodes
import os
import gc
import torch
import numpy as np
import math
from tqdm import tqdm
import re

from .nodes import WanVideoSampler
from .utils import log, print_memory

from .wanvideo.modules.clip import CLIPModel
from .wanvideo.modules.model import WanModel, rope_params
from .wanvideo.modules.t5 import T5EncoderModel
from .wanvideo.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .wanvideo.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from .enhance_a_video.globals import (
    enable_enhance,
    disable_enhance,
    set_enhance_weight,
    set_num_frames,
)

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file, save_torch_file, ProgressBar, common_upscale
import comfy.model_base
import comfy.latent_formats
import datetime

script_directory = os.path.dirname(os.path.abspath(__file__))

# region monkey patches for WanVideoSampler to avoid modifying sampler directly
_original_process = WanVideoSampler.process


# monkey patch
def apply_frame_blending(latent, windows, current_window_index):
    """
    Helper function to apply frame blending.
    Used by enhanced context scheduler in context.py.
    """
    if current_window_index >= len(windows) - 1:
        return

    current_window = windows[current_window_index]
    next_window = windows[current_window_index + 1]

    # Find overlap region
    overlap_frames = set(current_window).intersection(set(next_window))

    # If overlap exists, apply blending
    if overlap_frames:
        overlap_frames = sorted(list(overlap_frames))
        blend_range = len(overlap_frames)

        # Create smooth blending weights
        if blend_range > 1:
            for i, frame_idx in enumerate(overlap_frames):
                # Calculate blend factor (0.0 -> 1.0)
                blend_factor = i / (blend_range - 1)

                # Get the frame's position in each window
                current_pos = current_window.index(frame_idx)
                next_pos = next_window.index(frame_idx)

                # Blend the frame values
                latent[:, :, frame_idx, :, :] = (1.0 - blend_factor) * latent[
                    :, :, frame_idx, :, :
                ] + blend_factor * latent[:, :, frame_idx, :, :]

    return latent


# accounts for sampling interval in latent space
def sample_video_fps(total_latent_frames, original_fps, target_fps, latent_stride=4):
    """
    Sample frames at a consistent frames per second rate, accounting for latent space
    """
    # Convert to pixel space for time calculations
    total_pixel_frames = total_latent_frames * latent_stride
    pixel_video_length_sec = total_pixel_frames / original_fps

    # Calculate target frames in pixel space
    target_pixel_frames = int(pixel_video_length_sec * target_fps)

    # Convert back to latent space for sampling
    target_latent_frames = (target_pixel_frames - 1) // latent_stride + 1

    if target_latent_frames >= total_latent_frames:
        return list(range(total_latent_frames))

    # Calculate sampling interval in latent space
    sample_interval = total_latent_frames / target_latent_frames
    frame_indices = [
        min(int(i * sample_interval), total_latent_frames - 1)
        for i in range(target_latent_frames)
    ]

    return frame_indices


# Helper function for easing calculations in interpolation
def ease_function(t, method="linear"):
    """
    Apply various easing functions to interpolation factor t

    Args:
        t: Interpolation factor (0 to 1)
        method: Easing method - linear, ease_in, ease_out, or ease_in_out

    Returns:
        Modified interpolation factor
    """
    if method == "ease_in":
        return t * t
    elif method == "ease_out":
        return 1 - (1 - t) * (1 - t)
    elif method == "ease_in_out":
        return 0.5 * (1 - math.cos(t * math.pi))
    else:  # Linear
        return t


# unified fps configuration node - this will serve as a central place to manage all FPS-related settings and (hopefully) reduce redundancy:


class WanVideoUnifiedFPSConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": (
                    ["cinema_24fps", "smooth_30fps", "high_quality_60fps", "custom"],
                    {
                        "default": "cinema_24fps",
                        "tooltip": "Predefined configurations for common video formats",
                    },
                ),
                "generation_fps": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.5,
                        "max": 30.0,
                        "step": 0.5,
                        "tooltip": "Frame rate for generation (lower = faster generation)",
                    },
                ),
                "output_fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 60.0,
                        "step": 1.0,
                        "tooltip": "Target frame rate of final video",
                    },
                ),
                "interpolation_method": (
                    ["linear", "slerp", "adaptive"],
                    {
                        "default": "linear",
                        "tooltip": "Method used to create intermediate frames",
                    },
                ),
                "enable_debugging": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Output additional debug visualizations",
                    },
                ),
            },
            "optional": {
                "advanced_config": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Show advanced configuration options",
                    },
                )
            },
        }

    RETURN_TYPES = ("WANVIDEOFPSCONFIG", "WANVIDCONTEXTFPS", "DICTIONARY")
    RETURN_NAMES = ("fps_config", "context_options", "debug_info")
    FUNCTION = "configure"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Universal FPS configuration for WanVideo workflows. Centralizes all timing parameters and ensures consistency."

    def configure(
        self,
        preset,
        generation_fps,
        output_fps,
        interpolation_method,
        enable_debugging,
        advanced_config=False,
    ):
        # Apply presets if not custom
        if preset == "cinema_24fps":
            generation_fps = 2.0
            output_fps = 24.0
            interpolation_method = "linear"
        elif preset == "smooth_30fps":
            generation_fps = 3.0
            output_fps = 30.0
            interpolation_method = "slerp"
        elif preset == "high_quality_60fps":
            generation_fps = 4.0
            output_fps = 60.0
            interpolation_method = "adaptive"

        # Calculate interpolation factor automatically
        interpolation_factor = max(1, round(output_fps / generation_fps))
        effective_output_fps = generation_fps * interpolation_factor

        # Create unified config
        config = {
            "generation_fps": generation_fps,
            "output_fps": output_fps,
            "source_fps": 30.0,  # Default reference
            "interpolation_factor": interpolation_factor,
            "effective_output_fps": effective_output_fps,
            "interpolation_method": interpolation_method,
            "context_frames": 81,
            "context_overlap": 16,
            "context_stride": 65,  # 81 - 16
            "latent_stride": 4,
            "latent_context_frames": 21,  # (81 - 1) // 4 + 1
            "latent_context_overlap": 4,  # 16 // 4
            "latent_context_stride": 17,  # (81 - 16) // 4
            "enable_frame_blending": True,
            "enable_debugging": enable_debugging,
        }

        # Create context options format for backward compatibility
        context_options = {
            "context_schedule": "uniform_standard_fps",
            "context_frames": config["context_frames"],
            "context_stride": config["context_stride"],
            "context_overlap": config["context_overlap"],
            "freenoise": True,
            "target_fps": generation_fps,
            "original_fps": 30.0,
            "output_fps": output_fps,
            "effective_output_fps": effective_output_fps,
            "recommended_interp_factor": interpolation_factor,
            "enable_frame_blending": True,
            "latent_stride": config["latent_stride"],
            "latent_context_frames": config["latent_context_frames"],
            "latent_context_overlap": config["latent_context_overlap"],
            "latent_context_stride": config["latent_context_stride"],
        }

        # Debug info dictionary with additional metadata
        debug_info = {
            "preset": preset,
            "generation_fps": generation_fps,
            "output_fps": output_fps,
            "interpolation_factor": interpolation_factor,
            "effective_output_fps": effective_output_fps,
            "interpolation_method": interpolation_method,
            "advanced_config_enabled": advanced_config,
            "config_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        log.info(f"Configured WanVideo with preset: {preset}")
        log.info(
            f"Generation FPS: {generation_fps}, Output FPS: {output_fps}, Interpolation Factor: {interpolation_factor}"
        )
        log.info(f"Effective output FPS will be: {effective_output_fps}")

        return (config, context_options, debug_info)


# interpolation debugger - this will provide visuals and data into what's happening during the interpolation process:


class WanVideoInterpolationDebugger:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_latents": ("LATENT",),
                "interpolated_latents": ("LATENT", {"default": None}),
                "fps_config": ("WANVIDEOFPSCONFIG",),
                "method": (["linear", "slerp", "adaptive"], {"default": "linear"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("debug_visualization",)
    FUNCTION = "visualize"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Creates visual debugging information for latent interpolation to help diagnose issues."

    def visualize(
        self,
        original_latents,
        interpolated_latents=None,
        fps_config=None,
        method="linear",
    ):
        """
        Create a visualization showing original frames, interpolated frames, and difference metrics
        """
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")  # Use Agg backend
        import numpy as np
        import torch
        import io
        from PIL import Image

        # Get shapes
        orig_shape = original_latents["samples"].shape
        interp_shape = (
            interpolated_latents["samples"].shape
            if interpolated_latents is not None
            else None
        )

        # Create figure
        fig = plt.figure(figsize=(14, 10))

        # Original latents info
        plt.subplot(4, 1, 1)
        plt.title(f"Original Latents: {orig_shape} ({orig_shape[2]} frames)")

        # Display frame distribution
        frames = orig_shape[2]
        plt.bar(range(frames), [1] * frames, color="blue", alpha=0.7)
        plt.xlabel("Frame Index")
        plt.ylabel("Keyframe")

        # Interpolated latents info (if available)
        if interpolated_latents is not None:
            plt.subplot(4, 1, 2)
            interp_factor = fps_config.get("interpolation_factor", 1)
            plt.title(
                f"Interpolated Latents: {interp_shape} ({interp_shape[2]} frames, {interp_factor}x interpolation)"
            )

            # Mark original frames vs interpolated frames
            interp_frames = interp_shape[2]

            colors = [
                "blue" if i % interp_factor == 0 else "green"
                for i in range(interp_frames)
            ]
            plt.bar(range(interp_frames), [1] * interp_frames, color=colors, alpha=0.7)

            # Add legend
            plt.bar(0, 0, color="blue", alpha=0.7, label="Original Keyframe")
            plt.bar(0, 0, color="green", alpha=0.7, label="Interpolated Frame")
            plt.legend()

            # Frame-to-frame difference analysis
            plt.subplot(4, 1, 3)
            plt.title("Frame-to-Frame Difference")

            # Calculate frame differences
            latents = interpolated_latents["samples"][0]  # [c, t, h, w]
            diffs = []

            for i in range(1, latents.shape[1]):
                # Calculate difference between consecutive frames
                diff = (
                    torch.abs(latents[:, i, :, :] - latents[:, i - 1, :, :])
                    .mean()
                    .item()
                )
                diffs.append(diff)

            plt.plot(range(1, len(diffs) + 1), diffs)
            plt.axhline(
                y=np.mean(diffs),
                color="r",
                linestyle="--",
                label=f"Mean: {np.mean(diffs):.4f}",
            )

            # Highlight keyframe transitions
            for i in range(0, interp_frames, interp_factor):
                if i > 0 and i < len(diffs) + 1:
                    plt.axvline(x=i, color="blue", alpha=0.3)

            plt.xlabel("Frame Index")
            plt.ylabel("Mean Absolute Difference")
            plt.legend()

            # Validation metrics
            plt.subplot(4, 1, 4)
            plt.title("Validation Metrics")
            plt.axis("off")

            # Calculate validation metrics
            expected_frames = (orig_shape[2] - 1) * interp_factor + 1
            contains_nan = torch.isnan(interpolated_latents["samples"]).any().item()
            min_val = interpolated_latents["samples"].min().item()
            max_val = interpolated_latents["samples"].max().item()
            mean_val = interpolated_latents["samples"].mean().item()
            std_val = interpolated_latents["samples"].std().item()

            # Create text box with validation metrics
            validation_text = (
                f"Method: {method}\n"
                f"Expected frames: {expected_frames}, Actual frames: {interp_shape[2]}\n"
                f"Frame count match: {'✓' if expected_frames == interp_shape[2] else '✗'}\n"
                f"Contains NaN values: {'✗' if contains_nan else '✓'}\n"
                f"Min value: {min_val:.4f}, Max value: {max_val:.4f}\n"
                f"Mean: {mean_val:.4f}, Std: {std_val:.4f}\n"
                f"Value range looks normal: {'✓' if -5 < min_val < 5 and -5 < max_val < 5 else '✗'}"
            )

            plt.text(
                0.5,
                0.5,
                validation_text,
                fontsize=10,
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
            )

        plt.tight_layout()

        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        img = Image.open(buf)
        plt_tensor = torch.from_numpy(np.array(img)).float() / 255.0

        # Clean up
        plt.close(fig)

        return (plt_tensor,)


class WanVideoFPSController:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "creation_mode": (
                    ["quality", "balanced", "performance"],
                    {
                        "default": "balanced",
                        "tooltip": "Balances between speed and quality",
                    },
                ),
                "output_fps": (
                    "INT",
                    {
                        "default": 24,
                        "min": 1,
                        "max": 60,
                        "step": 1,
                        "tooltip": "Target frame rate for final video",
                    },
                ),
                "style": (
                    ["cinematic", "animation", "timelapse", "slow-motion", "custom"],
                    {
                        "default": "cinematic",
                        "tooltip": "Preset generation profiles for common video styles",
                    },
                ),
            },
            "optional": {
                "custom_generation_fps": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 0.5,
                        "max": 30.0,
                        "step": 0.5,
                        "tooltip": "Only used when style=custom",
                    },
                ),
            },
        }

    RETURN_TYPES = ("WANVIDEOFPSCONTROLLER",)
    FUNCTION = "configure"
    CATEGORY = "WanVideoWrapper"

    def configure(self, creation_mode, output_fps, style, custom_generation_fps=None):
        # Preset configurations based on common workflows
        presets = {
            "cinematic": {"generation_fps": 4.0, "motion_smoothness": 0.7},
            "animation": {"generation_fps": 6.0, "motion_smoothness": 0.3},
            "timelapse": {"generation_fps": 8.0, "motion_smoothness": 0.2},
            "slow-motion": {"generation_fps": 1.0, "motion_smoothness": 0.8},
            "custom": {
                "generation_fps": custom_generation_fps,
                "motion_smoothness": 0.5,
            },
        }

        # Performance presets
        performance_multipliers = {
            "quality": 0.5,  # Generate more keyframes
            "balanced": 1.0,  # Use preset defaults
            "performance": 2.0,  # Generate fewer keyframes
        }

        # Get base configuration from style
        config = presets[style].copy()

        # Apply performance multiplier
        base_fps = config["generation_fps"]
        config["generation_fps"] = max(
            0.5, min(30, base_fps * performance_multipliers[creation_mode])
        )

        # Calculate interpolation factor automatically
        config["interpolation_factor"] = max(
            1, round(output_fps / config["generation_fps"])
        )
        config["output_fps"] = output_fps

        # Calculate effective output fps (what will actually be produced)
        config["effective_fps"] = (
            config["generation_fps"] * config["interpolation_factor"]
        )

        # Log clear information about what's happening
        log.info("\n=== WanVideo FPS Controller ===")
        log.info(f"Style: {style}")
        log.info(f"Mode: {creation_mode}")
        log.info(f"Generating at: {config['generation_fps']:.1f} FPS")
        log.info(f"Interpolating by factor: {config['interpolation_factor']}x")
        log.info(f"Output frame rate: {output_fps} FPS")

        if abs(config["effective_fps"] - output_fps) > 0.1:
            log.info(
                f"Note: Effective output will be {config['effective_fps']:.1f} FPS due to interpolation factor"
            )

        # Calculate generation time estimates
        log.info(
            f"You'll generate ~{config['generation_fps']:.1f} frames per second of video"
        )
        log.info(
            f"Each second of video will require ~{1.0 / config['generation_fps']:.1f}x the generation time"
        )
        log.info("===========================\n")

        return (config,)


class WanVideoFPSConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "generation_fps": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 0.5,
                        "max": 30.0,
                        "step": 0.5,
                        "tooltip": "Frame rate at which to generate video (lower = faster generation)",
                    },
                ),
                "output_fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 60.0,
                        "step": 1.0,
                        "tooltip": "Final video frame rate after interpolation",
                    },
                ),
                "source_fps": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 1.0,
                        "max": 60.0,
                        "step": 1.0,
                        "tooltip": "Reference FPS for time calculations (usually 30)",
                    },
                ),
                "context_frames": (
                    "INT",
                    {
                        "default": 81,
                        "min": 16,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Number of frames in each context window (pixel space)",
                    },
                ),
                "context_overlap": (
                    "INT",
                    {
                        "default": 16,
                        "min": 4,
                        "max": 100,
                        "step": 4,
                        "tooltip": "Overlap between context windows (pixel space)",
                    },
                ),
                "enable_frame_blending": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Apply temporal blending between context windows",
                    },
                ),
            }
        }

    RETURN_TYPES = ("WANVIDEOFPSCONFIG",)
    RETURN_NAMES = ("fps_config",)
    FUNCTION = "configure"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Central FPS configuration for WanVideo workflow. Calculates all derived timing values and ensures consistency."

    def configure(
        self,
        generation_fps,
        output_fps,
        source_fps,
        context_frames,
        context_overlap,
        enable_frame_blending,
    ):
        # Calculate interpolation factor automatically
        interpolation_factor = max(1, round(output_fps / generation_fps))

        # Round the effective output FPS to the nearest integer for display purposes
        effective_output_fps = generation_fps * interpolation_factor

        # Validate settings are compatible
        if abs(effective_output_fps - output_fps) > 0.1:
            log.warning(
                f"Warning: output_fps ({output_fps}) is not exactly divisible by generation_fps ({generation_fps})"
            )
            log.warning(
                f"Using interpolation_factor={interpolation_factor}, effective output FPS will be {effective_output_fps:.1f}"
            )

        # Adjust for latent space (WanVideo uses 4 pixel frames per latent frame)
        latent_stride = 4
        latent_context_frames = (context_frames - 1) // latent_stride + 1
        latent_context_overlap = max(1, context_overlap // latent_stride)

        # Calculate context stride based on context frames and overlap
        context_stride = context_frames - context_overlap
        latent_context_stride = max(1, context_stride // latent_stride)

        # Compute video duration parameters
        frame_duration_sec = 1.0 / source_fps
        seconds_per_latent_frame = latent_stride * frame_duration_sec

        log.info(f"FPS Configuration Summary:")
        log.info(
            f"  Generation: {generation_fps} FPS → Output: {effective_output_fps:.1f} FPS (x{interpolation_factor})"
        )
        log.info(
            f"  Context: {latent_context_frames} latent frames ({context_frames} pixel frames)"
        )
        log.info(
            f"  Overlap: {latent_context_overlap} latent frames ({context_overlap} pixel frames)"
        )
        log.info(
            f"  Frame blending: {'Enabled' if enable_frame_blending else 'Disabled'}"
        )

        config = {
            "generation_fps": generation_fps,
            "output_fps": output_fps,
            "effective_output_fps": effective_output_fps,
            "source_fps": source_fps,
            "interpolation_factor": interpolation_factor,
            "context_frames": context_frames,
            "context_overlap": context_overlap,
            "context_stride": context_stride,
            "latent_stride": latent_stride,
            "latent_context_frames": latent_context_frames,
            "latent_context_overlap": latent_context_overlap,
            "latent_context_stride": latent_context_stride,
            "seconds_per_latent_frame": seconds_per_latent_frame,
            "frame_duration_sec": frame_duration_sec,
            "enable_frame_blending": enable_frame_blending,
        }

        # Create context_options format for backward compatibility
        context_options = {
            "context_schedule": "uniform_standard_fps",
            "context_frames": context_frames,
            "context_stride": context_stride,
            "context_overlap": context_overlap,
            "freenoise": True,
            "target_fps": generation_fps,
            "original_fps": source_fps,
            "output_fps": output_fps,
            "effective_output_fps": effective_output_fps,
            "recommended_interp_factor": interpolation_factor,
            "enable_frame_blending": enable_frame_blending,
            "latent_stride": latent_stride,
            "latent_context_frames": latent_context_frames,
            "latent_context_overlap": latent_context_overlap,
            "latent_context_stride": latent_context_stride,
        }

        return (config, context_options)


class WanVideoContextOptionsFPS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context_schedule": (
                    [
                        "uniform_standard_fps",
                        "uniform_standard",
                        "uniform_looped",
                        "static_standard",
                    ],
                    {"default": "uniform_standard_fps"},
                ),
                "context_frames": ("INT", {"default": 81, "min": 2, "max": 1000}),
                "context_stride": ("INT", {"default": 4, "min": 4, "max": 100}),
                "context_overlap": ("INT", {"default": 16, "min": 4, "max": 100}),
                "freenoise": ("BOOLEAN", {"default": True}),
                "target_fps": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 30.0}),
                "original_fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 60.0}),
                "output_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0}),
            },
            "optional": {
                "fps_config": ("WANVIDEOFPSCONFIG", {"default": None}),
            },
        }

    RETURN_TYPES = ("WANVIDCONTEXTFPS",)
    RETURN_NAMES = ("context_options_fps",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Context options for WanVideo with FPS-based sampling as recommended in the Apollo paper. Allows splitting the video into context windows with consistent temporal sampling."

    def process(
        self,
        context_schedule,
        context_frames,
        context_stride,
        context_overlap,
        freenoise,
        target_fps,
        original_fps,
        output_fps,
        enable_frame_blending=True,
        fps_config=None,
    ):
        # If fps_config is provided, use its values instead
        if fps_config is not None:
            log.info("Using values from FPS config")
            context_frames = fps_config["context_frames"]
            context_overlap = fps_config["context_overlap"]
            context_stride = fps_config["context_stride"]
            target_fps = fps_config["generation_fps"]
            original_fps = fps_config["source_fps"]
            output_fps = fps_config["output_fps"]

            # Also get advanced values if present
            enable_frame_blending = fps_config.get(
                "enable_frame_blending", enable_frame_blending
            )
            recommended_interp_factor = fps_config.get(
                "interpolation_factor", max(1, round(output_fps / target_fps))
            )
            effective_output_fps = fps_config.get(
                "effective_output_fps", target_fps * recommended_interp_factor
            )
            latent_stride = fps_config.get("latent_stride", 4)
            latent_context_frames = fps_config.get(
                "latent_context_frames", (context_frames - 1) // latent_stride + 1
            )
            latent_context_overlap = fps_config.get(
                "latent_context_overlap", context_overlap // latent_stride
            )
            latent_context_stride = fps_config.get(
                "latent_context_stride",
                max(1, (context_frames - context_overlap) // latent_stride),
            )
        else:
            # Calculate derived values
            recommended_interp_factor = max(1, round(output_fps / target_fps))
            effective_output_fps = target_fps * recommended_interp_factor

            # Log recommendations
            log.info(f"Recommended interpolation factor: {recommended_interp_factor}")
            log.info(f"Effective output FPS: {effective_output_fps:.1f}")

            # Calculate latent space values
            latent_stride = 4  # WanVideo's latent stride
            latent_context_frames = (context_frames - 1) // latent_stride + 1
            latent_context_overlap = max(1, context_overlap // latent_stride)
            latent_context_stride = max(
                1, (context_frames - context_overlap) // latent_stride
            )

        # Create the context options dictionary
        context_options = {
            "context_schedule": context_schedule,
            "context_frames": context_frames,
            "context_stride": context_stride,
            "context_overlap": context_overlap,
            "freenoise": freenoise,
            "target_fps": target_fps,
            "original_fps": original_fps,
            "output_fps": output_fps,
            "effective_output_fps": effective_output_fps,
            "recommended_interp_factor": recommended_interp_factor,
            "enable_frame_blending": enable_frame_blending,
            "latent_stride": latent_stride,
            "latent_context_frames": latent_context_frames,
            "latent_context_overlap": latent_context_overlap,
            "latent_context_stride": latent_context_stride,
        }

        return (context_options,)


# sampler for fps sampling - used so that we don't cross wires with the sampler in nodes.py


class WanVideoEnhancedFPSSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "image_embeds": ("WANVIDIMAGE_EMBEDS",),
                "steps": ("INT", {"default": 30, "min": 1}),
                "cfg": (
                    "FLOAT",
                    {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01},
                ),
                "shift": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "scheduler": (
                    ["unipc", "dpm++", "dpm++_sde", "euler"],
                    {"default": "unipc"},
                ),
                "force_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "samples": (
                    "LATENT",
                    {"tooltip": "Init latents for video2video process"},
                ),
                "denoise_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "feta_args": ("FETAARGS",),
                "unified_config": (
                    "WANVIDEOFPSCONFIG",
                    {"tooltip": "Unified FPS configuration"},
                ),
                "teacache_args": ("TEACACHEARGS",),
                "temporal_emphasis": (
                    "TEMPORALEMPHASIS",
                    {
                        "tooltip": "Controls balance between temporal and spatial features"
                    },
                ),
                "perceiver_config": (
                    "PERCEIVERCONFIG",
                    {"tooltip": "Configuration for the Perceiver Resampler"},
                ),
                "enable_debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("samples", "debug_info")
    FUNCTION = "sample"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = (
        "Enhanced FPS-based WanVideo sampler with improved configuration and debugging"
    )

    def sample(
        self,
        model,
        text_embeds,
        image_embeds,
        steps,
        cfg,
        shift,
        seed,
        scheduler,
        force_offload=True,
        samples=None,
        denoise_strength=1.0,
        feta_args=None,
        unified_config=None,
        teacache_args=None,
        temporal_emphasis=None,
        perceiver_config=None,
        enable_debug=False,
    ):
        # Start timing for performance measurement
        import time

        start_time = time.time()

        patcher = model
        model = model.model
        transformer = model.diffusion_model

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        # Adjust steps based on denoise strength
        steps = int(steps / denoise_strength)

        # Set up sampling configuration
        config = {
            "sampling_mode": "fps",
            "generation_fps": 2.0,  # Default
            "original_fps": 30.0,  # Default
            "enable_frame_blending": True,
        }

        # Override with unified config if provided
        if unified_config is not None:
            for key, value in unified_config.items():
                config[key] = value

            log.info(
                f"Using unified config with generation FPS: {config['generation_fps']}"
            )

        # Set up the scheduler
        timesteps, sample_scheduler = self._setup_scheduler(
            scheduler, shift, steps, device, denoise_strength
        )

        # Initialize debug info collector
        debug_data = {
            "config": config,
            "timesteps": len(timesteps),
            "latent_shapes": [],
            "scheduler": scheduler,
            "generation_time": 0,
            "step_times": [],
        }

        # Initialize latents
        latent, latent_video_length = self._initialize_latents(
            image_embeds, samples, timesteps, seed, device
        )

        debug_data["latent_shapes"].append(latent.shape)
        debug_data["latent_video_length"] = latent_video_length

        # Set up frequency parameters for rotary embeddings
        freqs = self._setup_frequencies(
            transformer,
            latent_video_length,
            config.get("riflex_freq_index", 0),
            device,
        )

        # Prepare arguments for model forward pass
        base_args, arg_c, arg_null = self._prepare_model_args(
            transformer,
            text_embeds,
            image_embeds,
            freqs,
            device,
            temporal_emphasis,
            perceiver_config,
        )

        # Set up FEtA (Enhance-A-Video) if provided
        self._setup_feta(feta_args, config)

        # Set up block swapping for memory optimization if specified
        self._setup_block_swapping(model, transformer, device, offload_device)

        # Set up TeaCache for caching previous results if enabled
        self._setup_teacache(transformer, teacache_args)

        # Set up context windows based on config if using FPS sampling
        context_windows = None
        if config.get("sampling_mode") == "fps":
            from .context import get_context_scheduler

            context_scheduler = get_context_scheduler(
                config.get("context_schedule", "uniform_standard_fps")
            )

            context_windows = list(
                context_scheduler(
                    0,  # step
                    None,  # num_steps
                    latent_video_length,
                    config.get("latent_context_frames", 21),
                    config.get("latent_context_stride", 17),
                    config.get("latent_context_overlap", 4),
                    config.get("generation_fps", 2.0),
                    config.get("original_fps", 30.0),
                    True,  # closed_loop
                    config.get("enable_frame_blending", True),
                )
            )

            debug_data["context_windows"] = [list(window) for window in context_windows]
            debug_data["context_window_count"] = len(context_windows)

            # Log context windows info
            log.info(
                f"Created {len(context_windows)} context windows for {latent_video_length} frames"
            )

        # Set up progress tracking
        from comfy.utils import ProgressBar

        pbar = ProgressBar(steps)

        # Set up callback for preview
        from latent_preview import prepare_callback

        callback = prepare_callback(patcher, steps)

        # Start sampling
        log.info(
            f"Sampling {(latent_video_length - 1) * 4 + 1} frames at {latent.shape[3] * 8}x{latent.shape[2] * 8}"
        )
        log.info(f"Using {config.get('generation_fps', 2.0)} FPS for generation")

        # Initialize tracking for TeaCache
        intermediate_device = device

        # Prepare window indices tracker for blending
        window_indices = {}
        blend_enabled = config.get("enable_frame_blending", True)

        # Main sampling loop
        with torch.autocast(
            device_type=mm.get_autocast_device(device),
            dtype=model["dtype"],
            enabled=True,
        ):
            for i, t in enumerate(tqdm(timesteps)):
                step_start = time.time()

                # Prepare inputs
                latent_model_input = [latent.to(device)]
                timestep = torch.tensor([t], device=device)

                # Calculate step percentage for FEtA
                current_step_percentage = i / len(timesteps)

                # Handle FEtA enable/disable based on step percentage
                if feta_args is not None:
                    self._update_feta(feta_args, current_step_percentage)

                # Process by context windows if enabled
                if context_windows:
                    # Process using context windows
                    noise_pred = self._process_by_windows(
                        context_windows,
                        latent_model_input[0],
                        transformer,
                        timestep,
                        i,
                        t,
                        cfg if not isinstance(cfg, list) else cfg[i],
                        arg_c,
                        arg_null,
                        intermediate_device,
                        blend_enabled,
                        window_indices,
                    )
                else:
                    # Standard processing without windows
                    noise_pred = self._process_standard(
                        latent_model_input,
                        transformer,
                        timestep,
                        i,
                        t,
                        cfg if not isinstance(cfg, list) else cfg[i],
                        arg_c,
                        arg_null,
                        intermediate_device,
                        teacache_args is not None,
                    )

                # Move latent to intermediate device for scheduler step
                latent = latent.to(intermediate_device)

                # Step the scheduler
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=torch.Generator(device=torch.device("cpu")).manual_seed(
                        seed
                    ),
                )[0]

                # Update latent
                latent = temp_x0.squeeze(0)

                # Prepare next iteration
                x0 = [latent.to(device)]

                # Update progress display
                if callback is not None:
                    callback_latent = (
                        (latent_model_input[0] - noise_pred.to(t.device) * t / 1000)
                        .detach()
                        .permute(1, 0, 2, 3)
                    )
                    callback(i, callback_latent, None, steps)
                else:
                    pbar.update(1)

                # Record step time
                step_time = time.time() - step_start
                debug_data["step_times"].append(step_time)

                # Clean up to avoid OOM
                del latent_model_input, timestep

        # Record total generation time
        generation_time = time.time() - start_time
        debug_data["generation_time"] = generation_time

        # Report TeaCache stats if used
        if hasattr(transformer, "teacache_skipped_cond_steps"):
            log.info(
                f"TeaCache skipped: {transformer.teacache_skipped_cond_steps} cond steps, "
                f"{transformer.teacache_skipped_uncond_steps} uncond steps"
            )
            debug_data["teacache_skipped_cond"] = (
                transformer.teacache_skipped_cond_steps
            )
            debug_data["teacache_skipped_uncond"] = (
                transformer.teacache_skipped_uncond_steps
            )

        # Clean up and prepare to return results
        if force_offload:
            if model["manual_offloading"]:
                transformer.to(offload_device)
                mm.soft_empty_cache()
                gc.collect()

        # Create debug visualization if enabled
        debug_image = torch.zeros((100, 100, 3))
        if enable_debug:
            try:
                debug_image = self._create_debug_visualization(debug_data)
            except Exception as e:
                log.warning(f"Failed to create debug visualization: {e}")

        log.info(f"Generation complete in {generation_time:.2f} seconds")

        return ({"samples": x0[0].unsqueeze(0).cpu()}, debug_image)

    def _setup_scheduler(self, scheduler_type, shift, steps, device, denoise_strength):
        """Set up the diffusion scheduler"""
        if scheduler_type == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False
            )
            sample_scheduler.set_timesteps(steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif scheduler_type == "euler":
            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False
            )
            sampling_sigmas = get_sampling_sigmas(steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler, device=device, sigmas=sampling_sigmas
            )
        elif "dpm++" in scheduler_type:
            algorithm_type = (
                "sde-dpmsolver++" if scheduler_type == "dpm++_sde" else "dpmsolver++"
            )
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                shift=shift,
                use_dynamic_shifting=False,
                algorithm_type=algorithm_type,
            )
            sampling_sigmas = get_sampling_sigmas(steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler, device=device, sigmas=sampling_sigmas
            )
        else:
            raise NotImplementedError(f"Unsupported scheduler: {scheduler_type}")

        # Adjust timesteps for denoise_strength
        if denoise_strength < 1.0:
            steps = int(steps * denoise_strength)
            timesteps = timesteps[-(steps + 1) :]

        return timesteps, sample_scheduler

    def _initialize_latents(self, image_embeds, samples, timesteps, seed, device):
        """Initialize latent tensors for sampling"""
        seed_g = torch.Generator(device=torch.device("cpu"))
        seed_g.manual_seed(seed)

        # Handle different image_embeds formats
        if image_embeds.get("target_shape", None) is not None:
            # For T2V (text to video) mode
            target_shape = image_embeds["target_shape"]
            seq_len = image_embeds["max_seq_len"]

            # Generate noise based on target shape
            noise = torch.randn(
                *target_shape,
                dtype=torch.float32,
                device=torch.device("cpu"),
                generator=seed_g,
            )

            # Determine latent video length based on tensor dimensions
            latent_video_length = noise.shape[1] if noise.dim() == 4 else noise.shape[2]

        else:
            # For I2V (image to video) mode
            lat_h = image_embeds.get("lat_h")
            lat_w = image_embeds.get("lat_w")
            seq_len = image_embeds["max_seq_len"]

            # Default to 4D tensor for I2V mode based on WanVideo implementation
            noise = torch.randn(
                16,  # channels
                (image_embeds["num_frames"] - 1) // 4 + 1,  # time
                lat_h,  # height
                lat_w,  # width
                dtype=torch.float32,
                generator=seed_g,
                device=torch.device("cpu"),
            )
            latent_video_length = noise.shape[1]

        # Initialize from input samples if provided (for video2video)
        if samples is not None:
            latent_timestep = timesteps[:1].to(noise.device)
            sample_tensor = samples["samples"]

            # Make sure sample_tensor has the same shape as noise
            if sample_tensor.dim() != noise.dim():
                if noise.dim() == 4 and sample_tensor.dim() == 5:
                    sample_tensor = sample_tensor.squeeze(0)
                elif noise.dim() == 5 and sample_tensor.dim() == 4:
                    sample_tensor = sample_tensor.unsqueeze(0)

            # Apply noise with denoise strength via timestep
            noise = noise * latent_timestep / 1000 + (
                1 - latent_timestep / 1000
            ) * sample_tensor.to(noise.device)

        return noise.to(device), latent_video_length

    def _setup_frequencies(
        self, transformer, latent_video_length, riflex_freq_index, device
    ):
        """Set up frequency parameters for rotary embeddings"""
        from .wanvideo.modules.model import rope_params

        d = transformer.dim // transformer.num_heads
        freqs = torch.cat(
            [
                rope_params(
                    1024,
                    d - 4 * (d // 6),
                    L_test=latent_video_length,
                    k=riflex_freq_index,
                ),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        return freqs.to(device)

    def _prepare_model_args(
        self,
        transformer,
        text_embeds,
        image_embeds,
        freqs,
        device,
        temporal_emphasis,
        perceiver_config,
    ):
        """Prepare arguments for model forward passes"""
        # Base arguments for both conditional and unconditional paths
        base_args = {
            "clip_fea": image_embeds.get("clip_context", None),
            "seq_len": image_embeds.get("max_seq_len"),
            "device": device,
            "freqs": freqs,
        }

        # Add model-specific args for I2V (image to video) mode
        if transformer.model_type == "i2v" and "image_embeds" in image_embeds:
            base_args.update({"y": [image_embeds["image_embeds"]]})

        # Add temporal emphasis if provided
        if temporal_emphasis is not None:
            base_args.update({"temporal_emphasis": temporal_emphasis})

        # Add perceiver config if provided
        if perceiver_config is not None:
            base_args.update({"perceiver_config": perceiver_config})

        # Conditional and unconditional paths
        arg_c = base_args.copy()
        arg_c.update({"context": [text_embeds["prompt_embeds"][0]]})

        arg_null = base_args.copy()
        arg_null.update({"context": text_embeds["negative_prompt_embeds"]})

        return base_args, arg_c, arg_null

    def _setup_feta(self, feta_args, config):
        """Set up Enhance-A-Video if enabled"""
        from .enhance_a_video.globals import (
            enable_enhance,
            disable_enhance,
            set_enhance_weight,
            set_num_frames,
        )

        if feta_args is not None:
            set_enhance_weight(feta_args["weight"])

            # Use context frames if available
            if "latent_context_frames" in config:
                set_num_frames(config["latent_context_frames"])
            else:
                # Default in original code
                set_num_frames(21)  # 81 // 4 + 1

            enable_enhance()
        else:
            disable_enhance()

    def _update_feta(self, feta_args, current_step_percentage):
        """Update FEtA status based on current step percentage"""
        from .enhance_a_video.globals import enable_enhance, disable_enhance

        if feta_args is not None:
            feta_start_percent = feta_args["start_percent"]
            feta_end_percent = feta_args["end_percent"]

            if feta_start_percent <= current_step_percentage <= feta_end_percent:
                enable_enhance()
            else:
                disable_enhance()

    def _setup_block_swapping(self, model, transformer, device, offload_device):
        """Set up block swapping for memory optimization if specified in model"""
        if model["block_swap_args"] is not None:
            for name, param in transformer.named_parameters():
                if "block" not in name:
                    param.data = param.data.to(device)
                elif model["block_swap_args"]["offload_txt_emb"] and "txt_emb" in name:
                    param.data = param.data.to(offload_device)
                elif model["block_swap_args"]["offload_img_emb"] and "img_emb" in name:
                    param.data = param.data.to(offload_device)

            transformer.block_swap(
                model["block_swap_args"]["blocks_to_swap"] - 1,
                model["block_swap_args"]["offload_txt_emb"],
                model["block_swap_args"]["offload_img_emb"],
            )
        elif model["auto_cpu_offload"]:
            for module in transformer.modules():
                if hasattr(module, "offload"):
                    module.offload()
                if hasattr(module, "onload"):
                    module.onload()
        elif model["manual_offloading"]:
            transformer.to(device)

    def _setup_teacache(self, transformer, teacache_args):
        """Set up TeaCache if enabled"""
        if teacache_args is not None:
            transformer.enable_teacache = True
            transformer.rel_l1_thresh = teacache_args["rel_l1_thresh"]
            transformer.teacache_start_step = teacache_args["start_step"]
            transformer.teacache_cache_device = teacache_args["cache_device"]

            # Initialize tracking attributes
            transformer.teacache_skipped_cond_steps = 0
            transformer.teacache_skipped_uncond_steps = 0
        else:
            transformer.enable_teacache = False

    def _process_by_windows(
        self,
        windows,
        latent,
        transformer,
        timestep,
        step_idx,
        t,
        cfg_scale,
        arg_c,
        arg_null,
        intermediate_device,
        blend_enabled,
        window_indices,
    ):
        """Process sampling in context windows with optional frame blending"""
        # Zero-initialize noise prediction and counter tensors
        noise_pred = torch.zeros_like(latent, device=intermediate_device)
        counter = torch.zeros_like(latent, device=intermediate_device)

        # Process each context window
        for window_idx, window_frames in enumerate(windows):
            # Get the latent for this window
            # Adjust indexing based on actual tensor shape
            if latent.dim() == 4:  # [c, t, h, w]
                partial_latent = latent[:, window_frames, :, :]
            else:  # [b, c, t, h, w]
                partial_latent = latent[:, :, window_frames, :, :]

            partial_model_input = [partial_latent]

            # Keep track of which frames belong to this window
            if blend_enabled:
                for frame_idx in window_frames:
                    if frame_idx not in window_indices:
                        window_indices[frame_idx] = []
                    window_indices[frame_idx].append(window_idx)

            # Perform conditional forward pass
            noise_pred_cond = transformer(
                partial_model_input,
                t=timestep,
                current_step=step_idx,
                is_uncond=False,
                **arg_c,
            )[0].to(intermediate_device)

            # Perform unconditional forward pass if CFG > 1.0
            if cfg_scale != 1.0:
                noise_pred_uncond = transformer(
                    partial_model_input,
                    t=timestep,
                    current_step=step_idx,
                    is_uncond=True,
                    **arg_null,
                )[0].to(intermediate_device)

                noise_pred_window = noise_pred_uncond + cfg_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                noise_pred_window = noise_pred_cond

            # Create blending mask based on position in the window
            window_mask = self._create_window_mask(
                noise_pred_window,
                window_frames,
                window_idx,
                len(windows),
                window_indices if blend_enabled else None,
            )

            # Apply the window's noise prediction to the global tensor
            for i, frame_idx in enumerate(window_frames):
                # Adjust indexing based on actual tensor shape
                if noise_pred.dim() == 4:  # [c, t, h, w]
                    noise_pred[:, frame_idx, :, :] += (
                        noise_pred_window[:, i, :, :] * window_mask[:, i, :, :]
                    )
                    counter[:, frame_idx, :, :] += window_mask[:, i, :, :]
                else:  # [b, c, t, h, w]
                    noise_pred[:, :, frame_idx, :, :] += (
                        noise_pred_window[:, :, i, :, :] * window_mask[:, :, i, :, :]
                    )
                    counter[:, :, frame_idx, :, :] += window_mask[:, :, i, :, :]

        # Normalize by counter to get weighted average
        valid_mask = counter > 0
        noise_pred[valid_mask] /= counter[valid_mask]

        return noise_pred

    def _create_window_mask(
        self,
        noise_pred,
        window_frames,
        window_idx,
        total_windows,
        window_indices=None,
    ):
        """Create blending mask for window transitions"""
        device = noise_pred.device

        # Create mask with appropriate dimensions
        window_mask = torch.ones_like(noise_pred)

        # Skip if not doing frame blending
        if window_indices is None:
            return window_mask

        # Skip if this is the only window
        if total_windows <= 1:
            return window_mask

        # Apply left-side blending for all except first window with small overlap
        if window_idx > 0:
            # Find frames that are in both this window and the previous one
            overlap_idx = 0
            overlap_count = 0

            for i, frame_idx in enumerate(window_frames):
                if len(window_indices.get(frame_idx, [])) > 1:
                    # This frame belongs to multiple windows
                    overlap_count += 1

                    # Create gradual blend weights
                    if overlap_count > 0:
                        # Ease-in weight (0.0 -> 1.0)
                        t = overlap_idx / max(1, overlap_count - 1)
                        weight = min(1.0, 0.5 * (1.0 - math.cos(math.pi * t)))

                        # Apply weight to the correct dimensions
                        if window_mask.dim() == 4:  # [c, t, h, w]
                            window_mask[:, i, :, :] = weight
                        else:  # [b, c, t, h, w]
                            window_mask[:, :, i, :, :] = weight

                        overlap_idx += 1

        # Apply right-side blending for all except last window
        if window_idx < total_windows - 1:
            # Find frames that are in both this window and the next one
            overlap_frames = []

            for i, frame_idx in enumerate(reversed(window_frames)):
                if len(window_indices.get(frame_idx, [])) > 1:
                    overlap_frames.append((len(window_frames) - 1 - i, frame_idx))

            # Create gradual blend weights for these frames
            for idx, (i, frame_idx) in enumerate(overlap_frames):
                # Ease-out weight (1.0 -> 0.0)
                if len(overlap_frames) > 0:
                    t = idx / max(1, len(overlap_frames) - 1)
                    weight = min(1.0, 0.5 * (1.0 + math.cos(math.pi * t)))

                    # Apply weight to the correct dimensions
                    if window_mask.dim() == 4:  # [c, t, h, w]
                        window_mask[:, i, :, :] = weight
                    else:  # [b, c, t, h, w]
                        window_mask[:, :, i, :, :] = weight

        return window_mask

    def _process_standard(
        self,
        latent_model_input,
        transformer,
        timestep,
        step_idx,
        t,
        cfg_scale,
        arg_c,
        arg_null,
        intermediate_device,
        use_teacache,
    ):
        """Process sampling without context windows"""
        # Do TeaCache checking first if enabled
        should_calc = True
        if use_teacache and step_idx >= transformer.teacache_start_step:
            if hasattr(transformer, "previous_residual_uncond") and hasattr(
                transformer, "previous_residual_cond"
            ):
                # TeaCache logic would go here if implemented
                pass

        # Normal processing path
        if not use_teacache or should_calc:
            # Conditional forward pass
            noise_pred_cond = transformer(
                latent_model_input,
                t=timestep,
                current_step=step_idx,
                is_uncond=False,
                **arg_c,
            )[0].to(intermediate_device)

            # Unconditional forward pass if CFG > 1.0
            if cfg_scale != 1.0:
                noise_pred_uncond = transformer(
                    latent_model_input,
                    t=timestep,
                    current_step=step_idx,
                    is_uncond=True,
                    **arg_null,
                )[0].to(intermediate_device)

                noise_pred = noise_pred_uncond + cfg_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                noise_pred = noise_pred_cond

        return noise_pred


def _create_debug_visualization(self, debug_data):
    """Create a visualization of debug information"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import io
        from PIL import Image

        fig = plt.figure(figsize=(14, 10))
        plt.suptitle("WanVideo FPS Sampler Debug Information", fontsize=16)

        # Configuration summary
        plt.subplot(3, 1, 1)
        plt.axis("off")
        config_str = "\n".join(
            [
                f"Configuration Summary:",
                f"  Generation FPS: {debug_data['config'].get('generation_fps', 'N/A')}",
                f"  Scheduler: {debug_data['scheduler']}",
                f"  Total steps: {debug_data['timesteps']}",
                f"  Generation time: {debug_data['generation_time']:.2f} seconds",
                f"  Avg step time: {np.mean(debug_data['step_times']):.4f} seconds",
                f"  Video length: {debug_data.get('latent_video_length', 'N/A')} latent frames",
                f"  Context windows: {debug_data.get('context_window_count', 'N/A')}",
            ]
        )

        plt.text(0.1, 0.5, config_str, fontsize=12, ha="left", va="center")

        # Step timing information
        plt.subplot(3, 1, 2)
        plt.title("Step Times")
        if debug_data["step_times"]:
            plt.plot(debug_data["step_times"])
            plt.axhline(
                y=np.mean(debug_data["step_times"]),
                color="r",
                linestyle="--",
                label=f"Mean: {np.mean(debug_data['step_times']):.4f}s",
            )
            plt.xlabel("Step")
            plt.ylabel("Time (seconds)")
            plt.legend()
        else:
            plt.text(
                0.5, 0.5, "No step timing data available", ha="center", va="center"
            )

        # Context windows visualization if available
        plt.subplot(3, 1, 3)
        plt.title("Context Windows")

        if "context_windows" in debug_data:
            windows = debug_data["context_windows"]
            latent_length = debug_data.get("latent_video_length", 0)

            # Create a matrix showing which frames belong to which windows
            window_matrix = np.zeros((len(windows), latent_length))

            for i, window in enumerate(windows):
                for frame_idx in window:
                    if frame_idx < latent_length:
                        window_matrix[i, frame_idx] = 1

            plt.imshow(window_matrix, aspect="auto", cmap="viridis")
            plt.xlabel("Frame Index")
            plt.ylabel("Window Index")
            plt.colorbar(label="Frame Present")
        else:
            plt.text(
                0.5,
                0.5,
                "No context window data available",
                ha="center",
                va="center",
            )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        img = Image.open(buf)
        debug_image = torch.from_numpy(np.array(img)).float() / 255.0

        # Clean up
        plt.close(fig)

        return debug_image

    except Exception as e:
        log.warning(f"Error creating debug visualization: {e}")
        # Return placeholder image
        return torch.zeros((100, 400, 3))


# class WanVideoFPSSampler:
#     """
#     Apollo-inspired FPS-based video sampler for WanVideo.
#     Fully implements diffusion sampling with FPS-specific optimizations.
#     """

#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "model": ("WANVIDEOMODEL",),
#                 "text_embeds": ("WANVIDEOTEXTEMBEDS",),
#                 "image_embeds": ("WANVIDIMAGE_EMBEDS",),
#                 "steps": ("INT", {"default": 30, "min": 1}),
#                 "cfg": (
#                     "FLOAT",
#                     {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01},
#                 ),
#                 "shift": (
#                     "FLOAT",
#                     {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01},
#                 ),
#                 "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
#                 "scheduler": (
#                     ["unipc", "dpm++", "dpm++_sde", "euler"],
#                     {"default": "unipc"},
#                 ),
#                 "riflex_freq_index": (
#                     "INT",
#                     {
#                         "default": 0,
#                         "min": 0,
#                         "max": 1000,
#                         "step": 1,
#                         "tooltip": "Frequency index for RIFLEX, disabled when 0, default 4",
#                     },
#                 ),
#                 "force_offload": ("BOOLEAN", {"default": True}),
#                 "sampling_mode": (
#                     ["fps", "uniform"],
#                     {
#                         "default": "fps",
#                         "tooltip": "FPS sampling preserves temporal consistency, uniform sampling distributes frames evenly",
#                     },
#                 ),
#                 "generation_fps": (
#                     "FLOAT",
#                     {
#                         "default": 4.0,
#                         "min": 0.5,
#                         "max": 30.0,
#                         "step": 0.5,
#                         "tooltip": "Frame rate to generate video at - Apollo paper recommends 2-4 FPS",
#                     },
#                 ),
#                 "original_fps": (
#                     "FLOAT",
#                     {
#                         "default": 30.0,
#                         "min": 1.0,
#                         "max": 60.0,
#                         "step": 1.0,
#                         "tooltip": "Reference FPS for time calculations (usually 30)",
#                     },
#                 ),
#                 "output_fps": (
#                     "FLOAT",
#                     {
#                         "default": 24.0,
#                         "min": 1.0,
#                         "max": 60.0,
#                         "step": 1.0,
#                         "tooltip": "Final video frame rate after interpolation",
#                     },
#                 ),
#             },
#             "optional": {
#                 "samples": (
#                     "LATENT",
#                     {"tooltip": "Init latents for video2video process"},
#                 ),
#                 "denoise_strength": (
#                     "FLOAT",
#                     {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
#                 ),
#                 "feta_args": ("FETAARGS",),
#                 "context_options": (
#                     "WANVIDCONTEXTFPS",
#                     {"tooltip": "FPS-aware context window settings"},
#                 ),
#                 "fps_config": (
#                     "WANVIDEOFPSCONFIG",
#                     {"tooltip": "Central FPS configuration"},
#                 ),
#                 "teacache_args": ("TEACACHEARGS",),
#                 "temporal_emphasis": (
#                     "TEMPORALEMPHASIS",
#                     {
#                         "tooltip": "Controls balance between temporal and spatial features"
#                     },
#                 ),
#                 "perceiver_config": (
#                     "PERCEIVERCONFIG",
#                     {"tooltip": "Configuration for the Perceiver Resampler"},
#                 ),
#             },
#         }

#     RETURN_TYPES = ("LATENT",)
#     RETURN_NAMES = ("samples",)
#     FUNCTION = "sample"
#     CATEGORY = "WanVideoWrapper"
#     DESCRIPTION = "FPS-based WanVideo sampler with Apollo paper optimizations for better temporal consistency"

#     def sample(
#         self,
#         model,
#         text_embeds,
#         image_embeds,
#         steps,
#         cfg,
#         shift,
#         seed,
#         scheduler,
#         riflex_freq_index,
#         force_offload=True,
#         sampling_mode="fps",
#         generation_fps=4.0,
#         original_fps=30.0,
#         output_fps=24.0,
#         samples=None,
#         denoise_strength=1.0,
#         feta_args=None,
#         context_options=None,
#         fps_config=None,
#         teacache_args=None,
#         temporal_emphasis=None,
#         perceiver_config=None,
#     ):
#         # Set up unified configuration
#         config = self._prepare_config(
#             generation_fps,
#             original_fps,
#             output_fps,
#             context_options,
#             fps_config,
#             sampling_mode,
#         )

#         # Initialize model and devices
#         patcher = model
#         model = model.model
#         transformer = model.diffusion_model

#         device = mm.get_torch_device()
#         offload_device = mm.unet_offload_device()

#         # Set up diffusion scheduler
#         steps = int(steps / denoise_strength)
#         timesteps, sample_scheduler = self._setup_scheduler(
#             scheduler, shift, steps, device, denoise_strength
#         )

#         # Initialize latents
#         latent, latent_video_length = self._initialize_latents(
#             image_embeds, samples, timesteps, seed, device
#         )

#         # Set up frequency parameters
#         freqs = self._setup_frequencies(
#             transformer, latent_video_length, riflex_freq_index, device
#         )

#         # Prepare arguments for model forward pass
#         base_args, arg_c, arg_null = self._prepare_model_args(
#             transformer,
#             text_embeds,
#             image_embeds,
#             freqs,
#             device,
#             temporal_emphasis,
#             perceiver_config,
#         )

#         # Enable FEtA if provided
#         self._setup_feta(feta_args, config)

#         # Set up block swapping if specified in model
#         self._setup_block_swapping(model, transformer, device, offload_device)

#         # Set up TeaCache if enabled
#         self._setup_teacache(transformer, teacache_args)

#         # Set up progress tracking
#         from comfy.utils import ProgressBar

#         pbar = ProgressBar(steps)

#         # Set up context windows based on config
#         context_windows = self._setup_context_windows(config, latent_video_length)

#         # Optional: Prepare frame blending if enabled
#         blend_enabled = config.get("enable_frame_blending", False)
#         previous_latent = None if blend_enabled else None
#         window_indices = {}  # Keep track of which window each frame belongs to

#         # Set up callback for preview
#         from latent_preview import prepare_callback

#         callback = prepare_callback(patcher, steps)

#         # Start sampling
#         log.info(
#             f"Sampling {(latent_video_length - 1) * 4 + 1} frames at {latent.shape[3] * 8}x{latent.shape[2] * 8}"
#         )
#         log.info(
#             f"Using {config['generation_fps']} FPS generation, {config.get('effective_output_fps', output_fps)} FPS final output"
#         )

#         # Initialize tracking for TeaCache
#         intermediate_device = device
#         teacache_skipped_cond = 0
#         teacache_skipped_uncond = 0

#         # Main sampling loop
#         with torch.autocast(
#             device_type=mm.get_autocast_device(device),
#             dtype=model["dtype"],
#             enabled=True,
#         ):
#             for i, t in enumerate(tqdm(timesteps)):
#                 # Prepare inputs
#                 latent_model_input = [latent.to(device)]
#                 timestep = torch.tensor([t], device=device)

#                 # Calculate step percentage for FEtA and other control mechanisms
#                 current_step_percentage = i / len(timesteps)

#                 # Handle FEtA enable/disable based on step percentage
#                 if feta_args is not None:
#                     self._update_feta(feta_args, current_step_percentage)

#                 # Initialize noise prediction tensor
#                 noise_pred = None

#                 # Process by context windows if enabled
#                 if context_windows:
#                     # Process by windows
#                     noise_pred = self._process_by_windows(
#                         context_windows,
#                         latent_model_input[0],
#                         transformer,
#                         timestep,
#                         i,
#                         t,
#                         cfg[i] if isinstance(cfg, list) else cfg,
#                         arg_c,
#                         arg_null,
#                         intermediate_device,
#                         blend_enabled,
#                         window_indices,
#                     )
#                 else:
#                     # Standard whole-frame processing
#                     noise_pred = self._process_standard(
#                         latent_model_input,
#                         transformer,
#                         timestep,
#                         i,
#                         t,
#                         cfg[i] if isinstance(cfg, list) else cfg,
#                         arg_c,
#                         arg_null,
#                         intermediate_device,
#                         teacache_args is not None,
#                     )

#                 # Move latent to intermediate device for scheduler step
#                 latent = latent.to(intermediate_device)

#                 # Store previous latent for blending if enabled
#                 if blend_enabled:
#                     previous_latent = latent.clone()

#                 # Step the scheduler
#                 temp_x0 = sample_scheduler.step(
#                     noise_pred.unsqueeze(0),
#                     t,
#                     latent.unsqueeze(0),
#                     return_dict=False,
#                     generator=torch.Generator(device=torch.device("cpu")).manual_seed(
#                         seed
#                     ),
#                 )[0]

#                 # Update latent
#                 latent = temp_x0.squeeze(0)

#                 # Prepare next iteration
#                 x0 = [latent.to(device)]

#                 # Update progress display
#                 if callback is not None:
#                     callback_latent = (
#                         (latent_model_input[0] - noise_pred.to(t.device) * t / 1000)
#                         .detach()
#                         .permute(1, 0, 2, 3)
#                     )
#                     callback(i, callback_latent, None, steps)
#                 else:
#                     pbar.update(1)

#                 # Clean up to avoid OOM
#                 del latent_model_input, timestep

#         # Report TeaCache stats if used
#         if hasattr(transformer, "teacache_skipped_cond_steps"):
#             log.info(
#                 f"TeaCache skipped: {transformer.teacache_skipped_cond_steps} cond steps, "
#                 f"{transformer.teacache_skipped_uncond_steps} uncond steps"
#             )

#         # Clean up and return result
#         if force_offload:
#             if model["manual_offloading"]:
#                 transformer.to(offload_device)
#                 mm.soft_empty_cache()
#                 gc.collect()

#         return ({"samples": x0[0].unsqueeze(0).cpu()},)

#     def _standardize_tensor_dims(self, tensor, expected_dims=5):
#         """
#         Standardize tensor to have expected dimensions
#         For WanVideo, typically converting between:
#         - 4D: [c, t, h, w]
#         - 5D: [b, c, t, h, w]
#         """
#         current_dims = tensor.dim()

#         if current_dims == expected_dims:
#             return tensor

#         if current_dims == 4 and expected_dims == 5:
#             # Add batch dimension
#             return tensor.unsqueeze(0)
#         elif current_dims == 5 and expected_dims == 4:
#             # Remove batch dimension (assuming batch=1)
#             return tensor.squeeze(0)

#         # If dimensions differ by more than 1, raise error
#         raise ValueError(
#             f"Cannot convert tensor from {current_dims}D to {expected_dims}D"
#         )

#     def _prepare_config(
#         self,
#         generation_fps,
#         original_fps,
#         output_fps,
#         context_options,
#         fps_config,
#         sampling_mode,
#     ):
#         """Unify configuration sources into a single config dict"""
#         # Start with default values
#         config = {
#             "generation_fps": generation_fps,
#             "original_fps": original_fps,
#             "output_fps": output_fps,
#             "interpolation_factor": max(1, round(output_fps / generation_fps)),
#             "effective_output_fps": generation_fps
#             * max(1, round(output_fps / generation_fps)),
#             "sampling_mode": sampling_mode,
#             "context_schedule": "uniform_standard_fps",
#             "context_frames": 81,
#             "context_stride": 4,
#             "context_overlap": 16,
#             "freenoise": True,
#             "enable_frame_blending": True,
#             "latent_stride": 4,
#         }

#         # Override with fps_config if available (highest priority)
#         if fps_config is not None:
#             for key, value in fps_config.items():
#                 config[key] = value
#             log.info(f"Using fps_config settings")

#         # Override/supplement with context_options if available
#         elif context_options is not None:
#             for key, value in context_options.items():
#                 config[key] = value

#             # Force FPS schedule if using fps sampling mode
#             if (
#                 sampling_mode == "fps"
#                 and config["context_schedule"] != "uniform_standard_fps"
#             ):
#                 config["context_schedule"] = "uniform_standard_fps"
#                 log.info(
#                     f"Changed context schedule to uniform_standard_fps for FPS sampling"
#                 )

#         # Compute latent space values if not present
#         config["latent_context_frames"] = (config["context_frames"] - 1) // config[
#             "latent_stride"
#         ] + 1
#         config["latent_context_overlap"] = (
#             config["context_overlap"] // config["latent_stride"]
#         )
#         config["latent_context_stride"] = max(
#             1,
#             (config["context_frames"] - config["context_overlap"])
#             // config["latent_stride"],
#         )

#         return config

#     def _setup_scheduler(self, scheduler_type, shift, steps, device, denoise_strength):
#         """Set up the diffusion scheduler based on specified type"""

#         if scheduler_type == "unipc":
#             sample_scheduler = FlowUniPCMultistepScheduler(
#                 num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False
#             )
#             sample_scheduler.set_timesteps(steps, device=device, shift=shift)
#             timesteps = sample_scheduler.timesteps
#         elif scheduler_type == "euler":
#             sample_scheduler = FlowMatchEulerDiscreteScheduler(
#                 num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False
#             )
#             sampling_sigmas = get_sampling_sigmas(steps, shift)
#             timesteps, _ = retrieve_timesteps(
#                 sample_scheduler, device=device, sigmas=sampling_sigmas
#             )
#         elif "dpm++" in scheduler_type:
#             algorithm_type = (
#                 "sde-dpmsolver++" if scheduler_type == "dpm++_sde" else "dpmsolver++"
#             )
#             sample_scheduler = FlowDPMSolverMultistepScheduler(
#                 num_train_timesteps=1000,
#                 shift=shift,
#                 use_dynamic_shifting=False,
#                 algorithm_type=algorithm_type,
#             )
#             sampling_sigmas = get_sampling_sigmas(steps, shift)
#             timesteps, _ = retrieve_timesteps(
#                 sample_scheduler, device=device, sigmas=sampling_sigmas
#             )
#         else:
#             raise NotImplementedError(f"Unsupported scheduler: {scheduler_type}")

#         # Adjust timesteps for denoise_strength
#         if denoise_strength < 1.0:
#             steps = int(steps * denoise_strength)
#             timesteps = timesteps[-(steps + 1) :]

#         return timesteps, sample_scheduler

#     def _initialize_latents(self, image_embeds, samples, timesteps, seed, device):
#         """Initialize latent tensors for sampling"""
#         seed_g = torch.Generator(device=torch.device("cpu"))
#         seed_g.manual_seed(seed)

#         if image_embeds.get("target_shape", None) is not None:
#             # For T2V (text to video) mode
#             target_shape = image_embeds["target_shape"]
#             seq_len = image_embeds["max_seq_len"]

#             # Generate noise based on target shape
#             noise = torch.randn(
#                 *target_shape,
#                 dtype=torch.float32,
#                 device=torch.device("cpu"),
#                 generator=seed_g,
#             )

#             # Determine latent video length based on tensor dimensions
#             latent_video_length = noise.shape[1] if noise.dim() == 4 else noise.shape[2]

#         else:
#             # For I2V (image to video) mode
#             lat_h = image_embeds.get("lat_h")
#             lat_w = image_embeds.get("lat_w")
#             seq_len = image_embeds["max_seq_len"]

#             # Default to 4D tensor for I2V mode based on WanVideo implementation
#             noise = torch.randn(
#                 16,  # channels
#                 (image_embeds["num_frames"] - 1) // 4 + 1,  # time
#                 lat_h,  # height
#                 lat_w,  # width
#                 dtype=torch.float32,
#                 generator=seed_g,
#                 device=torch.device("cpu"),
#             )
#             latent_video_length = noise.shape[1]

#         # Initialize from input samples if provided (for video2video)
#         if samples is not None:
#             latent_timestep = timesteps[:1].to(noise.device)
#             sample_tensor = samples["samples"]

#             # Make sure sample_tensor has the same shape as noise
#             if sample_tensor.dim() != noise.dim():
#                 if noise.dim() == 4 and sample_tensor.dim() == 5:
#                     sample_tensor = sample_tensor.squeeze(0)
#                 elif noise.dim() == 5 and sample_tensor.dim() == 4:
#                     sample_tensor = sample_tensor.unsqueeze(0)

#             # Apply noise with denoise strength via timestep
#             noise = noise * latent_timestep / 1000 + (
#                 1 - latent_timestep / 1000
#             ) * sample_tensor.to(noise.device)

#         return noise.to(device), latent_video_length

#     def _setup_frequencies(
#         self, transformer, latent_video_length, riflex_freq_index, device
#     ):
#         """Set up frequency parameters for rotary embeddings"""
#         from .wanvideo.modules.model import rope_params

#         d = transformer.dim // transformer.num_heads
#         freqs = torch.cat(
#             [
#                 rope_params(
#                     1024,
#                     d - 4 * (d // 6),
#                     L_test=latent_video_length,
#                     k=riflex_freq_index,
#                 ),
#                 rope_params(1024, 2 * (d // 6)),
#                 rope_params(1024, 2 * (d // 6)),
#             ],
#             dim=1,
#         )

#         return freqs.to(device)

#     def _prepare_model_args(
#         self,
#         transformer,
#         text_embeds,
#         image_embeds,
#         freqs,
#         device,
#         temporal_emphasis,
#         perceiver_config,
#     ):
#         """Prepare arguments for model forward passes"""
#         # Base arguments for both conditional and unconditional paths
#         base_args = {
#             "clip_fea": image_embeds.get("clip_context", None),
#             "seq_len": image_embeds.get("max_seq_len"),
#             "device": device,
#             "freqs": freqs,
#         }

#         # Add model-specific args for I2V (image to video) mode
#         if transformer.model_type == "i2v" and "image_embeds" in image_embeds:
#             base_args.update({"y": [image_embeds["image_embeds"]]})

#         # Add temporal emphasis if provided
#         if temporal_emphasis is not None:
#             base_args.update({"temporal_emphasis": temporal_emphasis})

#         # Add perceiver config if provided
#         if perceiver_config is not None:
#             base_args.update({"perceiver_config": perceiver_config})

#         # Conditional and unconditional paths
#         arg_c = base_args.copy()
#         arg_c.update({"context": [text_embeds["prompt_embeds"][0]]})

#         arg_null = base_args.copy()
#         arg_null.update({"context": text_embeds["negative_prompt_embeds"]})

#         return base_args, arg_c, arg_null

#     def _setup_feta(self, feta_args, config):
#         """Set up Enhance-A-Video if enabled"""
#         from .enhance_a_video.globals import (
#             enable_enhance,
#             disable_enhance,
#             set_enhance_weight,
#             set_num_frames,
#         )

#         if feta_args is not None:
#             set_enhance_weight(feta_args["weight"])
#             feta_start_percent = feta_args["start_percent"]
#             feta_end_percent = feta_args["end_percent"]

#             # Use context frames if available
#             if "latent_context_frames" in config:
#                 set_num_frames(config["latent_context_frames"])
#             else:
#                 # Default in original code
#                 set_num_frames(21)  # 81 // 4 + 1

#             enable_enhance()
#         else:
#             disable_enhance()

#     def _update_feta(self, feta_args, current_step_percentage):
#         """Update FEtA status based on current step percentage"""
#         from .enhance_a_video.globals import enable_enhance, disable_enhance

#         if feta_args is not None:
#             feta_start_percent = feta_args["start_percent"]
#             feta_end_percent = feta_args["end_percent"]

#             if feta_start_percent <= current_step_percentage <= feta_end_percent:
#                 enable_enhance()
#             else:
#                 disable_enhance()

#     def _setup_block_swapping(self, model, transformer, device, offload_device):
#         """Set up block swapping for memory optimization if specified in model"""
#         if model["block_swap_args"] is not None:
#             for name, param in transformer.named_parameters():
#                 if "block" not in name:
#                     param.data = param.data.to(device)
#                 elif model["block_swap_args"]["offload_txt_emb"] and "txt_emb" in name:
#                     param.data = param.data.to(offload_device)
#                 elif model["block_swap_args"]["offload_img_emb"] and "img_emb" in name:
#                     param.data = param.data.to(offload_device)

#             transformer.block_swap(
#                 model["block_swap_args"]["blocks_to_swap"] - 1,
#                 model["block_swap_args"]["offload_txt_emb"],
#                 model["block_swap_args"]["offload_img_emb"],
#             )
#         elif model["auto_cpu_offload"]:
#             for module in transformer.modules():
#                 if hasattr(module, "offload"):
#                     module.offload()
#                 if hasattr(module, "onload"):
#                     module.onload()
#         elif model["manual_offloading"]:
#             transformer.to(device)

#     def _setup_teacache(self, transformer, teacache_args):
#         """Set up TeaCache if enabled"""
#         if teacache_args is not None:
#             transformer.enable_teacache = True
#             transformer.rel_l1_thresh = teacache_args["rel_l1_thresh"]
#             transformer.teacache_start_step = teacache_args["start_step"]
#             transformer.teacache_cache_device = teacache_args["cache_device"]

#             # Initialize tracking attributes
#             transformer.teacache_skipped_cond_steps = 0
#             transformer.teacache_skipped_uncond_steps = 0
#         else:
#             transformer.enable_teacache = False

#     def _setup_context_windows(self, config, latent_video_length):
#         """Set up context windows based on configuration"""
#         if config.get("sampling_mode") != "fps" or "context_schedule" not in config:
#             return None

#         from .context import get_context_scheduler

#         context_schedule = config["context_schedule"]
#         context = get_context_scheduler(context_schedule)

#         # Generate window definitions
#         windows = list(
#             context(
#                 0,  # step
#                 None,  # num_steps
#                 latent_video_length,
#                 config.get("latent_context_frames", 21),
#                 config.get("latent_context_stride", 17),
#                 config.get("latent_context_overlap", 4),
#                 config.get("target_fps", 4.0),
#                 config.get("original_fps", 30.0),
#                 True,  # closed_loop
#                 config.get("enable_frame_blending", True),
#             )
#         )

#         # Logging for diagnostics
#         total_frames = set()
#         for window in windows:
#             total_frames.update(window)

#         log.info(
#             f"Created {len(windows)} context windows covering {len(total_frames)}/{latent_video_length} frames"
#         )

#         return windows

#     def _process_by_windows(
#         self,
#         windows,
#         latent,
#         transformer,
#         timestep,
#         step_idx,
#         t,
#         cfg_scale,
#         arg_c,
#         arg_null,
#         intermediate_device,
#         blend_enabled,
#         window_indices,
#     ):
#         """Process sampling in context windows with optional frame blending"""
#         # Zero-initialize noise prediction and counter tensors
#         noise_pred = torch.zeros_like(latent, device=intermediate_device)
#         counter = torch.zeros_like(latent, device=intermediate_device)

#         # Process each context window
#         for window_idx, window_frames in enumerate(windows):
#             # Get the latent for this window
#             # Adjust indexing based on actual tensor shape
#             if latent.dim() == 4:  # [c, t, h, w]
#                 partial_latent = latent[:, window_frames, :, :]
#             else:  # [b, c, t, h, w]
#                 partial_latent = latent[:, :, window_frames, :, :]

#             partial_model_input = [partial_latent]

#             # Keep track of which frames belong to this window
#             if blend_enabled:
#                 for frame_idx in window_frames:
#                     if frame_idx not in window_indices:
#                         window_indices[frame_idx] = []
#                     window_indices[frame_idx].append(window_idx)

#             # Perform conditional forward pass
#             noise_pred_cond = transformer(
#                 partial_model_input,
#                 t=timestep,
#                 current_step=step_idx,
#                 is_uncond=False,
#                 **arg_c,
#             )[0].to(intermediate_device)

#             # Perform unconditional forward pass if CFG > 1.0
#             if cfg_scale != 1.0:
#                 noise_pred_uncond = transformer(
#                     partial_model_input,
#                     t=timestep,
#                     current_step=step_idx,
#                     is_uncond=True,
#                     **arg_null,
#                 )[0].to(intermediate_device)

#                 noise_pred_window = noise_pred_uncond + cfg_scale * (
#                     noise_pred_cond - noise_pred_uncond
#                 )
#             else:
#                 noise_pred_window = noise_pred_cond

#             # Create blending mask based on position in the window
#             window_mask = self._create_window_mask(
#                 noise_pred_window,
#                 window_frames,
#                 window_idx,
#                 len(windows),
#                 window_indices if blend_enabled else None,
#             )

#             # Apply the window's noise prediction to the global tensor
#             for i, frame_idx in enumerate(window_frames):
#                 # Adjust indexing based on actual tensor shape
#                 if noise_pred.dim() == 4:  # [c, t, h, w]
#                     noise_pred[:, frame_idx, :, :] += (
#                         noise_pred_window[:, i, :, :] * window_mask[:, i, :, :]
#                     )
#                     counter[:, frame_idx, :, :] += window_mask[:, i, :, :]
#                 else:  # [b, c, t, h, w]
#                     noise_pred[:, :, frame_idx, :, :] += (
#                         noise_pred_window[:, :, i, :, :] * window_mask[:, :, i, :, :]
#                     )
#                     counter[:, :, frame_idx, :, :] += window_mask[:, :, i, :, :]

#         # Normalize by counter to get weighted average
#         valid_mask = counter > 0
#         noise_pred[valid_mask] /= counter[valid_mask]

#         return noise_pred

#     def _create_window_mask(
#         self, noise_pred, window_frames, window_idx, total_windows, window_indices=None
#     ):
#         """Create blending mask for window transitions"""
#         device = noise_pred.device

#         # Create mask with appropriate dimensions
#         window_mask = torch.ones_like(noise_pred)

#         # Skip if not doing frame blending
#         if window_indices is None:
#             return window_mask

#         # Skip if this is the only window
#         if total_windows <= 1:
#             return window_mask

#         # Apply left-side blending for all except first window with small overlap
#         if window_idx > 0:
#             # Find frames that are in both this window and the previous one
#             overlap_idx = 0
#             overlap_count = 0

#             for i, frame_idx in enumerate(window_frames):
#                 if len(window_indices.get(frame_idx, [])) > 1:
#                     # This frame belongs to multiple windows
#                     overlap_count += 1

#                     # Create gradual blend weights
#                     if overlap_count > 0:
#                         # Ease-in weight (0.0 -> 1.0)
#                         t = overlap_idx / max(1, overlap_count - 1)
#                         weight = min(1.0, 0.5 * (1.0 - math.cos(math.pi * t)))

#                         # Apply weight to the correct dimensions
#                         if window_mask.dim() == 4:  # [c, t, h, w]
#                             window_mask[:, i, :, :] = weight
#                         else:  # [b, c, t, h, w]
#                             window_mask[:, :, i, :, :] = weight

#                         overlap_idx += 1

#         # Apply right-side blending for all except last window
#         if window_idx < total_windows - 1:
#             # Find frames that are in both this window and the next one
#             overlap_frames = []

#             for i, frame_idx in enumerate(reversed(window_frames)):
#                 if len(window_indices.get(frame_idx, [])) > 1:
#                     overlap_frames.append((len(window_frames) - 1 - i, frame_idx))

#             # Create gradual blend weights for these frames
#             for idx, (i, frame_idx) in enumerate(overlap_frames):
#                 # Ease-out weight (1.0 -> 0.0)
#                 if len(overlap_frames) > 0:
#                     t = idx / max(1, len(overlap_frames) - 1)
#                     weight = min(1.0, 0.5 * (1.0 + math.cos(math.pi * t)))

#                     # Apply weight to the correct dimensions
#                     if window_mask.dim() == 4:  # [c, t, h, w]
#                         window_mask[:, i, :, :] = weight
#                     else:  # [b, c, t, h, w]
#                         window_mask[:, :, i, :, :] = weight

#         return window_mask

#     def _process_standard(
#         self,
#         latent_model_input,
#         transformer,
#         timestep,
#         step_idx,
#         t,
#         cfg_scale,
#         arg_c,
#         arg_null,
#         intermediate_device,
#         use_teacache,
#     ):
#         """Process sampling without context windows"""
#         # Do TeaCache checking first if enabled
#         should_calc = True
#         if use_teacache and step_idx >= transformer.teacache_start_step:
#             if hasattr(transformer, "previous_residual_uncond") and hasattr(
#                 transformer, "previous_residual_cond"
#             ):
#                 # TeaCache logic would go here if implemented
#                 pass

#         # Normal processing path
#         if not use_teacache or should_calc:
#             # Conditional forward pass
#             noise_pred_cond = transformer(
#                 latent_model_input,
#                 t=timestep,
#                 current_step=step_idx,
#                 is_uncond=False,
#                 **arg_c,
#             )[0].to(intermediate_device)

#             # Unconditional forward pass if CFG > 1.0
#             if cfg_scale != 1.0:
#                 noise_pred_uncond = transformer(
#                     latent_model_input,
#                     t=timestep,
#                     current_step=step_idx,
#                     is_uncond=True,
#                     **arg_null,
#                 )[0].to(intermediate_device)

#                 noise_pred = noise_pred_uncond + cfg_scale * (
#                     noise_pred_cond - noise_pred_uncond
#                 )
#             else:
#                 noise_pred = noise_pred_cond

#         return noise_pred


# latent interpolation / interpolator node


class WanVideoLatentInterpolator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "interpolation_factor": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": 12,
                        "step": 1,
                        "tooltip": "Number of frames to interpolate between each original frame",
                    },
                ),
                "method": (
                    ["linear", "slerp", "adaptive", "debug_simple"],
                    {
                        "default": "linear",
                        "tooltip": "Interpolation method: linear is straight averaging, slerp is spherical linear interpolation, adaptive chooses the best method per frame pair",
                    },
                ),
                "ease_method": (
                    ["linear", "ease_in", "ease_out", "ease_in_out"],
                    {
                        "default": "ease_out",
                        "tooltip": "Easing function to control interpolation timing",
                    },
                ),
            },
            "optional": {
                "context_options": (
                    "WANVIDCONTEXTFPS",
                    {"tooltip": "For recommended interpolation factor"},
                ),
                "fps_config": (
                    "WANVIDEOFPSCONFIG",
                    {
                        "tooltip": "If provided, will use recommended interpolation factor from FPS settings"
                    },
                ),
                "use_recommended_factor": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use recommended interpolation factor from FPS config",
                    },
                ),
                "motion_smoothness": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Controls motion smoothing across interpolated frames (0=off, 1=max)",
                    },
                ),
                "debug_output": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Output additional debug information",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("samples", "debug_image")
    FUNCTION = "interpolate"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Interpolates between latent frames for smoother video with higher effective frame rates."

    def interpolate(
        self,
        samples,
        interpolation_factor,
        method="linear",
        ease_method="ease_out",
        context_options=None,
        fps_config=None,
        use_recommended_factor=True,
        motion_smoothness=0.5,
        debug_output=False,
    ):
        # First check fps_config (highest priority)
        if fps_config is not None and use_recommended_factor:
            if "interpolation_factor" in fps_config:
                interpolation_factor = fps_config["interpolation_factor"]
                log.info(
                    f"Using interpolation factor {interpolation_factor} from fps_config"
                )

                # Also use interpolation method if provided
                if "interpolation_method" in fps_config:
                    method = fps_config["interpolation_method"]
                    log.info(f"Using interpolation method {method} from fps_config")

        # Then check context_options (backwards compatibility)
        elif context_options is not None and use_recommended_factor:
            if "recommended_interp_factor" in context_options:
                interpolation_factor = context_options["recommended_interp_factor"]
                log.info(
                    f"Using interpolation factor {interpolation_factor} from context_options"
                )

        # Log the configuration
        log.info(
            f"Interpolating with factor {interpolation_factor}, method: {method}, easing: {ease_method}"
        )

        # Get latent tensor
        latents = samples["samples"]

        # If interpolation_factor is 1, just return the original latents
        if interpolation_factor <= 1:
            log.info("Interpolation factor <= 1, returning original latents")
            return ({"samples": latents}, torch.zeros((1, 1, 1, 1)))

        # Validate input
        if torch.isnan(latents).any():
            log.warning(
                "Input latents contain NaN values! Results may be unpredictable."
            )

        # Get shapes
        b, c, t, h, w = latents.shape
        device = latents.device

        # Create empty tensor for interpolated latents
        interpolated_t = (t - 1) * interpolation_factor + 1
        log.info(f"Interpolating {t} frames to {interpolated_t} frames")

        interpolated_latents = torch.zeros(
            (b, c, interpolated_t, h, w), device=device, dtype=latents.dtype
        )

        # Copy original frames to their positions in the interpolated sequence
        for i in range(t):
            interpolated_latents[:, :, i * interpolation_factor, :, :] = latents[
                :, :, i, :, :
            ]

        # Apply the selected interpolation method
        if method == "debug_simple":
            # Simplified method for debugging
            interpolated_latents = self._apply_simplified_interpolation(
                latents, interpolated_latents, interpolation_factor, ease_method, t
            )
        elif method == "linear":
            interpolated_latents = self._apply_linear_interpolation(
                latents, interpolated_latents, interpolation_factor, ease_method, t
            )
        elif method == "slerp":
            interpolated_latents = self._apply_slerp_interpolation(
                latents,
                interpolated_latents,
                interpolation_factor,
                ease_method,
                t,
                h,
                w,
                b,
                c,
            )
        elif method == "adaptive":
            interpolated_latents = self._apply_adaptive_interpolation(
                latents,
                interpolated_latents,
                interpolation_factor,
                ease_method,
                t,
                h,
                w,
                b,
                c,
            )

        # Apply temporal smoothing if requested
        if motion_smoothness > 0:
            interpolated_latents = self._apply_temporal_smoothing(
                interpolated_latents, strength=motion_smoothness
            )

        # Generate debug visualization if requested
        debug_image = torch.zeros((1, 1, 1, 1))
        if debug_output:
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                import io
                from PIL import Image

                # Create simplified debug visualization
                fig, ax = plt.subplots(figsize=(10, 6))

                # Create a timeline showing original vs interpolated frames
                timeline = [
                    "original" if i % interpolation_factor == 0 else "interpolated"
                    for i in range(interpolated_t)
                ]

                colors = [
                    "blue" if frame == "original" else "green" for frame in timeline
                ]
                ax.bar(range(len(timeline)), [1] * len(timeline), color=colors)

                ax.set_title(
                    f"Frame Interpolation: {t} → {interpolated_t} frames ({method} method)"
                )
                ax.set_xlabel("Frame Index")
                ax.set_yticks([])

                # Create legend
                ax.bar(0, 0, color="blue", label="Original")
                ax.bar(0, 0, color="green", label="Interpolated")
                plt.legend()

                # Convert to image
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                img = Image.open(buf)
                debug_image = torch.from_numpy(np.array(img)).float() / 255.0
                plt.close(fig)

            except Exception as e:
                log.warning(f"Failed to create debug visualization: {e}")
                debug_image = torch.zeros((100, 400, 3))

        log.info(f"Interpolation complete: {t} frames → {interpolated_t} frames")
        return ({"samples": interpolated_latents}, debug_image)

    def _apply_simplified_interpolation(
        self, latents, interpolated_latents, interpolation_factor, ease_method, t
    ):
        """A simpler, more reliable interpolation method for debugging"""
        for i in range(t - 1):
            start_frame = latents[:, :, i, :, :]
            end_frame = latents[:, :, i + 1, :, :]

            # Simple linear interpolation between frames
            for j in range(1, interpolation_factor):
                # Apply easing function to the interpolation factor
                alpha_raw = j / interpolation_factor
                alpha = self._ease_function(alpha_raw, ease_method)

                frame_idx = i * interpolation_factor + j
                interpolated_latents[:, :, frame_idx, :, :] = (
                    1 - alpha
                ) * start_frame + alpha * end_frame

        return interpolated_latents

    def _apply_linear_interpolation(
        self, latents, interpolated_latents, interpolation_factor, ease_method, t
    ):
        """Standard linear interpolation between frames"""
        for i in range(t - 1):
            start_frame = latents[:, :, i, :, :]
            end_frame = latents[:, :, i + 1, :, :]

            # Interpolate frames
            for j in range(1, interpolation_factor):
                # Apply easing function to the interpolation factor
                alpha_raw = j / interpolation_factor
                alpha = self._ease_function(alpha_raw, ease_method)

                frame_idx = i * interpolation_factor + j
                interpolated_latents[:, :, frame_idx, :, :] = (
                    1 - alpha
                ) * start_frame + alpha * end_frame

        return interpolated_latents

    def _apply_slerp_interpolation(
        self,
        latents,
        interpolated_latents,
        interpolation_factor,
        ease_method,
        t,
        h,
        w,
        b,
        c,
    ):
        """
        Spherical Linear Interpolation (SLERP) with improved stability
        """
        for i in range(t - 1):
            start_frame = latents[:, :, i, :, :]
            end_frame = latents[:, :, i + 1, :, :]

            # Flatten spatial dimensions for slerp calculation
            start_flat = start_frame.reshape(b, c, -1)
            end_flat = end_frame.reshape(b, c, -1)

            # Normalize vectors (avoid division by zero)
            start_norm = torch.nn.functional.normalize(start_flat, dim=2, eps=1e-8)
            end_norm = torch.nn.functional.normalize(end_flat, dim=2, eps=1e-8)

            # Compute cosine similarity
            dot = torch.sum(start_norm * end_norm, dim=2, keepdim=True)

            # Clamp dot product to avoid numerical instability
            dot = torch.clamp(dot, -0.9999, 0.9999)

            # Calculate threshold for linear vs slerp interpolation
            linear_threshold = 0.9995
            linear_falloff = 0.0002

            # Create smooth transition between linear and slerp
            linear_mask = torch.clamp(
                (dot - (linear_threshold - linear_falloff)) / linear_falloff, 0.0, 1.0
            )
            slerp_mask = 1.0 - linear_mask

            for j in range(1, interpolation_factor):
                # Apply easing function to the interpolation factor
                alpha_raw = j / interpolation_factor
                alpha = self._ease_function(alpha_raw, ease_method)

                # Compute angle and sin(angle) for slerp
                theta = torch.acos(dot) * alpha
                sin_theta = torch.sin(theta)
                sin_theta_complement = torch.sin((1.0 - alpha) * torch.acos(dot))

                # Add small epsilon to avoid division by zero
                sin_acos_dot = torch.sin(torch.acos(dot))
                sin_acos_dot = torch.clamp(sin_acos_dot, min=1e-8)

                # Linear interpolation component
                linear_interp = (1.0 - alpha) * start_flat + alpha * end_flat

                # SLERP interpolation component
                slerp_interp = (
                    sin_theta_complement / sin_acos_dot * start_flat
                    + sin_theta / sin_acos_dot * end_flat
                )

                # Combine linear and slerp based on the mask
                interp_flat = linear_mask * linear_interp + slerp_mask * slerp_interp

                # Reshape back to original dimensions
                frame_idx = i * interpolation_factor + j
                interpolated_latents[:, :, frame_idx, :, :] = interp_flat.reshape(
                    b, c, h, w
                )

        return interpolated_latents

    def _apply_adaptive_interpolation(
        self,
        latents,
        interpolated_latents,
        interpolation_factor,
        ease_method,
        t,
        h,
        w,
        b,
        c,
    ):
        """
        Adaptive interpolation that chooses between linear and SLERP based on frame characteristics
        """
        for i in range(t - 1):
            start_frame = latents[:, :, i, :, :]
            end_frame = latents[:, :, i + 1, :, :]

            # Calculate frame difference to decide which method to use
            frame_diff = torch.abs(end_frame - start_frame).mean()
            frame_mag = (
                torch.abs(end_frame).mean() + torch.abs(start_frame).mean()
            ) / 2
            relative_diff = frame_diff / (frame_mag + 1e-8)

            # Higher difference = more SLERP weight (better for big changes)
            # Lower difference = more linear weight (better for subtle changes)
            slerp_weight = torch.clamp(relative_diff * 10, 0.0, 1.0)

            # Flatten for calculations
            start_flat = start_frame.reshape(b, c, -1)
            end_flat = end_frame.reshape(b, c, -1)

            # Calculate normalized vectors for SLERP
            start_norm = torch.nn.functional.normalize(start_flat, dim=2, eps=1e-8)
            end_norm = torch.nn.functional.normalize(end_flat, dim=2, eps=1e-8)
            dot = torch.sum(start_norm * end_norm, dim=2, keepdim=True)
            dot = torch.clamp(dot, -0.9999, 0.9999)

            for j in range(1, interpolation_factor):
                # Apply easing function
                alpha_raw = j / interpolation_factor
                alpha = self._ease_function(alpha_raw, ease_method)

                # Linear interpolation (simple, works well for subtle changes)
                linear_interp = (1.0 - alpha) * start_flat + alpha * end_flat

                # SLERP interpolation (better for larger movements)
                theta = torch.acos(dot) * alpha
                sin_theta = torch.sin(theta)
                sin_theta_complement = torch.sin((1.0 - alpha) * torch.acos(dot))
                sin_acos_dot = torch.clamp(torch.sin(torch.acos(dot)), min=1e-8)

                slerp_interp = (
                    sin_theta_complement / sin_acos_dot * start_flat
                    + sin_theta / sin_acos_dot * end_flat
                )

                # Blend between linear and SLERP based on frame difference
                interp_flat = (
                    1 - slerp_weight
                ) * linear_interp + slerp_weight * slerp_interp

                # Reshape back to original dimensions
                frame_idx = i * interpolation_factor + j
                interpolated_latents[:, :, frame_idx, :, :] = interp_flat.reshape(
                    b, c, h, w
                )

        return interpolated_latents

    def _apply_temporal_smoothing(self, latents, strength=0.5, kernel_size=3):
        """
        Apply temporal smoothing to reduce jitter in the interpolated frames
        """
        # Skip if smoothing disabled or kernel too large for sequence
        b, c, t, h, w = latents.shape
        if strength <= 0 or t < kernel_size:
            return latents

        # Create a copy for smoothing
        smoothed = latents.clone()

        # Simple 1D temporal convolution
        # Only smooth the interpolated frames, not the keyframes
        half_k = kernel_size // 2

        # Determine frame indices that represent original frames
        interp_factor = 0
        for i in range(2, t):
            if torch.allclose(
                latents[:, :, 0, :, :], latents[:, :, i, :, :], atol=1e-5
            ):
                interp_factor = i
                break

        if interp_factor == 0:
            # Couldn't detect interpolation factor, assume all frames need smoothing
            keyframes = []
        else:
            # Mark keyframes (original frames) to avoid smoothing them
            keyframes = list(range(0, t, interp_factor))

        # Apply smoothing to non-keyframes
        for i in range(t):
            # Skip keyframes (original frames)
            if i in keyframes:
                continue

            # Calculate window bounds
            start_idx = max(0, i - half_k)
            end_idx = min(t, i + half_k + 1)

            # Get temporal window
            window = latents[:, :, start_idx:end_idx, :, :]

            # Calculate temporal weights based on distance
            weights = torch.ones(end_idx - start_idx, device=latents.device)
            center_pos = i - start_idx
            for j in range(len(weights)):
                weights[j] = 1.0 - min(1.0, abs(j - center_pos) / half_k) * 0.5

            # Normalize weights
            weights = weights / weights.sum()

            # Apply weighted average
            smoothed[:, :, i, :, :] = 0  # Reset
            for j in range(len(weights)):
                if j == center_pos:
                    # Original frame gets higher weight
                    smoothed[:, :, i, :, :] += window[:, :, j, :, :] * (1.0 - strength)
                else:
                    # Add weighted contribution from neighboring frames
                    smoothed[:, :, i, :, :] += window[:, :, j, :, :] * (
                        weights[j] * strength
                    )

        return smoothed

    def _ease_function(self, t, method="linear"):
        """
        Apply various easing functions to interpolation factor t
        """
        if method == "ease_in":
            return t * t
        elif method == "ease_out":
            return 1 - (1 - t) * (1 - t)
        elif method == "ease_in_out":
            return 0.5 * (1 - math.cos(t * math.pi))
        else:  # Linear
            return t


class WanVideoTemporalEmphasis:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "temporal_emphasis": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "Higher values emphasize motion, lower values emphasize static content",
                    },
                ),
                "motion_smoothness": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "Controls motion interpolation smoothness - higher values reduce jitter",
                    },
                ),
                "detail_preservation": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "Controls preservation of spatial details across frames",
                    },
                ),
            },
        }

    RETURN_TYPES = ("TEMPORALEMPHASIS",)
    RETURN_NAMES = ("temporal_emphasis",)
    FUNCTION = "set_emphasis"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Controls the emphasis on temporal features vs. spatial features based on Apollo paper findings"

    def set_emphasis(self, temporal_emphasis, motion_smoothness, detail_preservation):
        """
        Configure temporal emphasis parameters for the video generation

        Args:
            temporal_emphasis: Higher values emphasize motion, lower values emphasize static content
            motion_smoothness: Controls motion interpolation smoothness
            detail_preservation: Controls preservation of spatial details

        Returns:
            Dictionary of temporal emphasis parameters
        """
        log.info(
            f"Setting temporal emphasis to {temporal_emphasis}, motion smoothness to {motion_smoothness}, detail preservation to {detail_preservation}"
        )

        return (
            {
                "temporal_emphasis": temporal_emphasis,
                "motion_smoothness": motion_smoothness,
                "detail_preservation": detail_preservation,
            },
        )


class WanVideoPerceiverResampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tokens_per_frame": (
                    "INT",
                    {
                        "default": 32,
                        "min": 4,
                        "max": 128,
                        "step": 4,
                        "tooltip": "Number of tokens per frame after resampling",
                    },
                ),
                "num_layers": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 6,
                        "step": 1,
                        "tooltip": "Number of attention layers in the Perceiver Resampler",
                    },
                ),
                "num_heads": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "tooltip": "Number of attention heads in the Perceiver Resampler",
                    },
                ),
                "head_dim": (
                    "INT",
                    {
                        "default": 64,
                        "min": 32,
                        "max": 128,
                        "step": 8,
                        "tooltip": "Dimension of each attention head",
                    },
                ),
            },
        }

    RETURN_TYPES = ("PERCEIVERCONFIG",)
    RETURN_NAMES = ("perceiver_config",)
    FUNCTION = "configure"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Configures a Perceiver Resampler for more effective token resampling based on Apollo paper findings"

    def configure(self, tokens_per_frame, num_layers, num_heads, head_dim):
        """
        Configure the Perceiver Resampler parameters

        Args:
            tokens_per_frame: Number of tokens per frame after resampling
            num_layers: Number of attention layers
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head

        Returns:
            Configuration dictionary for the Perceiver Resampler
        """
        log.info(
            f"Configuring Perceiver Resampler with {tokens_per_frame} tokens per frame, {num_layers} layers, {num_heads} heads, and {head_dim} head dimension"
        )

        # Create configuration for the Perceiver
        config = {
            "tokens_per_frame": tokens_per_frame,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
        }

        return (config,)


class WanVideoKeyframeConditioner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "keyframes": ("IMAGE",),
                "fps_config": ("WANVIDEOFPSCONFIG",),
                "keyframe_positions": (
                    "STRING",
                    {
                        "default": "0.0,0.5,1.0",
                        "tooltip": "Positions as fractions (0-1) of video duration",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "vae": ("WANVAE",),
                "clip": ("WANCLIP",),
            },
            "optional": {
                "mask": (
                    "MASK",
                    {
                        "tooltip": "Optional mask to apply conditioning only to specific regions"
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latents", "keyframe_visualization")
    FUNCTION = "condition"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = (
        "Uses reference images at specific timestamps to guide video generation"
    )

    def condition(
        self, keyframes, fps_config, keyframe_positions, strength, vae, clip, mask=None
    ):
        """
        Prepare keyframe conditioning for video generation.

        Args:
            keyframes: Reference images (B,H,W,C)
            fps_config: FPS configuration
            keyframe_positions: Comma-separated positions (0-1)
            strength: Conditioning strength
            vae: VAE for encoding keyframes
            clip: CLIP model for image embeddings
            mask: Optional mask for regional conditioning

        Returns:
            Latent keyframes and visualization
        """
        # Parse keyframe positions
        positions = [float(p.strip()) for p in keyframe_positions.split(",")]
        if len(positions) > keyframes.shape[0]:
            positions = positions[: keyframes.shape[0]]
        elif len(positions) < keyframes.shape[0]:
            # Distribute remaining frames evenly
            remaining_count = keyframes.shape[0] - len(positions)
            for i in range(1, remaining_count + 1):
                ratio = i / (remaining_count + 1)
                positions.append(ratio)
            positions.sort()

        log.info(f"Keyframe positions: {positions}")

        # Calculate corresponding frame indices
        total_frames = (
            fps_config.get("latent_context_frames", 32) * 4
        )  # Convert to pixel frames
        frame_indices = [
            min(total_frames - 1, max(0, int(p * total_frames))) for p in positions
        ]

        # Prepare device
        device = mm.get_torch_device()

        # Need to move models to the right device for processing
        clip.model.to(device)
        vae.to(device)

        # 1. Encode keyframes with CLIP for high-level semantic guidance
        # This is similar to WanVideoImageClipEncode
        from comfy.clip_vision import clip_preprocess

        image_mean = [0.48145466, 0.4578275, 0.40821073]
        image_std = [0.26862954, 0.26130258, 0.27577711]

        # Process each keyframe
        clip_embeds = []
        for i, img in enumerate(keyframes):
            # Preprocess for CLIP
            pixel_values = clip_preprocess(
                img.unsqueeze(0).to(device),
                size=224,
                mean=image_mean,
                std=image_std,
                crop=True,
            ).float()

            # Get CLIP embedding
            clip_embed = clip.visual(pixel_values)
            clip_embeds.append(clip_embed * strength)

        # 2. Encode keyframes with VAE for structural guidance
        # First resize images to match target resolution
        h, w = (
            fps_config.get("latent_context_frames", 32) * 8,
            fps_config.get("latent_context_frames", 32) * 8,
        )
        resized_keyframes = []

        for img in keyframes:
            # Reshape and resize
            resized = common_upscale(img.movedim(-1, 1), w, h, "lanczos", "disabled")
            resized = resized.transpose(0, 1)  # Match VAE input format
            resized = resized * 2 - 1  # Convert to -1 to 1 range
            resized_keyframes.append(resized)

        # Stack and encode
        keyframe_batch = torch.stack(resized_keyframes).to(device)
        latent_keyframes = vae.encode([keyframe_batch], device)[0]

        # Apply strength
        latent_keyframes = latent_keyframes * strength

        # Move models back to CPU to free memory
        clip.model.to(mm.unet_offload_device())
        vae.to(mm.unet_offload_device())

        # Create visualization showing keyframe positions in timeline
        vis_height = 100
        vis_width = 500
        visualization = torch.ones((vis_height, vis_width, 3), dtype=torch.float32)

        # Add timeline
        timeline_y = vis_height // 2
        visualization[timeline_y - 1 : timeline_y + 2, :, :] = 0.7

        # Add keyframe markers
        marker_radius = 5
        for pos in positions:
            x_pos = int(pos * vis_width)
            # Draw circle
            for y in range(timeline_y - marker_radius, timeline_y + marker_radius + 1):
                for x in range(x_pos - marker_radius, x_pos + marker_radius + 1):
                    if 0 <= y < vis_height and 0 <= x < vis_width:
                        dist = ((y - timeline_y) ** 2 + (x - x_pos) ** 2) ** 0.5
                        if dist <= marker_radius:
                            # Red markers
                            visualization[y, x, 0] = 1.0
                            visualization[y, x, 1] = 0.2
                            visualization[y, x, 2] = 0.2

        # Add keyframe thumbnails
        thumb_size = 70
        for i, pos in enumerate(positions):
            if i < len(keyframes):
                x_pos = int(pos * (vis_width - thumb_size))
                # Resize keyframe
                thumb = common_upscale(
                    keyframes[i].permute(2, 0, 1).unsqueeze(0),
                    thumb_size,
                    thumb_size,
                    "lanczos",
                    "disabled",
                )
                thumb = thumb[0].permute(1, 2, 0)

                # Place at top of visualization
                y_start = 5
                visualization[
                    y_start : y_start + thumb_size, x_pos : x_pos + thumb_size
                ] = thumb

        # Return keyframe data and visualization
        return {
            "keyframes": latent_keyframes,
            "clip_embeds": clip_embeds,
            "frame_indices": frame_indices,
            "strength": strength,
            "mask": mask,
        }, visualization


class WanVideoTimelinePreview:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"controller": ("WANVIDEOFPSCONTROLLER",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_preview"

    def create_preview(self, controller):
        # Create a visual timeline showing:
        # - Original frames marked as keyframes
        # - Interpolated frames shown differently
        # - Frame numbers and timestamps
        # This gives a visual representation of what's happening

        width, height = 800, 100
        timeline = torch.ones((height, width, 3), dtype=torch.float32)

        # Draw timeline
        fps = controller["generation_fps"]
        interp_factor = controller["interpolation_factor"]
        log.info(f"fps: {fps}, interp_factor: {interp_factor}")

        # Example code to draw keyframes vs interpolated frames
        for i in range(min(30, int(width / 20))):  # Show first 30 frames max
            x_pos = 20 + i * 20

            # Is this a keyframe or interpolated?
            is_keyframe = (i % interp_factor) == 0

            if is_keyframe:
                # Draw red marker for keyframe
                color = torch.tensor([1.0, 0.3, 0.3])
                size = 8
            else:
                # Draw blue marker for interpolated frame
                color = torch.tensor([0.3, 0.3, 1.0])
                size = 4

            # Draw marker
            y_pos = height // 2
            timeline[y_pos - size : y_pos + size, x_pos - size : x_pos + size] = color

            # todo: Add frame number
            # todo: add a text rendering function

        return (timeline,)


class WanVideoWorkflowPreset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": (
                    [
                        "cinema_24fps",
                        "smooth_30fps",
                        "high_quality_60fps",
                        "debug_mode",
                    ],
                    {
                        "default": "cinema_24fps",
                        "tooltip": "Predefined workflow configurations",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "a cinematic shot of a city street with people walking",
                    },
                ),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8}),
                "duration_seconds": (
                    "FLOAT",
                    {"default": 5.0, "min": 1.0, "max": 30.0, "step": 0.5},
                ),
            },
            "optional": {
                "custom_config": ("BOOLEAN", {"default": False}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg_scale": (
                    "FLOAT",
                    {"default": 6.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "enable_debug": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("WANVIDEOFPSCONFIG", "WANVIDCONTEXTFPS", "INT", "STRING")
    RETURN_NAMES = ("fps_config", "context_options", "frame_count", "formatted_prompt")
    FUNCTION = "generate_preset"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Creates a complete preset configuration for WanVideo generation with optimized settings"

    def generate_preset(
        self,
        preset,
        prompt,
        width,
        height,
        duration_seconds,
        custom_config=False,
        steps=30,
        cfg_scale=6.0,
        enable_debug=True,
    ):
        """Generate complete preset configuration for a workflow"""

        # Base configuration
        config = {
            "source_fps": 30.0,  # Default reference
            "latent_stride": 4,
            "enable_frame_blending": True,
            "enable_debugging": enable_debug,
        }

        # Apply presets
        if preset == "cinema_24fps":
            # Standard cinematic preset
            config.update(
                {
                    "generation_fps": 2.0,
                    "output_fps": 24.0,
                    "interpolation_factor": 12,
                    "interpolation_method": "linear",
                    "steps": 30 if not custom_config else steps,
                    "cfg_scale": 6.0 if not custom_config else cfg_scale,
                }
            )
        elif preset == "smooth_30fps":
            # Smoother motion with higher generation FPS
            config.update(
                {
                    "generation_fps": 3.0,
                    "output_fps": 30.0,
                    "interpolation_factor": 10,
                    "interpolation_method": "slerp",
                    "steps": 24 if not custom_config else steps,
                    "cfg_scale": 7.0 if not custom_config else cfg_scale,
                }
            )
        elif preset == "high_quality_60fps":
            # Maximum quality high fps
            config.update(
                {
                    "generation_fps": 4.0,
                    "output_fps": 60.0,
                    "interpolation_factor": 15,
                    "interpolation_method": "adaptive",
                    "steps": 40 if not custom_config else steps,
                    "cfg_scale": 8.0 if not custom_config else cfg_scale,
                }
            )
        elif preset == "debug_mode":
            # For debugging issues
            config.update(
                {
                    "generation_fps": 2.0,
                    "output_fps": 8.0,
                    "interpolation_factor": 4,
                    "interpolation_method": "debug_simple",
                    "steps": 10 if not custom_config else steps,
                    "cfg_scale": 5.0 if not custom_config else cfg_scale,
                }
            )

        # Calculate derived values
        config["effective_output_fps"] = (
            config["generation_fps"] * config["interpolation_factor"]
        )

        # Calculate frames needed
        pixel_frames = int(duration_seconds * config["generation_fps"])
        latent_frames = (pixel_frames - 1) // config["latent_stride"] + 1

        # Set up context handling
        context_frames = 81
        context_overlap = 16
        context_stride = context_frames - context_overlap

        config.update(
            {
                "context_frames": context_frames,
                "context_overlap": context_overlap,
                "context_stride": context_stride,
                "latent_context_frames": (context_frames - 1) // config["latent_stride"]
                + 1,
                "latent_context_overlap": context_overlap // config["latent_stride"],
                "latent_context_stride": context_stride // config["latent_stride"],
            }
        )

        # Create context options format for backward compatibility
        context_options = {
            "context_schedule": "uniform_standard_fps",
            "context_frames": config["context_frames"],
            "context_stride": config["context_stride"],
            "context_overlap": config["context_overlap"],
            "freenoise": True,
            "target_fps": config["generation_fps"],
            "original_fps": 30.0,
            "output_fps": config["output_fps"],
            "effective_output_fps": config["effective_output_fps"],
            "recommended_interp_factor": config["interpolation_factor"],
            "enable_frame_blending": config["enable_frame_blending"],
            "latent_stride": config["latent_stride"],
            "latent_context_frames": config["latent_context_frames"],
            "latent_context_overlap": config["latent_context_overlap"],
            "latent_context_stride": config["latent_context_stride"],
        }

        # Format prompt to include video quality instructions
        formatted_prompt = self._format_prompt(prompt, preset)

        # Log summary
        log.info(f"Generated {preset} workflow preset:")
        log.info(f"  Resolution: {width}x{height}, Duration: {duration_seconds}s")
        log.info(
            f"  Generation FPS: {config['generation_fps']}, Output FPS: {config['output_fps']}"
        )
        log.info(
            f"  Generated frames: {pixel_frames}, Interpolated to: {pixel_frames * config['interpolation_factor']}"
        )

        return (config, context_options, latent_frames, formatted_prompt)

    def _format_prompt(self, prompt, preset):
        """Format prompt with quality keywords based on preset"""
        quality_keywords = {
            "cinema_24fps": "cinematic quality, professional lighting, detailed",
            "smooth_30fps": "smooth motion, high detail, professional quality",
            "high_quality_60fps": "hyper-detailed, ultra high definition, crystal clear image quality",
            "debug_mode": "simple, basic colors, clear motion",
        }

        # Add quality keywords if not already in prompt
        quality = quality_keywords.get(preset, "")
        if quality and not any(
            kw in prompt.lower() for kw in quality.lower().split(", ")
        ):
            formatted_prompt = f"{prompt.strip()}, {quality}"
        else:
            formatted_prompt = prompt.strip()

        return formatted_prompt


NODE_CLASS_MAPPINGS_FPS = {
    "WanVideoUnifiedFPSConfig": WanVideoUnifiedFPSConfig,
    "WanVideoInterpolationDebugger": WanVideoInterpolationDebugger,
    "WanVideoEnhancedFPSSampler": WanVideoEnhancedFPSSampler,
    "WanVideoWorkflowPreset": WanVideoWorkflowPreset,
    "WanVideoContextOptionsFPS": WanVideoContextOptionsFPS,
    "WanVideoLatentInterpolator": WanVideoLatentInterpolator,
    "WanVideoTemporalEmphasis": WanVideoTemporalEmphasis,
    "WanVideoPerceiverResampler": WanVideoPerceiverResampler,
    "WanVideoFPSConfig": WanVideoFPSConfig,
    "WanVideoKeyframeConditioner": WanVideoKeyframeConditioner,
}

NODE_DISPLAY_NAME_MAPPINGS_FPS = {
    "WanVideoUnifiedFPSConfig": "WanVideo Unified FPS Config",
    "WanVideoInterpolationDebugger": "WanVideo Interpolation Debugger",
    "WanVideoEnhancedFPSSampler": "WanVideo Enhanced FPS Sampler",
    "WanVideoWorkflowPreset": "WanVideo Workflow Preset",
    "WanVideoContextOptionsFPS": "WanVideo ContextOptions FPS",
    "WanVideoLatentInterpolator": "WanVideo Latent Interpolator",
    "WanVideoTemporalEmphasis": "WanVideo Temporal Emphasis",
    "WanVideoPerceiverResampler": "WanVideo Perceiver Resampler",
    "WanVideoFPSConfig": "WanVideo FPS Config",
    "WanVideoKeyframeConditioner": "WanVideo Keyframe Conditioner",
}
