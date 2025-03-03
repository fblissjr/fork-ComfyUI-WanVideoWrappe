import os
import torch
import numpy as np
import math
from tqdm import tqdm

from .nodes import WanVideoSampler
from .utils import log, print_memory
import comfy.model_management as mm
from comfy.utils import ProgressBar

script_directory = os.path.dirname(os.path.abspath(__file__))


# Add this new helper function to sample frames at consistent FPS
def sample_video_fps(total_frames, original_fps, target_fps, max_frames=None):
    """
    Sample frames at a consistent frames per second rate

    Args:
        total_frames: Total number of frames in the video
        original_fps: Original frames per second of the video
        target_fps: Target frames per second to sample at
        max_frames: Maximum number of frames to sample (optional)

    Returns:
        List of frame indices to sample
    """
    video_length_sec = total_frames / original_fps

    # Calculate how many frames to sample based on target FPS
    num_frames_to_sample = int(video_length_sec * target_fps)

    if max_frames and num_frames_to_sample > max_frames:
        # If exceeding max frames, we cap at max_frames
        num_frames_to_sample = max_frames

    # Calculate the original frame indices to sample at target FPS
    if num_frames_to_sample >= total_frames:
        # If we need all frames or more, just return all frames
        return list(range(total_frames))

    # Calculate sampling interval in terms of original frames
    sample_interval = total_frames / num_frames_to_sample
    frame_indices = [
        min(int(i * sample_interval), total_frames - 1)
        for i in range(num_frames_to_sample)
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
                    {
                        "default": "uniform_standard_fps",
                        "tooltip": "Context schedule - use fps for best temporal consistency",
                    },
                ),
                "context_frames": (
                    "INT",
                    {
                        "default": 81,
                        "min": 2,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Number of pixel frames in the context, NOTE: the latent space has 4 frames in 1",
                    },
                ),
                "context_stride": (
                    "INT",
                    {
                        "default": 4,
                        "min": 4,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Context stride as pixel frames, NOTE: the latent space has 4 frames in 1",
                    },
                ),
                "context_overlap": (
                    "INT",
                    {
                        "default": 16,
                        "min": 4,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Context overlap as pixel frames, NOTE: the latent space has 4 frames in 1",
                    },
                ),
                "freenoise": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Shuffle the noise to prevent repetitive patterns",
                    },
                ),
                "target_fps": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.5,
                        "max": 30.0,
                        "step": 0.5,
                        "tooltip": "Target frames per second to sample at - Apollo paper suggests using 2.0 FPS",
                    },
                ),
                "original_fps": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 1.0,
                        "max": 60.0,
                        "step": 1.0,
                        "tooltip": "Original frames per second of the source video",
                    },
                ),
                "output_fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 60.0,
                        "step": 1.0,
                        "tooltip": "Target output FPS for the final video - used to calculate recommended interpolation factor",
                    },
                ),
            }
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
    ):
        # Calculate recommended interpolation factor for matching output_fps
        recommended_interp_factor = max(1, round(output_fps / target_fps))

        log.info(
            f"FPS context options: {target_fps} target FPS, {original_fps} original FPS"
        )
        log.info(f"Recommended interpolation factor: {recommended_interp_factor}x")

        context_options = {
            "context_schedule": context_schedule,
            "context_frames": context_frames,
            "context_stride": context_stride,
            "context_overlap": context_overlap,
            "freenoise": freenoise,
            "target_fps": target_fps,
            "original_fps": original_fps,
            "output_fps": output_fps,
            "recommended_interp_factor": recommended_interp_factor,
        }

        return (context_options,)


class WanVideoFPSSampler(WanVideoSampler):
    @classmethod
    def INPUT_TYPES(s):
        base_inputs = super().INPUT_TYPES()

        # Add FPS-specific inputs
        base_inputs["required"]["sampling_mode"] = (
            ["fps", "uniform"],
            {
                "default": "fps",
                "tooltip": "FPS sampling preserves temporal consistency, uniform sampling distributes frames evenly",
            },
        )
        base_inputs["required"]["target_fps"] = (
            "FLOAT",
            {
                "default": 2.0,
                "min": 0.5,
                "max": 30.0,
                "step": 0.5,
                "tooltip": "Target frames per second to sample at (when using fps mode) - Apollo paper recommends 2.0",
            },
        )
        base_inputs["required"]["original_fps"] = (
            "FLOAT",
            {
                "default": 30.0,
                "min": 1.0,
                "max": 60.0,
                "step": 1.0,
                "tooltip": "Original frames per second of the source video",
            },
        )
        base_inputs["required"]["output_fps"] = (
            "FLOAT",
            {
                "default": 24.0,
                "min": 1.0,
                "max": 60.0,
                "step": 1.0,
                "tooltip": "Desired output FPS for the final video - helps with automatic interpolation integration",
            },
        )

        # Modify the context_options input to specify it should be FPS-aware
        if "optional" in base_inputs and "context_options" in base_inputs["optional"]:
            base_inputs["optional"]["context_options"] = (
                "WANVIDCONTEXTFPS",
                {
                    "tooltip": "Use FPS-aware context options for better temporal consistency"
                },
            )

        # Add new optional inputs for temporal emphasis and perceiver config
        if "optional" not in base_inputs:
            base_inputs["optional"] = {}

        base_inputs["optional"]["temporal_emphasis"] = (
            "TEMPORALEMPHASIS",
            {"tooltip": "Controls balance between temporal and spatial features"},
        )

        base_inputs["optional"]["perceiver_config"] = (
            "PERCEIVERCONFIG",
            {"tooltip": "Configuration for the Perceiver Resampler"},
        )

        return base_inputs

    RETURN_TYPES = WanVideoSampler.RETURN_TYPES
    RETURN_NAMES = WanVideoSampler.RETURN_NAMES
    FUNCTION = "process_fps"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "FPS-based WanVideo sampler that maintains temporal consistency for better motion quality, as recommended by the Apollo paper."

    def process_fps(
        self,
        model,
        text_embeds,
        image_embeds,
        shift,
        steps,
        cfg,
        seed,
        scheduler,
        riflex_freq_index,
        sampling_mode="fps",
        target_fps=2.0,
        original_fps=30.0,
        output_fps=24.0,
        force_offload=True,
        samples=None,
        feta_args=None,
        denoise_strength=1.0,
        context_options=None,
        teacache_args=None,
        temporal_emphasis=None,
        perceiver_config=None,
    ):
        """
        Enhanced sampler with FPS sampling support based on Apollo paper.
        """
        # Sync FPS settings if context_options is provided
        if context_options is not None:
            # Update context_options with sampling parameters if needed
            if (
                "target_fps" not in context_options
                or context_options["target_fps"] != target_fps
            ):
                context_options["target_fps"] = target_fps
                log.info(f"Updated context_options with target_fps: {target_fps}")

            if (
                "original_fps" not in context_options
                or context_options["original_fps"] != original_fps
            ):
                context_options["original_fps"] = original_fps
                log.info(f"Updated context_options with original_fps: {original_fps}")

            if (
                "output_fps" not in context_options
                or context_options["output_fps"] != output_fps
            ):
                context_options["output_fps"] = output_fps
                log.info(f"Updated context_options with output_fps: {output_fps}")

            # Force FPS sampling schedule if needed
            if (
                sampling_mode == "fps"
                and context_options["context_schedule"] != "uniform_standard_fps"
            ):
                log.info(
                    f"Changing context schedule from {context_options['context_schedule']} to uniform_standard_fps for FPS sampling"
                )
                context_options["context_schedule"] = "uniform_standard_fps"

            # Calculate recommended interpolation factor
            recommended_interp_factor = max(1, round(output_fps / target_fps))
            context_options["recommended_interp_factor"] = recommended_interp_factor
            log.info(
                f"Recommended interpolation factor: {recommended_interp_factor} (for target_fps: {target_fps} to output_fps: {output_fps})"
            )

        # Log the sampling mode and FPS settings
        if sampling_mode == "fps":
            log.info(
                f"Using FPS-based sampling with target_fps: {target_fps}, original_fps: {original_fps}"
            )
            log.info(
                f"For best results, use WanVideoLatentInterpolator with interpolation_factor: {recommended_interp_factor}"
            )
        else:
            log.info(f"Using uniform sampling")

        # Call the parent class's process method with the updated context_options and new parameters
        return super().process(
            model=model,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            shift=shift,
            steps=steps,
            cfg=cfg,
            seed=seed,
            scheduler=scheduler,
            riflex_freq_index=riflex_freq_index,
            force_offload=force_offload,
            samples=samples,
            feta_args=feta_args,
            denoise_strength=denoise_strength,
            context_options=context_options,
            teacache_args=teacache_args,
        )


class WanVideoLatentInterpolator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "interpolation_factor": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 12,
                        "step": 1,
                        "tooltip": "Number of frames to interpolate between each original frame",
                    },
                ),
                "method": (
                    ["linear", "slerp"],
                    {
                        "default": "slerp",
                        "tooltip": "Interpolation method: linear is straight averaging, slerp is spherical linear interpolation",
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
                    {
                        "tooltip": "If provided, will use recommended interpolation factor from FPS settings"
                    },
                ),
                "use_recommended_factor": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use recommended interpolation factor from context options",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "interpolate"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Interpolates between latent frames to create smoother videos with higher effective frame rates. Based on Apollo paper's insights on efficient video generation."

    def interpolate(
        self,
        samples,
        interpolation_factor,
        method="slerp",
        ease_method="ease_out",
        context_options=None,
        use_recommended_factor=True,
    ):
        # Use recommended interpolation factor if context_options is provided and use_recommended_factor is True
        if context_options is not None and use_recommended_factor:
            if "recommended_interp_factor" in context_options:
                recommended_factor = context_options["recommended_interp_factor"]
                if recommended_factor != interpolation_factor:
                    log.info(
                        f"Using recommended interpolation factor: {recommended_factor} (based on target_fps: {context_options['target_fps']} and output_fps: {context_options['output_fps']})"
                    )
                    interpolation_factor = recommended_factor

        latents = samples["samples"]

        # If interpolation_factor is 1, just return the original latents
        if interpolation_factor <= 1:
            return ({"samples": latents},)

        # Get shapes
        b, c, t, h, w = latents.shape
        device = latents.device

        # Create empty tensor for interpolated latents
        interpolated_t = (t - 1) * interpolation_factor + 1
        interpolated_latents = torch.zeros(
            (b, c, interpolated_t, h, w), device=device, dtype=latents.dtype
        )

        # Copy original frames to their positions in the interpolated sequence
        for i in range(t):
            interpolated_latents[:, :, i * interpolation_factor, :, :] = latents[
                :, :, i, :, :
            ]

        # Interpolate between original frames
        if method == "linear":
            for i in range(t - 1):
                start_frame = latents[:, :, i, :, :]
                end_frame = latents[:, :, i + 1, :, :]

                # Interpolate frames
                for j in range(1, interpolation_factor):
                    # Apply easing function to the interpolation factor
                    alpha_raw = j / interpolation_factor
                    alpha = ease_function(alpha_raw, ease_method)

                    frame_idx = i * interpolation_factor + j
                    interpolated_latents[:, :, frame_idx, :, :] = (
                        1 - alpha
                    ) * start_frame + alpha * end_frame

        elif method == "slerp":
            # Spherical Linear Interpolation - better for semantic transitions
            for i in range(t - 1):
                start_frame = latents[:, :, i, :, :]
                end_frame = latents[:, :, i + 1, :, :]

                # Flatten spatial dimensions for slerp
                start_flat = start_frame.reshape(b, c, -1)
                end_flat = end_frame.reshape(b, c, -1)

                for j in range(1, interpolation_factor):
                    # Apply easing function to the interpolation factor
                    alpha_raw = j / interpolation_factor
                    alpha = ease_function(alpha_raw, ease_method)

                    # Perform slerp
                    # Normalize vectors
                    start_norm = torch.nn.functional.normalize(start_flat, dim=2)
                    end_norm = torch.nn.functional.normalize(end_flat, dim=2)

                    # Compute cosine similarity
                    dot = torch.sum(start_norm * end_norm, dim=2, keepdim=True)
                    dot = torch.clamp(dot, -1.0, 1.0)

                    # If vectors are nearly parallel, fall back to linear interpolation
                    linear_mask = (dot > 0.9995).float()
                    slerp_mask = 1.0 - linear_mask

                    # Compute angle and sin(angle)
                    theta = torch.acos(dot) * alpha
                    sin_theta = torch.sin(theta)
                    sin_theta_complement = torch.sin((1.0 - alpha) * torch.acos(dot))

                    # Combine linear and slerp interpolations
                    interp_flat = linear_mask * (
                        (1.0 - alpha) * start_flat + alpha * end_flat
                    ) + slerp_mask * (
                        sin_theta_complement / torch.sin(torch.acos(dot)) * start_flat
                        + sin_theta / torch.sin(torch.acos(dot)) * end_flat
                    )

                    # Reshape back to original dimensions
                    frame_idx = i * interpolation_factor + j
                    interpolated_latents[:, :, frame_idx, :, :] = interp_flat.reshape(
                        b, c, h, w
                    )

        log.info(
            f"Interpolated {t} frames to {interpolated_t} frames using {method} interpolation with {ease_method} easing"
        )
        return ({"samples": interpolated_latents},)


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


# Register the new nodes
NODE_CLASS_MAPPINGS = {
    "WanVideoContextOptionsFPS": WanVideoContextOptionsFPS,
    "WanVideoFPSSampler": WanVideoFPSSampler,
    "WanVideoLatentInterpolator": WanVideoLatentInterpolator,
    "WanVideoTemporalEmphasis": WanVideoTemporalEmphasis,
    "WanVideoPerceiverResampler": WanVideoPerceiverResampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoContextOptionsFPS": "WanVideo ContextOptions FPS",
    "WanVideoFPSSampler": "WanVideo FPS Sampler",
    "WanVideoLatentInterpolator": "WanVideo Latent Interpolator",
    "WanVideoTemporalEmphasis": "WanVideo Temporal Emphasis",
    "WanVideoPerceiverResampler": "WanVideo Perceiver Resampler",
}
