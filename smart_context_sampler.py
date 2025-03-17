import torch
import math
from tqdm import tqdm
import comfy.model_management as mm

from .wanvideo.modules.clip import CLIPModel
from .wanvideo.modules.model import WanModel, rope_params
from .wanvideo.modules.t5 import T5EncoderModel
from .wanvideo.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .wanvideo.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from .enhance_a_video.globals import enable_enhance, disable_enhance, set_enhance_weight, set_num_frames

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file, save_torch_file, ProgressBar, common_upscale
import comfy.model_base
import comfy.latent_formats
from comfy.clip_vision import clip_preprocess, ClipVisionModel
from comfy.sd import load_lora_for_models
from .entity_tracker import EntityTracker

class WanVideoSmartSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "text_embeds": ("WANVIDEOTEXTEMBEDS", ),
                "image_embeds": ("WANVIDIMAGE_EMBEDS", ),
                "steps": ("INT", {"default": 30, "min": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Moves the model to the offload device after sampling"}),
                "scheduler": (["unipc", "dpm++", "dpm++_sde", "euler"],
                    {"default": 'unipc'}),
                "enable_entity_tracking": ("BOOLEAN", {"default": True,
                                         "tooltip": "Enable entity tracking for smoother transitions"}),
                "transition_weight": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "How strongly to apply transitions between windows"}),
                "riflex_freq_index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "Frequency index for RIFLEX, disabled when 0, default 6. Allows for new frames to be generated after without looping"}),
            },
            "optional": {
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"} ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feta_args": ("FETAARGS", ),
                "context_options": ("WANVIDCONTEXT", ),
                "teacache_args": ("TEACACHEARGS", ),
                "flowedit_args": ("FLOWEDITARGS", ),
                "batched_cfg": ("BOOLEAN", {"default": False, "tooltip": "Batch cond and uncond for faster sampling, possibly faster on some hardware, uses more memory"}),
                "embedding_count": ("INT", {"default": 1, "tooltip": "Number of embeddings in text_embeds"}),
            }
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Generates video with smart transitions between context windows"

    def __init__(self):
        self.entity_tracker = EntityTracker()
        self.latest_attn_weights = []
        self.window_features = {}

    def process(self, model, text_embeds, image_embeds, shift, steps, cfg, seed,
               enable_entity_tracking, transition_weight, force_offload=True, scheduler="unipc",
               samples=None, denoise_strength=1.0, feta_args=None, context_options=None,
               teacache_args=None, flowedit_args=None, riflex_freq_index=0, batched_cfg=False, embedding_count=1):
        patcher = model
        model = model
        transformer = model.diffusion_model

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        # Adjust steps based on denoise_strength
        steps = int(steps/denoise_strength)

        # Import required modules based on scheduler
        if scheduler == 'unipc':
            from .wanvideo.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler as Scheduler
        elif scheduler == 'dpm++_sde':
            from .wanvideo.utils.fm_solvers import FlowDPMSolverMultistepScheduler as Scheduler
            algorithm_type = "sde-dpmsolver++"
        elif scheduler == 'dpm++':
            from .wanvideo.utils.fm_solvers import FlowDPMSolverMultistepScheduler as Scheduler
            algorithm_type = "dpmsolver++"
        else:  # euler
            from diffusers.schedulers import FlowMatchEulerDiscreteScheduler as Scheduler
            algorithm_type = None

        # Create and configure scheduler
        if scheduler == 'unipc':
            sample_scheduler = Scheduler(
                num_train_timesteps=1000,
                shift=shift,
                use_dynamic_shifting=False
            )
            sample_scheduler.set_timesteps(steps, device=device, shift=shift)
        elif 'dpm++' in scheduler:
            sample_scheduler = Scheduler(
                num_train_timesteps=1000,
                shift=shift,
                use_dynamic_shifting=False,
                algorithm_type=algorithm_type
            )
            from .wanvideo.utils.fm_solvers import get_sampling_sigmas, retrieve_timesteps
            sampling_sigmas = get_sampling_sigmas(steps, shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)
        else:
            sample_scheduler = Scheduler(
                num_train_timesteps=1000,
                shift=shift,
                use_dynamic_shifting=False
            )
            from .wanvideo.utils.fm_solvers import get_sampling_sigmas, retrieve_timesteps
            sampling_sigmas = get_sampling_sigmas(steps, shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)

        # Apply denoise_strength
        if denoise_strength < 1.0:
            steps = int(steps * denoise_strength)
            timesteps = timesteps[-(steps + 1):]

        # Get context settings
        if context_options is not None:
            context_frames = context_options["context_frames"]
            context_stride = context_options.get("context_stride", 3)
            context_overlap = context_options.get("context_overlap", 16)
            context_schedule = context_options.get("context_schedule", "uniform_standard")

            # Configure entity tracker for context windowing
            if enable_entity_tracking:
                self.entity_tracker.setup_for_context(
                    context_frames=context_frames,
                    context_overlap=context_overlap,
                    context_schedule=context_schedule
                )
        else:
            # Default values
            context_frames = 81
            context_overlap = 16
            context_schedule = "uniform_standard"

        # Setup for image conditioning
        image_cond = None
        clip_fea = None
        if transformer.model_type == "i2v":
            lat_h = image_embeds.get("lat_h", None)
            lat_w = image_embeds.get("lat_w", None)
            image_cond = image_embeds.get("image_embeds", None)
            clip_fea = image_embeds.get("clip_context", None)

            if lat_h is None or lat_w is None:
                # Try to get values from control_images if it's control conditioning
                control_images = image_embeds.get("control_images", None)
                if control_images is not None:
                    lat_h = control_images.shape[3]
                    lat_w = control_images.shape[4]
                else:
                    raise ValueError("Cannot determine latent dimensions for I2V model")

        # Setup for T2V
        latent_shape = None
        if transformer.model_type == "t2v":
            target_shape = image_embeds.get("target_shape", None)
            if target_shape is None:
                raise ValueError("Target shape must be provided for T2V model")
            latent_shape = target_shape

        # Generate initial noise
        seed_g = torch.Generator(device=torch.device("cuda"))
        seed_g.manual_seed(seed)

        # Create the initial latents
        if transformer.model_type == "i2v":
            latent_video_length = (context_frames - 1) // 4 + 1
            noise = torch.randn(
                16,  # channels
                latent_video_length,
                lat_h,
                lat_w,
                dtype=torch.float32,
                generator=seed_g,
                device=device
            )
        else:  # t2v
            noise = torch.randn(
                latent_shape[0],
                latent_shape[1],
                latent_shape[2],
                latent_shape[3],
                dtype=torch.float32,
                generator=seed_g,
                device=device
            )
            latent_video_length = latent_shape[1]

        # Set up sequence length for transformer
        seq_len = math.ceil((image_embeds.get("lat_h", latent_shape[2]) *
                             image_embeds.get("lat_w", latent_shape[3])) / 4 * latent_video_length)

        # Set up rope frequencies
        d = transformer.dim // transformer.num_heads
        from .wanvideo.modules.model import rope_params
        freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6), L_test=latent_video_length),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1).to("cuda")

        # Process window generation
        # If we're using context windowing from existing option
        if context_options is not None and enable_entity_tracking:
            # Use the existing context windowing with enhanced entity tracking
            from context import get_context_scheduler
            context_generator = get_context_scheduler(context_schedule)

            # Setup progress tracking
            total_windows = 0
            for i in range(steps):
                total_windows += sum(1 for _ in context_generator(
                    i, steps, latent_video_length,
                    (context_frames - 1) // 4 + 1, context_stride,
                    context_overlap // 4))

            pbar = tqdm(total=total_windows, desc="Generating with entity tracking")

            # Initialize latents
            latents = noise.to(device)
            if samples is not None and denoise_strength < 1.0:
                latent_timestep = timesteps[:1].to(noise)
                latents = latents * latent_timestep / 1000 + (1 - latent_timestep / 1000) * samples["samples"].squeeze(0).to(noise)

            # Setup entity tracking
            self.entity_tracker.continuity_strength = transition_weight
            window_features = {}
            window_count = 0

            # For each denoising step
            for idx, t in enumerate(tqdm(timesteps, desc="Denoising steps")):
                # Get context windows for this step
                context_queue = list(context_generator(
                    idx, steps, latent_video_length,
                    (context_frames - 1) // 4 + 1, context_stride,
                    context_overlap // 4))

                # Create storage for aggregating predictions across windows
                latent_model_input = latents
                noise_pred = torch.zeros_like(latent_model_input)
                counter = torch.zeros_like(latent_model_input[:, 0:1])

                # Step through each context window
                for c in context_queue:
                    window_id = f"window_{window_count}"
                    window_count += 1

                    # Select appropriate embedding based on position
                    if embedding_count > 1:
                        # Find which section this window belongs to
                        section_size = latent_video_length / embedding_count
                        prompt_index = min(int(max(c) / section_size), embedding_count - 1)
                    else:
                        prompt_index = 0

                    # Prepare window inputs
                    partial_latent = latent_model_input[:, :, c, :, :]
                    partial_img_cond = None
                    if image_cond is not None:
                        partial_img_cond = image_cond[:, c, :, :]

                    # Process this window
                    timestep = torch.tensor([t]).to(device)

                    # Capture attention for entity tracking
                    if idx < 2:  # Only need to capture in early steps
                        self.latest_attn_weights = []
                        self.entity_tracker.setup_attention_hooks(
                            transformer, attention_mode=transformer.attention_mode)

                    # Run model prediction
                    with torch.autocast(device_type=mm.get_autocast_device(device),
                                        dtype=model.get("dtype", torch.float32), enabled=True):
                        # Base parameters for model
                        base_params = {
                            'clip_fea': clip_fea,
                            'seq_len': seq_len,
                            'device': device,
                            'freqs': freqs,
                            't': timestep,
                            'current_step': idx,
                            'current_step_percentage': idx / len(timesteps),
                            'y': [partial_img_cond] if partial_img_cond is not None else None,
                        }

                        # Get positive embedding for this window
                        positive = text_embeds["prompt_embeds"][prompt_index]

                        # Process window prediction
                        if not batched_cfg:
                            # Run conditional and unconditional passes separately
                            noise_pred_cond, teacache_state_cond = transformer(
                                [partial_latent],
                                context=[positive],
                                is_uncond=False,
                                **base_params
                            )
                            noise_pred_cond = noise_pred_cond[0]

                            if math.isclose(cfg, 1.0):
                                window_pred = noise_pred_cond
                            else:
                                noise_pred_uncond, _ = transformer(
                                    [partial_latent],
                                    context=text_embeds["negative_prompt_embeds"],
                                    is_uncond=True,
                                    **base_params
                                )
                                window_pred = noise_pred_uncond[0] + cfg * (noise_pred_cond - noise_pred_uncond[0])
                        else:
                            # Run batched conditional/unconditional
                            latent_double = torch.cat([partial_latent, partial_latent])
                            combined_context = [positive] + text_embeds["negative_prompt_embeds"]

                            noise_preds, teacache_state_cond = transformer(
                                [latent_double],
                                context=combined_context,
                                is_uncond=False,
                                **base_params
                            )

                            noise_pred_cond, noise_pred_uncond = noise_preds[0].chunk(2)
                            window_pred = noise_pred_uncond + cfg * (noise_pred_cond - noise_pred_uncond)

                    # Clean up hooks
                    if idx < 2:
                        self.entity_tracker.cleanup_hooks()

                    # Extract entity information if this is the first step
                    if idx == 0:
                        if hasattr(self, 'latest_attn_weights') and self.latest_attn_weights:
                            entity_maps = self.entity_tracker.extract_entity_maps(self.latest_attn_weights)
                            motion_vectors = self.entity_tracker.compute_motion_vectors(entity_maps)
                            window_features[window_id] = {
                                'entity_maps': entity_maps,
                                'motion_vectors': motion_vectors
                            }

                    # Create window mask for blending - standard version
                    window_mask = torch.ones_like(noise_pred[:, :, 0:1])

                    # Enhance with entity tracking if we have data from previous windows
                    if idx > 0 and window_count > 1:
                        prev_window_id = f"window_{window_count-2}"  # Skip to the true previous window
                        if prev_window_id in window_features:
                            prev_features = window_features[prev_window_id]

                            # Check for overlap with previous window
                            prev_c_set = set(range(c[0] - context_overlap//4, c[0]))

                            # If we have overlap with previous window, apply entity-aware blending
                            if prev_c_set:
                                # Compute entity-aware blend weights
                                blend_weights = self.entity_tracker.compute_continuity_masks(
                                    latent_model_input[:, :, c[0] - context_overlap//4:c[0]],
                                    partial_latent,
                                    prev_features['entity_maps'],
                                    prev_features['motion_vectors']
                                )

                                # Apply blending at overlap
                                for i in range(min(context_overlap//4, len(c))):
                                    window_mask[:, :, i:i+1] = blend_weights[i].unsqueeze(0).unsqueeze(0)

                    # Apply window mask
                    for i, frame_idx in enumerate(c):
                        noise_pred[:, :, frame_idx] += window_pred[:, :, i] * window_mask[:, :, i:i+1]
                        counter[:, :, frame_idx] += window_mask[:, :, i:i+1]

                    pbar.update(1)

                # Normalize predictions by counter
                noise_pred = noise_pred / (counter + 1e-8)

                # Update latents with scheduler step
                latents = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g
                )[0].squeeze(0)

            # Return the final latents
            if force_offload:
                transformer.to(offload_device)
                mm.soft_empty_cache()

            # Final result
            return ({"samples": latents.unsqueeze(0)},)

        # Regular processing without context windowing enhancement
        else:
            # Use the default WanVideoSampler approach but with entity awareness at window boundaries
            # This is more complex and would require modifying the standard WanVideoSampler logic
            # For this implementation, we'll delegate to the regular sampler
            from .nodes import WanVideoSampler

            # Create instance of regular sampler
            regular_sampler = WanVideoSampler()

            # Call original process method with our parameters
            result = regular_sampler.process(
                model=model, text_embeds=text_embeds, image_embeds=image_embeds,
                shift=shift, steps=steps, cfg=cfg, seed=seed,
                force_offload=force_offload, scheduler=scheduler,
                samples=samples, denoise_strength=denoise_strength,
                feta_args=feta_args, context_options=context_options,
                teacache_args=teacache_args, flowedit_args=flowedit_args, riflex_freq_index=riflex_freq_index
            )

            return result
