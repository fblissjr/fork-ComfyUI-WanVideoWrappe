import torch
import logging
import comfy.model_management as mm

class WanVideoGranularTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "t5": ("WANTEXTENCODER",),
            "base_prompts": ("STRING", {"default": "", "multiline": True, 
                              "tooltip": "Main scene prompts separated by | character"}),
            "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            "transition_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "How strongly to blend prompts at transitions"}),
            "transition_count": ("INT", {"default": 3, "min": 1, "max": 10,
                                "tooltip": "Number of transition embeddings between main prompts"}),
            },
            "optional": {
                "force_offload": ("BOOLEAN", {"default": True}),
                "model_to_offload": ("WANVIDEOMODEL", {"tooltip": "Model to move to offload_device before encoding"}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", "INT")
    RETURN_NAMES = ("text_embeds", "embedding_count")
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Creates granular text embeddings with smooth transitions between main prompts"

    def process(self, t5, base_prompts, negative_prompt, transition_strength, transition_count,
                force_offload=True, model_to_offload=None):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        if model_to_offload is not None:
            logging.info(f"Moving video model to {offload_device}")
            model_to_offload.model.to(offload_device)
            mm.soft_empty_cache()

        encoder = t5["model"]
        dtype = t5["dtype"]

        # Handle pipe-delimited prompts from existing workflows
        if "|" in base_prompts:
            main_prompts = [p.strip() for p in base_prompts.split('|')]
        else:
            # Handle as single prompt
            main_prompts = [base_prompts]
            
        if len(main_prompts) < 2:
            # If only one prompt, duplicate it to ensure we have at least two
            main_prompts.append(main_prompts[0])
        
        # Generate all prompts including transitions
        all_prompts = []
        
        for i in range(len(main_prompts)):
            # Add the main prompt
            all_prompts.append(main_prompts[i])
            
            # Add transition prompts between main prompts (except after the last one)
            if i < len(main_prompts) - 1:
                current_prompt = main_prompts[i]
                next_prompt = main_prompts[i+1]
                
                # Create transition prompts
                for t in range(transition_count):
                    # Calculate interpolation factor
                    blend = (t + 1) / (transition_count + 1) * transition_strength
                    
                    # Create transition prompt by combining both prompts
                    if blend < 0.1:
                        transition_prompt = f"{current_prompt}, transitioning to {next_prompt}"
                    else:
                        transition_prompt = f"{current_prompt} ({1-blend:.2f}), {next_prompt} ({blend:.2f})"
                    all_prompts.append(transition_prompt)
        
        logging.info(f"Generated {len(all_prompts)} prompts: {all_prompts}")
        
        # Move encoder to device for processing
        encoder.model.to(device)
        
        # Encode all prompts
        with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype, enabled=True):
            context = encoder(all_prompts, device)
            context_null = encoder([negative_prompt], device)

        # Move everything to the right device
        context = [t.to(device) for t in context]
        context_null = [t.to(device) for t in context_null]

        if force_offload:
            encoder.model.to(offload_device)
            mm.soft_empty_cache()

        prompt_embeds_dict = {
                "prompt_embeds": context,
                "negative_prompt_embeds": context_null,
            }
            
        return (prompt_embeds_dict, len(all_prompts))