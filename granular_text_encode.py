import torch
import logging
import comfy.model_management as mm
import traceback

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

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
                "fixed_sequence_length": ("INT", {"default": 0, "min": 0, "max": 30, 
                                      "tooltip": "Force all embeddings to same sequence length (0=disabled)"})
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", "INT")
    RETURN_NAMES = ("text_embeds", "embedding_count")
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Creates granular text embeddings with smooth transitions between main prompts"

    def normalize_embedding_shapes(self, embeddings, target_length=None):
        """Normalize embeddings to have consistent shapes"""
        if not embeddings or len(embeddings) < 2:
            return embeddings
            
        # Get device and dtype info
        device = embeddings[0].device
        dtype = embeddings[0].dtype
        
        # Get feature dimension (last dimension)
        feature_dim = embeddings[0].shape[-1]
        
        # Get sequence lengths 
        seq_lengths = [e.shape[0] for e in embeddings]
        log.info(f"DEBUG: Original embedding shapes: {seq_lengths}")
        
        # Set target length to max if not specified
        if target_length is None or target_length <= 0:
            target_length = max(seq_lengths)
            log.info(f"DEBUG: Using max sequence length: {target_length}")
        
        # Process each embedding
        normalized = []
        for i, embed in enumerate(embeddings):
            curr_len = embed.shape[0]
            
            if curr_len < target_length:
                # Pad with zeros
                log.info(f"DEBUG: Padding embedding {i} from {curr_len} to {target_length}")
                padding = torch.zeros(
                    target_length - curr_len, feature_dim, device=device, dtype=dtype
                )
                normalized.append(torch.cat([embed, padding], dim=0))
            elif curr_len > target_length:
                # Truncate to target length
                log.info(f"DEBUG: Truncating embedding {i} from {curr_len} to {target_length}")
                normalized.append(embed[:target_length])
            else:
                # Already the right length
                normalized.append(embed)
                
        # Double-check result
        final_shapes = [e.shape for e in normalized]
        log.info(f"DEBUG: Normalized embedding shapes: {final_shapes}")
        
        return normalized

    def process(self, t5, base_prompts, negative_prompt, transition_strength, transition_count,
                force_offload=True, model_to_offload=None, fixed_sequence_length=0):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        if model_to_offload is not None:
            log.info(f"Moving video model to {offload_device}")
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
        
        log.info(f"Generated {len(all_prompts)} prompts: {all_prompts}")
        
        # Move encoder to device for processing
        encoder.model.to(device)
        
        # Encode all prompts
        try:
            with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype, enabled=True):
                context = encoder(all_prompts, device)
                context_null = encoder([negative_prompt], device)
                
            # Print shape info for debugging
            log.info(f"DEBUG: Embedding shapes before normalization: {[t.shape for t in context]}")
            
            # Apply shape normalization if enabled
            if fixed_sequence_length > 0:
                log.info(f"DEBUG: Normalizing embeddings to fixed length: {fixed_sequence_length}")
                context = self.normalize_embedding_shapes(context, fixed_sequence_length)
            elif any(t.shape[0] != context[0].shape[0] for t in context):
                log.info(f"DEBUG: Embeddings have inconsistent shapes, normalizing...")
                context = self.normalize_embedding_shapes(context)
                
        except Exception as e:
            log.info(f"ERROR during encoding: {str(e)}")
            log.info(traceback.format_exc())
            # Fallback to basic encoding if something goes wrong
            with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype, enabled=True):
                context = encoder([all_prompts[0]], device)
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