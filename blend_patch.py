def apply_blend_fix(prompt_index, c, section_size, text_embeds):
    """
    Apply blending between prompts at section boundaries
    """
    # Check if blending is enabled
    blend_width = text_embeds.get("blend_width", 0)
    
    if blend_width > 0 and prompt_index < len(text_embeds["prompt_embeds"]) - 1:
        # Calculate position within section (0-1)
        position = (max(c) % section_size) / section_size
        # Calculate blend zone (as proportion of section)
        blend_zone = blend_width / section_size
        
        if position > (1.0 - blend_zone):
            # In transition zone
            raw_ratio = (position - (1.0 - blend_zone)) / blend_zone
            
            # Apply curve
            blend_method = text_embeds.get("blend_method", "linear")
            if blend_method == "smooth":
                blend_ratio = raw_ratio * raw_ratio * (3 - 2 * raw_ratio)
            elif blend_method == "ease_in":
                blend_ratio = raw_ratio * raw_ratio
            elif blend_method == "ease_out":
                blend_ratio = raw_ratio * (2 - raw_ratio)
            else:
                blend_ratio = raw_ratio
            
            # Get embeddings
            current_embed = text_embeds["prompt_embeds"][prompt_index]
            next_embed = text_embeds["prompt_embeds"][prompt_index + 1]
            
            # Simple blend (if shapes match)
            if current_embed.shape == next_embed.shape:
                print(f"BLENDING prompts {prompt_index}→{prompt_index+1} at position {position:.2f}")
                return current_embed * (1 - blend_ratio) + next_embed * blend_ratio
    
    # Not in transition zone, return original embedding
    return text_embeds["prompt_embeds"][prompt_index]