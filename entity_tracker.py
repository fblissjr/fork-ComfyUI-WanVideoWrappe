import torch
import numpy as np
from einops import rearrange

class EntityTracker:
    """
    Tracks entities across context windows using attention map analysis
    to ensure consistency in appearance and motion.
    """
    
    def __init__(self):
        self.entities = {}  # Store entity data across windows
        self.window_cache = {}  # Cache previous window features
        self.motion_vectors = {}  # Track entity movements
        self.continuity_strength = 1.0  # Adjustable strength for continuity
        self.hooks = []
        self.attention_mode = "sdpa"
        self.context_aware = False
        self.context_frames = 81
        self.context_overlap = 16
    
    def setup_for_context(self, context_frames=81, context_overlap=16, context_schedule="uniform_standard"):
        """Configure entity tracker for specific context window settings"""
        self.context_aware = True
        self.context_frames = context_frames
        self.context_overlap = context_overlap
        self.context_schedule = context_schedule
    
    def extract_entity_maps(self, attn_weights, layer_idx=-3):
        """
        Extract entity saliency maps from DiT attention weights.
        
        Args:
            attn_weights: Output attention weights from DiT
            layer_idx: Which layer to use (negative indexing from end)
            
        Returns:
            Dictionary of entity saliency maps
        """
        if not attn_weights:
            return {}
        
        # For larger models, we might need to adjust the layer selection
        if len(attn_weights) > 30:  # 14B model has more layers
            layer_idx = -6  # Use a different layer for larger models
            
        # Extract attention outputs from specified layer
        attn_output = attn_weights[layer_idx]
        
        # Get output attention vectors
        output_vectors = attn_output['output']  # [B, seq_len, dim]
        
        # Calculate self-similarity to find clusters (entities)
        similarity = torch.matmul(output_vectors, output_vectors.transpose(1, 2))
        
        # Threshold to find distinct entity clusters (adaptive threshold)
        threshold = similarity.mean() + 0.5 * similarity.std()
        entity_masks = similarity > threshold
        
        # For each major cluster, create an entity map
        entity_maps = {}
        num_clusters = min(5, (entity_masks.sum(dim=1) > 3).sum().item())  # Limit to top 5 entities
        
        for i in range(num_clusters):
            # Find center of cluster i
            cluster_scores = similarity.sum(dim=2)
            max_idx = torch.argmax(cluster_scores).item()
            
            # Get similarity to this point as entity map
            entity_map = similarity[:, max_idx, :]
            entity_maps[f"entity_{i}"] = entity_map
            
            # Mask out this cluster to find the next one
            similarity[:, :, similarity[0, max_idx] > threshold/2] = 0
        
        return entity_maps
    
    def compute_motion_vectors(self, entity_maps, frames_per_window=16):
        """
        Compute motion vectors for each entity based on position changes.
        
        Args:
            entity_maps: Dictionary of entity saliency maps
            frames_per_window: Number of frames per window
            
        Returns:
            Dictionary of entity motion vectors
        """
        motion_vectors = {}
        
        for entity_id, entity_map in entity_maps.items():
            # Reshape to [frames, height, width]
            map_shape = entity_map.shape[0]
            # Guess the height and width
            h_guess = int(np.sqrt(map_shape / frames_per_window))
            w_guess = map_shape // (frames_per_window * h_guess)
            
            try:
                map_3d = entity_map.reshape(frames_per_window, h_guess, w_guess)
            except:
                # If reshaping fails, use a fallback approach
                print(f"Reshaping failed for entity {entity_id}, using fallback")
                continue
            
            # Track position per frame by finding center of mass
            positions = []
            for f in range(frames_per_window):
                # Get center of mass
                h_indices = torch.arange(map_3d.shape[1], device=map_3d.device)
                w_indices = torch.arange(map_3d.shape[2], device=map_3d.device)
                
                # Avoid division by zero
                total_mass = map_3d[f].sum()
                if total_mass > 0:
                    h_pos = (map_3d[f] * h_indices.unsqueeze(1)).sum() / total_mass
                    w_pos = (map_3d[f] * w_indices.unsqueeze(0)).sum() / total_mass
                    positions.append((h_pos.item(), w_pos.item()))
                else:
                    positions.append((h_guess/2, w_guess/2))  # Center
            
            # Compute velocity by finite difference
            velocities = []
            for i in range(1, len(positions)):
                dh = positions[i][0] - positions[i-1][0]
                dw = positions[i][1] - positions[i-1][1]
                velocities.append((dh, dw))
            
            # Final velocity is average of last 3 frames
            final_velocity = np.mean(velocities[-3:], axis=0) if len(velocities) >= 3 else (
                np.mean(velocities, axis=0) if velocities else (0, 0)
            )
            
            motion_vectors[entity_id] = {
                'velocity': final_velocity,
                'last_position': positions[-1] if positions else (h_guess/2, w_guess/2)
            }
        
        return motion_vectors
    
    def compute_continuity_masks(self, prev_window, next_window, entity_maps, motion_vectors):
        """
        Compute mask weighing how much each pixel should blend based on entity consistency.
        
        Args:
            prev_window: Latents from previous window
            next_window: Latents from next window
            entity_maps: Entity saliency maps from previous window
            motion_vectors: Entity motion vectors
            
        Returns:
            Blend weight mask prioritizing entity consistency
        """
        # Start with uniform blend weights
        overlap_frames = min(16, prev_window.shape[2], next_window.shape[2])
        blend_weights = torch.ones(overlap_frames, 1, 
                                  next_window.shape[3], 
                                  next_window.shape[4], 
                                  device=next_window.device)
        
        # Don't override FETA enhancements
        if hasattr(next_window, '_feta_enhanced') and next_window._feta_enhanced:
            return blend_weights * 0.5  # Apply lighter blending when FETA is active
        
        if not entity_maps or not motion_vectors:
            return blend_weights
            
        # Adjust weights to prioritize entity consistency
        for entity_id, motion in motion_vectors.items():
            if entity_id not in entity_maps:
                continue
                
            # Extract entity map
            entity_map = entity_maps[entity_id]
            map_shape = entity_map.shape[0]
            h_guess = int(np.sqrt(map_shape / overlap_frames))
            w_guess = map_shape // (overlap_frames * h_guess)
            
            # Make sure dimensions match
            if h_guess * w_guess * overlap_frames != map_shape:
                continue
            
            try:
                map_3d = entity_map.reshape(overlap_frames, h_guess, w_guess)
            except:
                continue
            
            # Scale map_3d to match latent dimensions
            if h_guess != blend_weights.shape[2] or w_guess != blend_weights.shape[3]:
                map_3d = torch.nn.functional.interpolate(
                    map_3d.unsqueeze(0).unsqueeze(0),
                    size=(overlap_frames, blend_weights.shape[2], blend_weights.shape[3]),
                    mode='trilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
            
            # Predict where entity should be in next window
            predicted_positions = []
            last_pos = motion['last_position']
            velocity = motion['velocity']
            
            for i in range(overlap_frames):  # For overlap frames
                new_pos = (last_pos[0] + velocity[0] * (i+1), 
                           last_pos[1] + velocity[1] * (i+1))
                predicted_positions.append(new_pos)
            
            # Create mask emphasizing predicted positions
            for i, pos in enumerate(predicted_positions):
                h, w = int(pos[0]), int(pos[1])
                if 0 <= h < blend_weights.shape[2] and 0 <= w < blend_weights.shape[3]:
                    # Create a gaussian falloff around predicted position
                    h_indices = torch.arange(blend_weights.shape[2], device=blend_weights.device)
                    w_indices = torch.arange(blend_weights.shape[3], device=blend_weights.device)
                    h_dist = (h_indices.unsqueeze(1) - h) ** 2
                    w_dist = (w_indices.unsqueeze(0) - w) ** 2
                    dist = torch.sqrt(h_dist + w_dist)
                    gaussian = torch.exp(-dist / 5.0)  # Sigma = 5
                    
                    # Increase blend weights around predicted position
                    blend_weights[i, 0] += gaussian * self.continuity_strength * 2.0
        
        # Normalize
        blend_weights = blend_weights / blend_weights.max()
        return blend_weights
        
    def setup_attention_hooks(self, transformer, attention_mode="sdpa"):
        """Setup hooks to capture attention outputs"""
        self.attention_mode = attention_mode
        
        # Clear any existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Select the right hook function based on attention mode
        hook_fn = self.attention_hook_default
        if "sage" in attention_mode.lower():
            hook_fn = self.attention_hook_sage
        
        # Add hooks to the last few transformer blocks
        for block in transformer.blocks[-6:]:
            self.hooks.append(block.self_attn.register_forward_hook(hook_fn))
    
    def attention_hook_default(self, module, input, output):
        """Standard hook for capturing attention outputs"""
        if not hasattr(self, 'latest_attn_weights'):
            self.latest_attn_weights = []
        
        self.latest_attn_weights.append({
            'layer': module.__class__.__name__,
            'output': output[0].detach()  # Attention output vectors
        })
    
    def attention_hook_sage(self, module, input, output):
        """Hook for capturing SageAttention outputs"""
        if not hasattr(self, 'latest_attn_weights'):
            self.latest_attn_weights = []
        
        # SageAttention output format may be different
        if isinstance(output, tuple):
            output_tensor = output[0]
        else:
            output_tensor = output
            
        self.latest_attn_weights.append({
            'layer': module.__class__.__name__,
            'output': output_tensor.detach()
        })
    
    def cleanup_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []