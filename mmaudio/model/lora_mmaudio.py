"""
MMAudio-specific LoRA wrapper and utilities.
Handles the integration of LoRA with MMAudio's transformer architecture.
"""

import logging
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
from pathlib import Path

from mmaudio.model.networks import MMAudio
from mmaudio.model.lora import LoRAConfig, apply_lora_to_model, LoRAWrapper, save_lora_weights, load_lora_weights
from mmaudio.model.flow_matching import FlowMatching

log = logging.getLogger(__name__)


class MMAudioLoRA(nn.Module):
    """
    LoRA wrapper specifically designed for MMAudio.
    Provides easy interface for training, inference, and weight management.
    """
    
    def __init__(
        self,
        base_model: MMAudio,
        lora_config: Optional[LoRAConfig] = None,
        enable_gradient_checkpointing: bool = True
    ):
        super().__init__()
        
        self.base_model = base_model
        self.lora_config = lora_config or self._get_default_config()
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        
        # Apply LoRA to the model
        self.model, self.lora_modules = apply_lora_to_model(
            model=self.base_model,
            config=self.lora_config,
            verbose=True
        )
        
        # Setup gradient checkpointing for memory efficiency
        if enable_gradient_checkpointing:
            self._setup_gradient_checkpointing()
            
        # Track training state
        self.training_step = 0
        
    def _get_default_config(self) -> LoRAConfig:
        """Get default LoRA configuration optimized for MMAudio"""
        return LoRAConfig(
            rank=64,
            alpha=128.0,
            dropout=0.1,
            target_modules=[
                # Joint blocks - cross-modal attention (PRIORITY 1)
                "joint_blocks.*.latent_block.attn.qkv",
                "joint_blocks.*.latent_block.ffn.linear1",
                "joint_blocks.*.latent_block.ffn.linear2",
                "joint_blocks.*.clip_block.attn.qkv", 
                "joint_blocks.*.clip_block.ffn.linear1",
                "joint_blocks.*.clip_block.ffn.linear2",
                "joint_blocks.*.text_block.attn.qkv",
                "joint_blocks.*.text_block.ffn.linear1", 
                "joint_blocks.*.text_block.ffn.linear2",
                
                # Fused blocks - final processing (PRIORITY 2)  
                "fused_blocks.*.attn.qkv",
                "fused_blocks.*.ffn.linear1",
                "fused_blocks.*.ffn.linear2",
            ]
        )
        
    def _setup_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        
        # Checkpoint joint blocks
        for block in self.model.joint_blocks:
            if hasattr(block, 'forward'):
                original_forward = block.forward
                
                def make_checkpointed_forward(original_fn):
                    def checkpointed_forward(*args, **kwargs):
                        if self.training:
                            return torch.utils.checkpoint.checkpoint(original_fn, *args, **kwargs)
                        else:
                            return original_fn(*args, **kwargs)
                    return checkpointed_forward
                    
                block.forward = make_checkpointed_forward(original_forward)
                
        # Checkpoint fused blocks  
        for block in self.model.fused_blocks:
            if hasattr(block, 'forward'):
                original_forward = block.forward
                
                def make_checkpointed_forward(original_fn):
                    def checkpointed_forward(*args, **kwargs):
                        if self.training:
                            return torch.utils.checkpoint.checkpoint(original_fn, *args, **kwargs)
                        else:
                            return original_fn(*args, **kwargs)
                    return checkpointed_forward
                    
                block.forward = make_checkpointed_forward(original_forward)
                
        log.info("Gradient checkpointing enabled for transformer blocks")
        
    def forward(self, *args, **kwargs):
        """Forward pass through the LoRA-adapted model"""
        return self.model(*args, **kwargs)
        
    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Get only the LoRA parameters for optimizer"""
        lora_params = []
        
        for module in self.model.modules():
            if isinstance(module, LoRAWrapper):
                lora_params.extend(list(module.lora.parameters()))
                
        return lora_params
        
    def get_parameter_stats(self) -> Dict[str, int]:
        """Get detailed parameter statistics"""
        
        total_params = sum(p.numel() for p in self.model.parameters())
        lora_params = sum(p.numel() for p in self.get_lora_parameters())
        frozen_params = total_params - lora_params
        
        return {
            'total_parameters': total_params,
            'lora_parameters': lora_params, 
            'frozen_parameters': frozen_params,
            'trainable_ratio': lora_params / total_params,
            'memory_reduction_factor': frozen_params // lora_params if lora_params > 0 else 0
        }
        
    def print_parameter_info(self):
        """Print detailed parameter information"""
        
        stats = self.get_parameter_stats()
        
        print(f"\n{'='*70}")
        print(f"MMAudio LoRA Parameter Information")
        print(f"{'='*70}")
        print(f"Base model:           {type(self.base_model).__name__}")
        print(f"LoRA rank:            {self.lora_config.rank}")
        print(f"LoRA alpha:           {self.lora_config.alpha}")
        print(f"LoRA dropout:         {self.lora_config.dropout}")
        print(f"")
        print(f"Total parameters:     {stats['total_parameters']:,}")
        print(f"LoRA parameters:      {stats['lora_parameters']:,}")
        print(f"Frozen parameters:    {stats['frozen_parameters']:,}")
        print(f"Trainable ratio:      {stats['trainable_ratio']*100:.2f}%")
        print(f"Memory reduction:     ~{stats['memory_reduction_factor']:.0f}x")
        print(f"")
        print(f"Target modules ({len(self.lora_modules)}):")
        
        for name in sorted(self.lora_modules.keys()):
            rank = self.lora_modules[name].lora.rank
            params = self.lora_modules[name].lora.lora_A.numel() + self.lora_modules[name].lora.lora_B.numel()
            print(f"  {name:50} (rank={rank:2d}, params={params:,})")
            
        print(f"{'='*70}\n")
        
    def save_lora_checkpoint(
        self,
        save_path: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        **metadata
    ):
        """Save LoRA checkpoint with training state"""
        
        checkpoint = {
            'lora_weights': {},
            'lora_config': {
                'rank': self.lora_config.rank,
                'alpha': self.lora_config.alpha,
                'dropout': self.lora_config.dropout,
                'target_modules': self.lora_config.target_modules
            },
            'training_step': self.training_step,
            'model_config': {
                'model_class': type(self.base_model).__name__,
                'model_params': self.get_parameter_stats()
            },
            'metadata': metadata
        }
        
        # Save LoRA weights
        for name, lora_module in self.lora_modules.items():
            checkpoint['lora_weights'][name] = {
                'lora_A': lora_module.lora.lora_A.data.clone(),
                'lora_B': lora_module.lora.lora_B.data.clone(),
            }
            
        # Save optimizer state
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        # Save scheduler state  
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
        torch.save(checkpoint, save_path)
        log.info(f"LoRA checkpoint saved to {save_path}")
        
    @classmethod
    def load_from_checkpoint(
        cls,
        base_model: MMAudio,
        checkpoint_path: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        strict: bool = True
    ) -> Tuple['MMAudioLoRA', Dict]:
        """Load LoRA model from checkpoint"""
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Reconstruct LoRA config
        lora_config = LoRAConfig(
            rank=checkpoint['lora_config']['rank'],
            alpha=checkpoint['lora_config']['alpha'], 
            dropout=checkpoint['lora_config']['dropout'],
            target_modules=checkpoint['lora_config']['target_modules']
        )
        
        # Create LoRA model
        lora_model = cls(
            base_model=base_model,
            lora_config=lora_config,
            enable_gradient_checkpointing=True
        )
        
        # Load LoRA weights
        for name, lora_module in lora_model.lora_modules.items():
            if name in checkpoint['lora_weights']:
                lora_module.lora.lora_A.data = checkpoint['lora_weights'][name]['lora_A']
                lora_module.lora.lora_B.data = checkpoint['lora_weights'][name]['lora_B']
            elif strict:
                raise ValueError(f"LoRA weights not found for module: {name}")
                
        # Restore training step
        lora_model.training_step = checkpoint.get('training_step', 0)
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        log.info(f"LoRA model loaded from {checkpoint_path}")
        log.info(f"Training step: {lora_model.training_step}")
        
        return lora_model, checkpoint.get('metadata', {})
        
    def merge_and_save(self, save_path: Path):
        """Merge LoRA weights into base model and save for inference"""
        
        import copy
        
        # Create a copy of the base model
        merged_model = copy.deepcopy(self.base_model)
        
        # Apply LoRA weights to the copy
        temp_model, temp_lora_modules = apply_lora_to_model(
            model=merged_model,
            config=self.lora_config,
            verbose=False
        )
        
        # Load our trained LoRA weights
        for name, lora_module in temp_lora_modules.items():
            if name in self.lora_modules:
                lora_module.lora.lora_A.data = self.lora_modules[name].lora.lora_A.data.clone()
                lora_module.lora.lora_B.data = self.lora_modules[name].lora.lora_B.data.clone()
                
        # Merge weights and replace modules
        for name, lora_module in temp_lora_modules.items():
            merged_layer = lora_module.merge_weights()
            
            # Replace in the model
            parts = name.split('.')
            current = merged_model
            for part in parts[:-1]:
                current = getattr(current, part)
            setattr(current, parts[-1], merged_layer)
            
        # Save merged model
        torch.save({
            'model_state_dict': merged_model.state_dict(),
            'model_config': type(self.base_model).__name__,
            'lora_config': {
                'rank': self.lora_config.rank,
                'alpha': self.lora_config.alpha,
                'merged': True
            },
            'training_step': self.training_step
        }, save_path)
        
        log.info(f"Merged model saved to {save_path}")
        return merged_model
        
    def enable_lora(self):
        """Enable LoRA adaptations"""
        for lora_module in self.lora_modules.values():
            lora_module.lora.enable()
            
    def disable_lora(self):
        """Disable LoRA adaptations (use only base model)"""
        for lora_module in self.lora_modules.values():
            lora_module.lora.disable()


def create_lora_mmaudio(
    base_model: MMAudio,
    rank: int = 64,
    alpha: float = 128.0,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None
) -> MMAudioLoRA:
    """Convenience function to create LoRA-adapted MMAudio model"""
    
    config = LoRAConfig(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules
    )
    
    return MMAudioLoRA(
        base_model=base_model,
        lora_config=config,
        enable_gradient_checkpointing=True
    )


def load_pretrained_with_lora(
    model_name: str,
    lora_checkpoint: Optional[Path] = None,
    device: str = 'cuda'
) -> MMAudioLoRA:
    """Load pretrained MMAudio model with LoRA adaptation"""
    
    # Load base model
    from mmaudio.model.networks import get_my_mmaudio
    from mmaudio.eval_utils import all_model_cfg
    
    if model_name not in all_model_cfg:
        raise ValueError(f"Unknown model: {model_name}")
        
    model_config = all_model_cfg[model_name]
    model_config.download_if_needed()
    
    # Load base model with pretrained weights
    base_model = get_my_mmaudio(model_name).to(device)
    checkpoint = torch.load(model_config.model_path, map_location=device)
    base_model.load_state_dict(checkpoint, strict=True)
    
    log.info(f"Loaded pretrained model: {model_name}")
    
    # Create LoRA model
    if lora_checkpoint and lora_checkpoint.exists():
        lora_model, metadata = MMAudioLoRA.load_from_checkpoint(
            base_model=base_model,
            checkpoint_path=lora_checkpoint
        )
        log.info(f"LoRA weights loaded from {lora_checkpoint}")
    else:
        lora_model = create_lora_mmaudio(base_model)
        log.info("Created new LoRA model with default configuration")
        
    return lora_model