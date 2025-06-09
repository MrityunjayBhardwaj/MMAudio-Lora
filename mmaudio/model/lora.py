"""
LoRA (Low-Rank Adaptation) implementation for MMAudio.
Designed for parameter-efficient fine-tuning on small datasets.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

log = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """
    LoRA adaptation for Linear layers.
    Implements: output = original_linear(x) + lora_B(lora_A(x)) * (alpha / rank)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        alpha: float = 128.0,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Optional dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Scaling factor
        self.scaling = alpha / rank
        
        # Track if LoRA is enabled
        self.enabled = True
        
    def forward(self, x: torch.Tensor, original_weight: torch.Tensor, original_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass combining original linear layer with LoRA adaptation"""
        
        # Original linear transformation
        result = F.linear(x, original_weight, original_bias)
        
        if not self.enabled or self.rank == 0:
            return result
            
        # LoRA adaptation: x -> lora_A -> dropout -> lora_B -> scale
        lora_x = F.linear(x, self.lora_A)
        lora_x = self.dropout(lora_x)
        lora_output = F.linear(lora_x, self.lora_B)
        
        return result + lora_output * self.scaling
        
    def enable(self):
        """Enable LoRA adaptation"""
        self.enabled = True
        
    def disable(self):
        """Disable LoRA adaptation (only use original weights)"""
        self.enabled = False
        
    def merge_weights(self, original_weight: torch.Tensor) -> torch.Tensor:
        """Merge LoRA weights into original weight matrix for inference"""
        if self.rank == 0:
            return original_weight
            
        # Compute LoRA weight: B @ A * scaling
        lora_weight = self.lora_B @ self.lora_A * self.scaling
        return original_weight + lora_weight
        
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, alpha={self.alpha}"


class LoRAWrapper(nn.Module):
    """
    Wrapper that adds LoRA to an existing Linear layer without modifying the original
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 64,
        alpha: float = 128.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.original_layer = original_layer
        
        # Freeze original parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        # Create LoRA adaptation
        self.lora = LoRALinear(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=original_layer.bias is not None
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora(x, self.original_layer.weight, self.original_layer.bias)
        
    def merge_weights(self) -> nn.Linear:
        """Create a new Linear layer with merged weights"""
        merged_weight = self.lora.merge_weights(self.original_layer.weight)
        
        new_layer = nn.Linear(
            self.original_layer.in_features,
            self.original_layer.out_features,
            bias=self.original_layer.bias is not None
        )
        
        new_layer.weight.data = merged_weight
        if self.original_layer.bias is not None:
            new_layer.bias.data = self.original_layer.bias.data.clone()
            
        return new_layer


class LoRAConfig:
    """Configuration for LoRA application"""
    
    def __init__(
        self,
        rank: int = 64,
        alpha: float = 128.0,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        exclude_modules: Optional[List[str]] = None,
        bias: str = "none"  # "none", "all", "lora_only"
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or []
        self.exclude_modules = exclude_modules or []
        self.bias = bias
        
        # Default target modules for MMAudio
        if not self.target_modules:
            self.target_modules = [
                # Joint blocks - core cross-modal attention
                "joint_blocks.*.latent_block.attn.qkv",
                "joint_blocks.*.latent_block.ffn.linear1",
                "joint_blocks.*.latent_block.ffn.linear2", 
                "joint_blocks.*.clip_block.attn.qkv",
                "joint_blocks.*.clip_block.ffn.linear1",
                "joint_blocks.*.clip_block.ffn.linear2",
                "joint_blocks.*.text_block.attn.qkv", 
                "joint_blocks.*.text_block.ffn.linear1",
                "joint_blocks.*.text_block.ffn.linear2",
                
                # Fused blocks - final processing
                "fused_blocks.*.attn.qkv",
                "fused_blocks.*.ffn.linear1", 
                "fused_blocks.*.ffn.linear2",
            ]


def _match_module_name(name: str, patterns: List[str]) -> bool:
    """Check if module name matches any pattern (supports * wildcards)"""
    import fnmatch
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
    verbose: bool = True
) -> Tuple[nn.Module, Dict[str, LoRAWrapper]]:
    """
    Apply LoRA to specified modules in a model.
    
    Returns:
        - Modified model with LoRA layers
        - Dictionary mapping module names to LoRA wrappers
    """
    
    lora_modules = {}
    
    # Collect all target modules
    target_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module should get LoRA
            if _match_module_name(name, config.target_modules):
                if not _match_module_name(name, config.exclude_modules):
                    target_layers.append((name, module))
                    
    if verbose:
        log.info(f"Applying LoRA to {len(target_layers)} modules:")
        for name, _ in target_layers:
            log.info(f"  - {name}")
            
    # Apply LoRA to each target layer
    for name, original_layer in target_layers:
        # Create LoRA wrapper
        lora_wrapper = LoRAWrapper(
            original_layer=original_layer,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout
        )
        
        # Replace the module in the model
        _replace_module(model, name, lora_wrapper)
        lora_modules[name] = lora_wrapper
        
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if verbose:
        log.info(f"LoRA applied successfully:")
        log.info(f"  Total parameters: {total_params:,}")
        log.info(f"  LoRA parameters: {lora_params:,}")
        log.info(f"  Trainable ratio: {100 * lora_params / total_params:.2f}%")
        
    return model, lora_modules


def _replace_module(model: nn.Module, target_name: str, new_module: nn.Module):
    """Replace a module in the model by its dotted name"""
    
    parts = target_name.split('.')
    current = model
    
    # Navigate to parent module
    for part in parts[:-1]:
        current = getattr(current, part)
        
    # Replace the target module
    setattr(current, parts[-1], new_module)


def save_lora_weights(
    lora_modules: Dict[str, LoRAWrapper],
    save_path: Union[str, Path],
    metadata: Optional[Dict] = None
):
    """Save only LoRA weights and configuration"""
    
    save_dict = {
        'lora_weights': {},
        'lora_config': {},
        'metadata': metadata or {}
    }
    
    for name, lora_module in lora_modules.items():
        save_dict['lora_weights'][name] = {
            'lora_A': lora_module.lora.lora_A.data,
            'lora_B': lora_module.lora.lora_B.data,
        }
        save_dict['lora_config'][name] = {
            'rank': lora_module.lora.rank,
            'alpha': lora_module.lora.alpha,
            'in_features': lora_module.lora.in_features,
            'out_features': lora_module.lora.out_features,
        }
        
    torch.save(save_dict, save_path)
    log.info(f"LoRA weights saved to {save_path}")


def load_lora_weights(
    model: nn.Module,
    lora_path: Union[str, Path],
    strict: bool = True
) -> Tuple[nn.Module, Dict[str, LoRAWrapper]]:
    """Load LoRA weights into a model"""
    
    checkpoint = torch.load(lora_path, map_location='cpu')
    lora_weights = checkpoint['lora_weights']
    lora_config = checkpoint['lora_config']
    
    # Create LoRA config from saved data
    config = LoRAConfig(
        target_modules=list(lora_weights.keys()),
        rank=list(lora_config.values())[0]['rank'],  # Assume same rank for all
        alpha=list(lora_config.values())[0]['alpha']
    )
    
    # Apply LoRA to model
    model, lora_modules = apply_lora_to_model(model, config, verbose=False)
    
    # Load weights
    for name, lora_module in lora_modules.items():
        if name in lora_weights:
            lora_module.lora.lora_A.data = lora_weights[name]['lora_A']
            lora_module.lora.lora_B.data = lora_weights[name]['lora_B']
        elif strict:
            raise ValueError(f"LoRA weights not found for module: {name}")
            
    log.info(f"LoRA weights loaded from {lora_path}")
    return model, lora_modules


def merge_lora_weights(
    model: nn.Module,
    lora_modules: Dict[str, LoRAWrapper]
) -> nn.Module:
    """
    Merge LoRA weights into the base model for inference.
    Returns a new model with merged weights.
    """
    
    # Create a copy of the model
    import copy
    merged_model = copy.deepcopy(model)
    
    # Merge each LoRA module
    for name, lora_module in lora_modules.items():
        # Get the corresponding module in the merged model
        parts = name.split('.')
        current = merged_model
        
        for part in parts[:-1]:
            current = getattr(current, part)
            
        # Replace with merged layer
        merged_layer = lora_module.merge_weights()
        setattr(current, parts[-1], merged_layer)
        
    log.info("LoRA weights merged into base model")
    return merged_model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get only the LoRA parameters from a model"""
    
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRAWrapper):
            lora_params.extend(list(module.lora.parameters()))
            
    return lora_params


def print_lora_info(model: nn.Module):
    """Print detailed information about LoRA modules in the model"""
    
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(p.numel() for p in get_lora_parameters(model))
    original_params = total_params - lora_params
    
    print(f"\n{'='*60}")
    print(f"LoRA Model Information")
    print(f"{'='*60}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Original parameters:  {original_params:,}")
    print(f"LoRA parameters:      {lora_params:,}")
    print(f"Trainable ratio:      {100 * lora_params / total_params:.2f}%")
    print(f"Memory reduction:     ~{original_params // lora_params:.0f}x")
    
    # List LoRA modules
    lora_modules = []
    for name, module in model.named_modules():
        if isinstance(module, LoRAWrapper):
            lora_modules.append((name, module.lora.rank))
            
    print(f"\nLoRA Modules ({len(lora_modules)}):")
    for name, rank in lora_modules:
        print(f"  {name:50} (rank={rank})")
        
    print(f"{'='*60}\n")