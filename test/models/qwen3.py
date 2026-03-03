from dataclasses import dataclass
from typing import Optional
import torch
import dataclasses
from torch import nn

# from torchtitan.models.qwen3 import Qwen3Model
from torchtitan.models.qwen3 import Qwen3Model, Qwen3TransformerBlock
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.moe.moe import MoE


class AnnotatedMoE(MoE):
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, slen, dim = x.shape
        x = x.view(-1, dim)
        
        (
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert,
        ) = self.router(x, self.expert_bias)
        
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)
        
        (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        ) = self.reorderer(top_scores, selected_experts_indices)
        
        # shape (bs*slen*top_k, dim)
        routed_input = x[token_indices_experts_sorted // self.router.top_k]
        
        if self.score_before_experts:
            routed_input = (
                routed_input.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)
        
        # PIPER ANNOTATION 1: Dispatch tokens to experts via all_to_all
        with torch.fx.traceback.annotate({
            "collective": "all_to_all_single",
            "group": "ep",
            "reshape": ("output", (self.experts.num_experts, -1, dim))
        }):
            # The routed_input will be dispatched to experts on different ranks
            dispatched_input = routed_input
        
        routed_output = self.experts(dispatched_input, num_tokens_per_expert)
        
        # PIPER ANNOTATION 2: Gather expert outputs back via all_to_all
        with torch.fx.traceback.annotate({
            "collective": "all_to_all_single",
            "group": "ep",
            "reshape": ("input", (self.experts.num_experts, -1, dim))
        }):
            # The routed_output will be gathered back from expert ranks
            gathered_output = routed_output
        
        # Rest of the forward pass (same as parent)
        out = self.shared_experts(x) if self.shared_experts is not None else None
        
        # Unsort routed outputs
        routed_output_unsorted = torch.zeros(
            (bs * slen * self.router.top_k, dim),
            dtype=gathered_output.dtype,
            device=gathered_output.device,
        )
        routed_output_unsorted[token_indices_experts_sorted] = gathered_output
        routed_output_unsorted = routed_output_unsorted.reshape(
            -1, self.router.top_k, dim
        )
        
        if not self.score_before_experts:
            out_experts = (
                torch.bmm(
                    top_scores.reshape(-1, 1, self.router.top_k),
                    routed_output_unsorted.float(),
                )
                .to(x.dtype)
                .squeeze(1)
            )
        else:
            out_experts = routed_output_unsorted.sum(dim=1)
        
        if out is None:
            return out_experts.reshape(bs, slen, dim)
        return (out + out_experts).reshape(bs, slen, dim)


class AnnotatedQwen3TransformerBlock(Qwen3TransformerBlock):
    def __init__(self, config, *, layer_id: int, dim: int, n_layers: int):
        # Call parent init
        super().__init__(config, layer_id=layer_id, dim=dim, n_layers=n_layers)
        
        # Replace MoE with annotated version if MoE is enabled
        if self.moe_enabled and hasattr(self, 'moe'):
            # Create AnnotatedMoE using the config (simpler and correct)
            annotated_moe = AnnotatedMoE(config.moe, dim=dim)
            
            # Copy weights from original MoE if already initialized
            if hasattr(self.moe, 'experts'):
                annotated_moe.experts.load_state_dict(self.moe.experts.state_dict())
                annotated_moe.router.load_state_dict(self.moe.router.state_dict())
                if self.moe.shared_experts is not None:
                    annotated_moe.shared_experts.load_state_dict(
                        self.moe.shared_experts.state_dict()
                    )
            self.moe = annotated_moe


class PiperQwen3Model(Qwen3Model):
    """
    Qwen3Model with pipeline stage annotations.
    
    This wraps TorchTitan's Qwen3Model and adds stage boundaries
    in the forward pass for pipeline parallelism.
    """
    
    def __init__(self, config, num_stages: int = 2):
        # Temporarily replace layer config to use AnnotatedQwen3TransformerBlock
        original_layer_config = config.layer
        
        # Create a new config that uses AnnotatedQwen3TransformerBlock
        # We'll override the build method to use our annotated block
        class AnnotatedLayerConfig(original_layer_config.__class__):
            def build(self, *, layer_id: int, dim: int, n_layers: int):
                return AnnotatedQwen3TransformerBlock(
                    self, layer_id=layer_id, dim=dim, n_layers=n_layers
                )
        
        # Extract fields from the original config (works with slots=True)
        field_dict = {field.name: getattr(original_layer_config, field.name) 
                     for field in dataclasses.fields(original_layer_config)}
        annotated_layer_config = AnnotatedLayerConfig(**field_dict)
        config.layer = annotated_layer_config
        
        # Call parent init
        super().__init__(config)
        self.num_stages = num_stages
    
    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: Optional[AttentionMasksType] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward with pipeline stage annotations.
        
        The model is split into num_stages stages. Each stage processes
        a subset of layers.
        """
        num_layers = len(self.layers)
        layers_per_stage = num_layers // self.num_stages
        
        # Stage 0: Embedding + first N layers
        with torch.fx.traceback.annotate({"stage": 0}):
            h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
            
            # Process first set of layers
            for i in range(layers_per_stage * (self.num_stages - 1)):
                layer = self.layers[str(i)]
                h = layer(h, self.freqs_cis, attention_masks, positions)
        
        # Stage 1 (or last stage): Remaining layers + norm + output
        with torch.fx.traceback.annotate({"stage": self.num_stages - 1}):
            # Process remaining layers
            for i in range(layers_per_stage * (self.num_stages - 1), num_layers):
                layer = self.layers[str(i)]
                h = layer(h, self.freqs_cis, attention_masks, positions)
            
            # Final norm and output
            h = self.norm(h) if self.norm is not None else h
            output = self.output(h) if self.output is not None else h
        
        return output