from dataclasses import dataclass
from typing import Optional
import torch
import dataclasses
from torch import nn

from torchtitan.models.qwen3 import Qwen3Model, Qwen3TransformerBlock
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.moe.moe import MoE

from torchtitan.config import ActivationCheckpointConfig
from torchtitan.distributed.activation_checkpoint import apply_ac

from torch.utils.checkpoint import checkpoint

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
        
        routed_input = routed_input.reshape(
            2, self.experts.num_experts, -1 , dim)

        # dispatch tokens to experts via all_to_all
        with torch.fx.traceback.annotate({
            "collective": "all_to_all_single",
            "group": "ep",
        }):
            dispatched_input = routed_input.contiguous()
        
        dispatched_input = dispatched_input.reshape(-1, dim)
        routed_output = self.experts(dispatched_input, num_tokens_per_expert)
        routed_output = routed_output.reshape(self.experts.num_experts, -1, dim)

        # gather expert outputs back via all_to_all
        with torch.fx.traceback.annotate({
            "collective": "all_to_all_single",
            "group": "ep",
        }):
            gathered_output = routed_output.contiguous()
        
        gathered_output = gathered_output.reshape(-1, dim)
        out = self.shared_experts(x) if self.shared_experts is not None else None
        
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
        super().__init__(config, layer_id=layer_id, dim=dim, n_layers=n_layers)
        
        if self.moe_enabled:
            self.moe = AnnotatedMoE(config.moe, dim=dim)


class PiperQwen3Model(Qwen3Model):
    """
    Qwen3Model with pipeline stage annotations.
    
    This wraps TorchTitan's Qwen3Model and adds stage boundaries
    in the forward pass for pipeline parallelism.
    """
    
    def __init__(self, config, num_stages: int = 2, use_checkpointing: bool = False):
        original_layer_config = config.layer
        
        class AnnotatedLayerConfig(original_layer_config.__class__):
            def build(self, *, layer_id: int, dim: int, n_layers: int):
                return AnnotatedQwen3TransformerBlock(
                    self, layer_id=layer_id, dim=dim, n_layers=n_layers
                )
        
        field_dict = {field.name: getattr(original_layer_config, field.name) 
                     for field in dataclasses.fields(original_layer_config)}
        annotated_layer_config = AnnotatedLayerConfig(**field_dict)
        config.layer = annotated_layer_config
        
        super().__init__(config)
        self.num_stages = num_stages
        self.use_checkpointing = use_checkpointing

    
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
        
        # stage 0: embedding + first N layers
        with torch.fx.traceback.annotate({"stage": 0}):
            h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
            
            for i in range(layers_per_stage * (self.num_stages - 1)):
                layer = self.layers[str(i)]
                h = layer(h, self.freqs_cis, attention_masks, positions)
        
        # stage 1: remaining layers + norm + output
        with torch.fx.traceback.annotate({"stage": self.num_stages - 1}):
            for i in range(layers_per_stage * (self.num_stages - 1), num_layers):
                layer = self.layers[str(i)]
                h = layer(h, self.freqs_cis, attention_masks, positions)
            
            # final norm and output
            h = self.norm(h) if self.norm is not None else h
            output = self.output(h) if self.output is not None else h
        
        return output