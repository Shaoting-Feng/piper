import ray
import torch
import os
import gc
from torch._dynamo.backends.registry import register_backend

from .piper_utils import _serialize_graphmodule, piper_metadata, create_logger, LOG_LEVEL
from .piper_graph_transform import _get_dp_comm_ops, _split_gm_by_stages, _insert_a2a_ops, _insert_p2p_ops
from .piper_actor import _get_actor

logger = create_logger("piper_backend", LOG_LEVEL)


@register_backend
def piper(gm, example_inputs, **kwargs):

    original_gm = gm
    top_level_gm, submodules = _split_gm_by_stages(gm)
    
    num_stages = len(piper_metadata.stage_to_device.keys())
    dp_rank = int(os.environ['PIPER_DP_RANK'])
    pp_degree = int(os.environ['PIPER_PP_DEGREE'])
    dp_degree = int(os.environ['PIPER_DP_DEGREE'])

    del top_level_gm

    refs = []
    actor_stages = []
    for (stage_id, stage_gm, input_idxs, param_idxs, graphargs, placeholders) in submodules:
        stage_gm(*graphargs)
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(5):
            stage_gm(*graphargs)
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        torch.cuda.synchronize()
        print(f"Stage {stage_id} time measured on controller: {start.elapsed_time(end) / 5:.2f} ms")

        n_a2a_ops = 0
        if dp_degree > 1:
            stage_gm, n_a2a_ops = _insert_a2a_ops(stage_gm)

        actor_id = piper_metadata.stage_to_device[stage_id]
        actor = _get_actor(actor_id)
        actor_stages.append((actor, stage_id))
        
        stage_gm_data = _serialize_graphmodule(stage_gm)
        refs.append(actor._load_stage.remote(
            stage_id, 
            stage_gm_data,
            graphargs,
            input_idxs,
            param_idxs,
            n_a2a_ops,
        ))

        del stage_gm
        del graphargs
        del placeholders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ray.get(refs)

    # TODO: build dag by analyzing stage dependencies
    # this only supports sequential pipelines
    for stage_id in piper_metadata.stage_to_device.keys():
        if stage_id < len(piper_metadata.stage_to_device.keys()) - 1:
            piper_metadata.dag.add((stage_id, stage_id+1))
    
    example_outputs = original_gm(*example_inputs)

    def callback(*args):
        logger.warning("Should not directly call compiled module, running non-distributed execution")
        return example_outputs

    del original_gm
    del example_inputs

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return callback