import ray
from typing import Callable
import os

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .piper_utils import create_logger, LOG_LEVEL


# Coordinator needs GPUs when using profiling to infer stage boundaries
# @ray.remote(num_gpus=0.1)

# Use manual stage annotations- more stable
@ray.remote(num_gpus=0.1)
def run_dp_rank(dp_rank, dp_degree, pp_degree, world_size, training_func: Callable, *args, **kwargs):
    logger = create_logger("piper_coordinator", LOG_LEVEL)
    logger.debug(f"Running DP rank {dp_rank+1} of {dp_degree}")

    os.environ["PIPER_DP_RANK"] = str(dp_rank)
    os.environ["PIPER_DP_DEGREE"] = str(dp_degree)
    os.environ["PIPER_PP_DEGREE"] = str(pp_degree)
    os.environ["PIPER_WORLD_SIZE"] = str(world_size)
    os.environ["TORCH_LOGS"] = "+graph_breaks"
    return training_func(*args, **kwargs)


@ray.remote
class PiperProgramCoordinator:
    """Central Actor that Coordinates all the DP replicas of a single pipeline"""

    def __init__(self, dp_degree, pp_degree):
        self.dp_degree = dp_degree
        self.pp_degree = pp_degree
        self.world_size = dp_degree * pp_degree

    def run_program(self, training_func: Callable, pg, *args, **kwargs):
        from .piper_compile import _RANK0_ADDR_ACTOR, _COMPILED_DATA_ACTOR
        try:
            ray.kill(ray.get_actor(_RANK0_ADDR_ACTOR))
        except Exception:
            pass
        # Kill any stale compiled-data store from a previous run so that
        # dp_rank>0 workers cannot read outdated (e.g. wrong-model) stage data.
        try:
            ray.kill(ray.get_actor(_COMPILED_DATA_ACTOR))
        except Exception:
            pass

        return ray.get(
            [
                run_dp_rank.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=dp_rank,
                    )
                ).remote(
                    dp_rank,
                    self.dp_degree,
                    self.pp_degree,
                    self.world_size,
                    training_func,
                    *args,
                    **kwargs,
                )
                for dp_rank in range(self.dp_degree)
            ]
        )
