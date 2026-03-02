import ray
import torch
from typing import Callable
import os
import socket

from .piper_utils import create_logger, LOG_LEVEL


@ray.remote(num_gpus=0.1)
def run_dp_rank(dp_rank, dp_degree, pp_degree, world_size, master_addr, master_port, training_func: Callable, *args, **kwargs):
    logger = create_logger("piper_coordinator", LOG_LEVEL)
    logger.debug(f"Running DP rank {dp_rank+1} of {dp_degree}")

    os.environ['PIPER_DP_RANK'] = str(dp_rank)
    os.environ['PIPER_DP_DEGREE'] = str(dp_degree)
    os.environ['PIPER_PP_DEGREE'] = str(pp_degree)
    os.environ['PIPER_WORLD_SIZE'] = str(world_size)
    os.environ['PIPER_MASTER_ADDR'] = str(master_addr)
    os.environ['PIPER_MASTER_PORT'] = str(master_port)
    os.environ['TORCH_LOGS'] = "+graph_breaks"
    return training_func(*args, **kwargs)

def find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        port = s.getsockname()[1]
    return port

@ray.remote
class PiperProgramCoordinator:
    """ Central Actor that Coordinates all the DP replicas of a single pipeline"""
    def __init__(self, dp_degree, pp_degree):
        self.dp_degree = dp_degree
        self.pp_degree = pp_degree
        self.world_size = dp_degree * pp_degree
        self.master_port = find_free_port()
    
    def run_program(self, training_func: Callable, *args, **kwargs):
        return ray.get([run_dp_rank.remote(
                dp_rank, 
                self.dp_degree, 
                self.pp_degree,
                self.world_size, 
                "127.0.0.1", 
                self.master_port, 
                training_func,
                *args, 
                **kwargs) 
            for dp_rank in range(self.dp_degree)])
    

            


