import torch
import torch.nn as nn
import torch.nn.functional as F
import ray

from piper.piper import piper, distributed_stage
from piper.exec import piper_exec, Task, TaskType
from piper.utils import piper_metadata
from piper.compile import piper_setup 
from .schedule_helpers import build_zb1p_schedule, print_schedule


class MyModelVanilla(nn.Module):
    """
    Simple 2-layer MLP: 1024 -> 512 -> 512
    Used as the baseline (non-pipelined) model.
    """
    def __init__(self, in_dim=1024, hidden_dim=512, out_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class MyModelPipelined(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=512, out_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, dynamo_mb: int = 0):
        distributed_stage(0, actor_id=0, mb=dynamo_mb, optim=torch.optim.Adam)
        x = F.relu(self.fc1(x))

        distributed_stage(1, actor_id=1, mb=dynamo_mb, optim=torch.optim.Adam)
        x = F.relu(self.fc2(x))

        return x


def main():
    torch.manual_seed(0)
    ray.init(ignore_reinit_error=True)

    device = "cuda"

    init_model = MyModelVanilla().to(device)
    init_state = init_model.state_dict()

    model_base = MyModelVanilla().to(device)
    model_base.load_state_dict(init_state)
    opt_base = torch.optim.Adam(model_base.parameters(), lr=1e-3)

    model_pipe = MyModelPipelined().to(device)
    model_pipe.load_state_dict(init_state)

    batch_size = 8
    in_dim = 1024
    out_dim = 512
    num_mbs = 4

    x = torch.randn(batch_size, in_dim, device=device)
    y = torch.randn(batch_size, out_dim, device=device)
    loss_fn = nn.MSELoss(reduction="mean")

    compiled = piper_setup(
        model_pipe,
        example_inputs=[x],
        backend=piper,
    )

    actors = piper_metadata["actors"]
    num_actors = len(actors)
    assert num_actors == 2, f"Expected 2 actors, got {num_actors}"

    ray.get(actors[0].send_input.remote(x))
    ray.get(actors[num_actors - 1].send_truth.remote(y))

    schedule = build_zb1p_schedule(n_mbs=num_mbs, n_stages=2)
    print("Schedule:")
    print_schedule(schedule)

    def iter_schedule():
        out = piper_exec(compiled, schedule, [x], y, loss_fn, num_mbs)
        return ray.get(out)

    iter_schedule()

    piper_grads = []
    for stage_id in range(num_actors):
        stage_grads = ray.get(actors[stage_id].get_param_grads.remote(stage_id))
        piper_grads.extend(stage_grads)

    print(piper_grads)

    piper_params_after = []
    for stage_id in range(num_actors):
        stage_params = ray.get(actors[stage_id].get_params.remote(stage_id))
        piper_params_after.extend(stage_params)

    opt_base.zero_grad(set_to_none=True)

    for _ in range(num_mbs):
        out_base = model_base(x)
        loss_base = loss_fn(out_base, y)
        loss_base.backward()

    print(opt_base)
    opt_base.step()

    base_params_after = [
        p.detach().clone().cpu() for p in model_base.parameters()
    ]
    base_grads = [p.grad.detach().clone().cpu() for p in model_base.parameters()]

    assert len(piper_params_after) == len(base_params_after), \
        "Mismatch in # of parameters between pipeline and baseline models"

    # for i, (g_pipe, g_base) in enumerate(zip(piper_grads, base_grads)):
    #     if g_pipe is None and g_base is None:
    #         print(f"Grad {i}: both None")
    #         continue
    #     if g_pipe is None or g_base is None:
    #         print(f"Grad {i}: one is None, one not")
    #         continue
    #     diff = (g_pipe - g_base).abs().max().item()
    #     print(f"Grad {i}: max |Δ| = {diff:.6e}")


    max_abs_diff = 0.0
    for i, (p_pipe, p_base) in enumerate(zip(piper_params_after, base_params_after)):
        diff = (p_pipe - p_base).abs().max().item()
        max_abs_diff = max(max_abs_diff, diff)
        print(f"Param {i}: max |Δ| = {diff:.6e}")

    print(f"\n[RESULT] Max param diff after one step: {max_abs_diff:.6e}")


if __name__ == "__main__":
    main()
