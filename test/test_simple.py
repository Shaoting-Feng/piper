import torch
import ray
from torch import nn
from src.piper import piper, distributed_stage
from collections.abc import Iterable

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        distributed_stage(0, optim=torch.optim.Adam)
        x = self.relu(self.fc1(x))
        distributed_stage(1, optim=torch.optim.Adam)
        x = self.relu(self.fc2(x))
        distributed_stage(2, optim=torch.optim.Adam)
        x = self.fc3(x)
        return x

# Compile the model

model = torch.compile(MyModel().to('cuda'), backend=piper)
x = torch.randn(10, 1024).to('cuda')
y = torch.randn(10, 10).to('cuda')
out = model(x).get()

# Manually run a training step on a single batch

from src.piper_utils import piper_metadata
actors = piper_metadata['actors']

# forward pass (executing with fused forwards, for now)
model(x).get()

# backwards pass, compiling the backward graph for each stage
grad_ref = actors[2].backward_compile.remote(stage_id=2, mb_idx=None, inp=[y], loss_fn=torch.nn.functional.mse_loss)
grad_ref = actors[1].backward_compile.remote(stage_id=1, mb_idx=None, inp=grad_ref)
grad_ref = actors[0].backward_compile.remote(stage_id=0, mb_idx=None, inp=grad_ref)

ray.wait([grad_ref])