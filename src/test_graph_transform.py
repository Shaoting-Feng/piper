"""
Correctness test for _split_gm_by_stages and _profile_and_split_gm.

Verifies that when the stage GraphModules produced by each split function are
composed (run sequentially, feeding each stage's outputs into the next), the
final output matches the original model's output on the same input.

Both split functions are designed to operate on FX graphs captured by
torch.compile (AOT autograd), where model parameters are lifted to placeholder
inputs rather than represented as get_attr nodes.  This test captures graphs
in that same way via a custom torch.compile backend.

Usage:
    python -m src.test_graph_transform
"""
import argparse
import copy
import sys
import torch
import torch.nn as nn
import torch.fx as fx

from .piper_graph_transform import _split_gm_by_stages, _profile_and_split_gm


# ---------------------------------------------------------------------------
# Simple test model
# ---------------------------------------------------------------------------

class _Net(nn.Module):
    """Three-layer MLP that is easily splittable into two stages."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)
        self.fc3 = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ---------------------------------------------------------------------------
# FX graph capture via torch.compile
# ---------------------------------------------------------------------------

def _capture_gm(model: nn.Module, x: torch.Tensor):
    """
    Capture the FX GraphModule that torch.compile produces for *model*(*x*).

    Returns (gm, example_inputs, ref_output) where:
      - gm            : the FX GraphModule with proper example_value metadata
      - example_inputs: real tensors corresponding to gm's placeholder nodes
                        (lifted parameters first, then the data input x)
      - ref_output    : model(x) evaluated once with real weights
    """
    captured: dict = {}

    def _backend(gm, example_inputs):
        captured["gm"] = gm
        captured["inputs"] = list(example_inputs)
        return gm.forward

    with torch.no_grad():
        ref_output = model(x).detach().clone()

    compiled = torch.compile(model, backend=_backend, fullgraph=True)
    with torch.no_grad():
        compiled(x)

    return captured["gm"], captured["inputs"], ref_output


# ---------------------------------------------------------------------------
# Stage annotation
# ---------------------------------------------------------------------------

def _annotate_stages(gm: fx.GraphModule, num_stages: int) -> None:
    """
    Divide compute nodes (everything except placeholder / get_attr / output)
    into *num_stages* equal-ish chunks and stamp ``node.meta['custom']['stage']``.
    """
    compute = [n for n in gm.graph.nodes if n.op not in ("placeholder", "get_attr", "output")]
    chunk = max(1, (len(compute) + num_stages - 1) // num_stages)
    for i, node in enumerate(compute):
        stage = min(i // chunk, num_stages - 1)
        node.meta.setdefault("custom", {})["stage"] = stage


# ---------------------------------------------------------------------------
# Running the split stages
# ---------------------------------------------------------------------------

def _build_ph_map(gm: fx.GraphModule, example_inputs: list) -> dict:
    """
    Return a dict mapping each placeholder node's *name* to its corresponding
    real tensor from *example_inputs*.  Used to fill parameter slots in stage
    GraphModules by name matching.
    """
    ph_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
    return {node.name: val for node, val in zip(ph_nodes, example_inputs)}


def _run_stage_chain(
    submodules,
    orig_ph_map: dict,
) -> torch.Tensor:
    """
    Execute stage GraphModules sequentially, feeding each stage's outputs into
    the next stage's inter-stage placeholder inputs.

    For each stage_gm:
      - Placeholders whose names appear in *orig_ph_map* (parameters / original
        data inputs) are filled from that map.
      - Remaining placeholders (inter-stage activations from the previous stage)
        are filled from the previous stage's output in order.

    Returns the final stage's output tensor.
    """
    prev_output = None

    for stage_id, stage_gm, _input_idxs, _param_idxs, _graphargs, _placeholders in sorted(
        submodules, key=lambda t: t[0]
    ):
        ph_nodes = list(stage_gm.graph.find_nodes(op="placeholder"))
        args: list = [None] * len(ph_nodes)

        # 1. Fill placeholders by name from the original graph's inputs.
        for i, node in enumerate(ph_nodes):
            if node.name in orig_ph_map:
                args[i] = orig_ph_map[node.name]

        # 2. Fill remaining (inter-stage) placeholders from the previous stage's
        #    output, in left-to-right placeholder order.
        if prev_output is not None:
            inter = (
                (prev_output,)
                if isinstance(prev_output, torch.Tensor)
                else tuple(prev_output)
            )
            unfilled = [i for i, v in enumerate(args) if v is None]
            for pos, idx in enumerate(unfilled):
                if pos < len(inter):
                    args[idx] = inter[pos]

        with torch.no_grad():
            prev_output = stage_gm(*args)

    # Unwrap single-element tuples that some output nodes produce.
    if isinstance(prev_output, (list, tuple)) and len(prev_output) == 1:
        prev_output = prev_output[0]
    return prev_output


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_split_gm_by_stages(verbose: bool = False) -> None:
    """
    _split_gm_by_stages: stage GMs composed sequentially reproduce the
    original model's output.
    """
    torch.manual_seed(0)
    dim = 32
    model = _Net(dim)
    x = torch.randn(4, dim)

    gm, example_inputs, ref = _capture_gm(model, x)
    orig_ph_map = _build_ph_map(gm, example_inputs)

    # Annotate the captured graph so _split_gm_by_stages knows where to split.
    _annotate_stages(gm, num_stages=2)

    _, submodules = _split_gm_by_stages(gm)

    assert len(submodules) == 2, (
        f"_split_gm_by_stages: expected 2 stages, got {len(submodules)}"
    )

    result = _run_stage_chain(submodules, orig_ph_map)

    max_diff = (ref - result.detach()).abs().max().item()
    assert torch.allclose(ref, result.detach(), atol=1e-5), (
        f"_split_gm_by_stages output mismatch: max abs diff = {max_diff:.2e}"
    )

    if verbose:
        print(f"  max_diff={max_diff:.2e}")


def test_profile_and_split_gm(verbose: bool = False) -> None:
    """
    _profile_and_split_gm: auto-profiled stages composed sequentially
    reproduce the original model's output.

    _profile_and_split_gm may move call_module submodules to meta device
    during profiling.  Because torch.compile's AOT-traced graphs lift
    parameters to placeholder inputs (no call_module / get_attr nodes for
    weights), this is not an issue for simple models.  A deepcopy of the
    model is used for tracing to avoid any side-effects on the original.
    """
    torch.manual_seed(0)
    dim = 32
    model = _Net(dim)
    x = torch.randn(4, dim)

    gm, example_inputs, ref = _capture_gm(model, x)
    orig_ph_map = _build_ph_map(gm, example_inputs)

    # Use a fresh capture on a deep copy so profiling doesn't mutate `gm`.
    gm_for_profile, _, _ = _capture_gm(copy.deepcopy(model), x)

    # No stage annotations; _profile_and_split_gm determines the split.
    _, submodules = _profile_and_split_gm(gm_for_profile, num_stages=2)

    assert len(submodules) == 2, (
        f"_profile_and_split_gm: expected 2 stages, got {len(submodules)}"
    )

    # Run stage chain with the *original* example_inputs (real weights).
    result = _run_stage_chain(submodules, orig_ph_map)

    max_diff = (ref - result.detach()).abs().max().item()
    assert torch.allclose(ref, result.detach(), atol=1e-5), (
        f"_profile_and_split_gm output mismatch: max abs diff = {max_diff:.2e}"
    )

    if verbose:
        print(f"  max_diff={max_diff:.2e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Graph-transform correctness tests")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print("Running test_split_gm_by_stages ...", flush=True)
    test_split_gm_by_stages(verbose=args.verbose)
    print("  PASSED")

    print("Running test_profile_and_split_gm ...", flush=True)
    test_profile_and_split_gm(verbose=args.verbose)
    print("  PASSED")

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
