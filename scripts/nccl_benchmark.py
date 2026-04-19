"""
NCCL bandwidth benchmark for Qwen3-30B-A3B communication patterns.

Qwen3-30B-A3B config:
  hidden_size:           2048
  moe_intermediate_size: 768    (per-expert FFN width)
  num_experts:           128
  num_experts_per_tok:   8      (top-k)
  num_hidden_layers:     48
  num_attention_heads:   32
  num_key_value_heads:   4
  head_dim:              128
  vocab_size:            151936

All-to-All:
  Simulates MoE expert dispatch. Each GPU holds hidden states for BATCH_SIZE=64
  tokens and sends them to every other GPU (each GPU holds num_experts/world_size
  experts). Total send buffer per GPU: [world_size * BATCH_SIZE, hidden_size].

All-Reduce:
  Simulates gradient all-reduce of one MoE layer's expert weights:
    128 experts * 3 projections (up, gate, down) * 2048 * 768 = 604M params
    In bf16: ~1.15 GB — the dominant per-layer communication cost.

GPU configurations:
  4 GPUs  — within-node  (GPUs 0-3 on node 0)
  8 GPUs  — within-node  (all 8 GPUs on node 0)
  12 GPUs — cross-node   (6 GPUs on each of 2 nodes)
  16 GPUs — cross-node   (8 GPUs on each of 2 nodes)

Launch (run inside Docker container on head node):
  # 4 GPUs, single node
  torchrun --nproc_per_node=4 test/nccl_benchmark.py

  # 8 GPUs, single node
  torchrun --nproc_per_node=8 test/nccl_benchmark.py

  # 12 GPUs, 2 nodes (run on head; worker connects back)
  torchrun --nnodes=2 --nproc_per_node=6 \\
    --rdzv_backend=c10d --rdzv_endpoint=$HEAD_PRIVATE_IP:29500 \\
    test/nccl_benchmark.py

  # 16 GPUs, 2 nodes
  torchrun --nnodes=2 --nproc_per_node=8 \\
    --rdzv_backend=c10d --rdzv_endpoint=$HEAD_PRIVATE_IP:29500 \\
    test/nccl_benchmark.py

  On the worker node (inside its Docker container), run the same torchrun
  command for multi-node configurations.

To test EFA interface count impact:
  FI_EFA_NUM_NETWORK_CARDS=1 torchrun ...   # restrict to 1 EFA NIC
  torchrun ...                               # use all 4 EFA NICs (default)
"""

import os
import time

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Qwen3-30B-A3B model config
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 768
NUM_EXPERTS = 128
NUM_LAYERS = 48

# ---------------------------------------------------------------------------
# Benchmark config
# ---------------------------------------------------------------------------
BUCKET_BYTES = 25 * 1024 * 1024   # 25 MiB — tensor size for both kernels
DTYPE = torch.bfloat16
WARMUP = 10
ITERS = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def gb(n_bytes: int) -> float:
    return n_bytes / 1e9


def separator(title: str, width: int = 64) -> str:
    return f"\n{'─' * width}\n  {title}\n{'─' * width}"


# ---------------------------------------------------------------------------
# All-to-All benchmark
# ---------------------------------------------------------------------------
def benchmark_alltoall(rank: int, world_size: int, device: torch.device):
    """
    Fixed BUCKET_BYTES total send/recv tensor, split evenly across world_size
    peers.  Matches the all-reduce tensor size for direct comparison.

    Bandwidth reported:
      algbw = (world_size - 1) * chunk_bytes / time
              (data *sent* by one GPU per iteration, excluding self)
      busbw = algbw * (world_size - 1) / world_size   (nccl-tests convention)
    """
    n_elems = BUCKET_BYTES // (torch.finfo(DTYPE).bits // 8)
    # round down to a multiple of world_size so chunks are equal
    n_elems = (n_elems // world_size) * world_size
    send = torch.randn(n_elems, dtype=DTYPE, device=device)
    recv = torch.empty_like(send)
    send_chunks = list(send.chunk(world_size))
    recv_chunks = list(recv.chunk(world_size))

    for _ in range(WARMUP):
        dist.all_to_all(recv_chunks, send_chunks)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(f"  [{time.strftime('%H:%M:%S')}] all-to-all: warmup done, timing {ITERS} iters...", flush=True)

    start = time.perf_counter()
    for _ in range(ITERS):
        dist.all_to_all(recv_chunks, send_chunks)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / ITERS

    total_bytes = send.numel() * send.element_size()
    chunk_bytes = total_bytes // world_size
    algbw = (world_size - 1) * chunk_bytes / elapsed
    busbw = algbw * (world_size - 1) / world_size
    return elapsed, total_bytes, algbw, busbw


# ---------------------------------------------------------------------------
# All-Reduce benchmark
# ---------------------------------------------------------------------------
def benchmark_allreduce(rank: int, world_size: int, device: torch.device):
    """
    Gradient of one MoE layer's expert weights:
      128 experts * (up + gate + down) projections
      = 128 * 3 * 2048 * 768 = 603,979,776 params  (~1.15 GB bf16)

    Bandwidth reported:
      algbw = tensor_bytes / time
      busbw = algbw * 2 * (world_size - 1) / world_size   (ring all-reduce)
    """
    n_elems = BUCKET_BYTES // (torch.finfo(DTYPE).bits // 8)
    tensor = torch.randn(n_elems, dtype=DTYPE, device=device)

    for _ in range(WARMUP):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(f"  [{time.strftime('%H:%M:%S')}] all-reduce: warmup done, timing {ITERS} iters...", flush=True)

    start = time.perf_counter()
    for _ in range(ITERS):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / ITERS

    total_bytes = tensor.numel() * tensor.element_size()
    algbw = total_bytes / elapsed
    busbw = algbw * 2 * (world_size - 1) / world_size
    return elapsed, total_bytes, algbw, busbw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def log(rank, msg):
    """Print a timestamped progress message from rank 0."""
    if rank == 0:
        print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    log(rank, f"rank {rank}/{world_size} initializing NCCL process group...")
    dist.init_process_group("nccl", device_id=device)
    log(rank, "process group ready")

    topology = "within-node" if world_size <= 8 else "cross-node"

    if rank == 0:
        print(separator(f"NCCL Benchmark — {world_size} GPUs ({topology})"))
        print(f"  dtype:       {DTYPE}")
        print(f"  warmup/iters: {WARMUP}/{ITERS}")
        efa_cards = os.environ.get("FI_EFA_NUM_NETWORK_CARDS", "default (all)")
        print(f"  FI_EFA_NUM_NETWORK_CARDS: {efa_cards}")

    dist.barrier()

    # --- All-to-All ---
    log(rank, "all-to-all: warming up...")
    elapsed, total_bytes, algbw, _busbw = benchmark_alltoall(rank, world_size, device)
    if rank == 0:
        print(f"\n  All-to-All")
        print(f"    tensor:    {total_bytes / 1024**2:.1f} MiB total  ({total_bytes / 1024**2 / world_size:.1f} MiB/peer)")
        print(f"    time/iter: {elapsed * 1e3:.3f} ms")
        print(f"    bandwidth: {gb(algbw):.2f} GB/s")

    dist.barrier()

    # --- All-Reduce ---
    log(rank, "all-reduce: warming up...")
    elapsed, total_bytes, algbw, _busbw = benchmark_allreduce(rank, world_size, device)
    if rank == 0:
        print(f"\n  All-Reduce")
        print(f"    tensor:    {total_bytes / 1024**2:.1f} MiB")
        print(f"    time/iter: {elapsed * 1e3:.3f} ms")
        print(f"    bandwidth: {gb(algbw):.2f} GB/s")

    log(rank, "done")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
