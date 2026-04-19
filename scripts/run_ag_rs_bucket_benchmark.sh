#!/usr/bin/env bash
# Run the AR vs RS bucket microbenchmark across 2 nodes × 1 GPU.
#
# Requires:
#   SSH_KEY, HEAD_PUBLIC_IP, HEAD_PRIVATE_IP, and either WORKER1_PRIVATE_IP or WORKER_PRIVATE_IP
#
# Usage:
#   ./scripts/run_ag_rs_bucket_benchmark.sh [--warmup N] [--iters N] [--dtype DTYPE]

set -euo pipefail

: "${SSH_KEY:?SSH_KEY must be set}"
: "${HEAD_PUBLIC_IP:?HEAD_PUBLIC_IP must be set}"
: "${HEAD_PRIVATE_IP:?HEAD_PRIVATE_IP must be set}"

WORKER_IP="${WORKER1_PRIVATE_IP:-${WORKER_PRIVATE_IP:-}}"
: "${WORKER_IP:?WORKER1_PRIVATE_IP or WORKER_PRIVATE_IP must be set}"

WARMUP=10
ITERS=50
DTYPE=bfloat16
RDZV_PORT="${RDZV_PORT:-29500}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --warmup) WARMUP="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SSH_CFG=$(mktemp /tmp/ag-rs-ssh-cfg-XXXXXX)
trap 'rm -f "$SSH_CFG"' EXIT
cat > "$SSH_CFG" <<EOF
Host head
  HostName $HEAD_PUBLIC_IP
  User ubuntu
  IdentityFile $SSH_KEY
  StrictHostKeyChecking no

Host worker
  HostName $WORKER_IP
  User ubuntu
  IdentityFile $SSH_KEY
  StrictHostKeyChecking no
  ProxyCommand ssh -F $SSH_CFG -W %h:%p head
EOF

ssh_head()   { ssh -F "$SSH_CFG" head "$@"; }
ssh_worker() { ssh -F "$SSH_CFG" worker "$@"; }

BASE_ENV="-e NCCL_SOCKET_IFNAME=ens32 -e GLOO_SOCKET_IFNAME=ens32"
BENCH_CMD="docker exec $BASE_ENV -w /tmp/piper piper_ray bash -lc 'torchrun --nnodes=2 --nproc_per_node=1 --node_rank=%NODE_RANK% --master_addr=$HEAD_PRIVATE_IP --master_port=$RDZV_PORT test/allgather_reduce_scatter_bucket_bench.py --warmup $WARMUP --iters $ITERS --dtype $DTYPE'"

tmpout=$(mktemp)
trap 'rm -f "$SSH_CFG" "$tmpout"' EXIT

echo "Starting AR/RS bucket benchmark on head=$HEAD_PUBLIC_IP worker=$WORKER_IP port=$RDZV_PORT"

ssh_head "${BENCH_CMD//%NODE_RANK%/0}" 2>&1 | tee "$tmpout" &
HEAD_PID=$!

ssh_worker "${BENCH_CMD//%NODE_RANK%/1}" >/dev/null 2>&1 &
WORKER_PID=$!

set +e
wait "$HEAD_PID"
HEAD_STATUS=$?
wait "$WORKER_PID"
WORKER_STATUS=$?
set -e

if [[ $HEAD_STATUS -ne 0 || $WORKER_STATUS -ne 0 ]]; then
    echo "Benchmark failed: head_status=$HEAD_STATUS worker_status=$WORKER_STATUS" >&2
    exit 1
fi
