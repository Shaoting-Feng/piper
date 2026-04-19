#!/usr/bin/env bash
# Run piper qwen training on EC2 cluster via SSH.
#
# Usage:
#   ./scripts/run-qwen-ec2.sh [OPTIONS]
#
# All options can also be set as environment variables.
#
# Options:
#   --model MODEL    Model size (default: $MODEL or 9B)
#   --schedule NAME  Pipeline schedule (default: $SCHEDULE or 1f1b)
#   --pp N           Pipeline parallel degree (default: $PP or 2)
#   --dp N           Data parallel degree (default: $DP or 1)
#   --ep             Enable expert parallel (default: off)
#   --zero-stage N   ZeRO stage 0/1/2/3 (default: $ZERO_STAGE or 0)
#   --bucket-size N  Bucket size in MB (default: $BUCKET_SIZE if set)
#   --batch-size N   Global batch size (default: $BATCH_SIZE or 4)
#   --seq-len N      Sequence length (default: $SEQ_LEN or 512)
#   --mbs N          Microbatch size (default: $MBS or 4)
#   --gradient-accumulation / --no-gradient-accumulation
#                    Delay bucket reductions to the last occurrence (default: on)
#   --ar-a2a-same-stream / --no-ar-a2a-same-stream
#                    Run A2A ops on the same CUDA stream as AR ops (default: off)
#   --overlap-zero-ops / --no-overlap-zero-ops
#                    Run overlap_zero_ops on the per-rank DAGs (default: off)
#   --overlap-chunks / --no-overlap-chunks
#                    Run overlap_chunks on the per-rank DAGs (default: off)
#   --warmup N       Warmup iterations (default: $WARMUP or 2)
#   --iters N        Training iterations (default: $ITERS or 5)
#   --ray-port PORT  Ray port (default: $RAY_PORT or 6379)
#   --nsight         Enable Nsight profiling (saves to ray_tmp)
#   --log-to-file    Tee all output to out/ec2/<experiment-name>_<timestamp>.log
#   --out-dir DIR    Directory for log files (default: out/ec2)
#   --remote-output-dir DIR
#                    Directory inside the container for test_qwen outputs (default: /tmp/piper/out)
#
# Reads from environment: SSH_KEY, HEAD_PUBLIC_IP, HEAD_PRIVATE_IP

set -euo pipefail

MODEL="${MODEL:-9B}"
SCHEDULE="${SCHEDULE:-1f1b}"
PP="${PP:-2}"
DP="${DP:-1}"
ZERO_STAGE="${ZERO_STAGE:-0}"
BUCKET_SIZE="${BUCKET_SIZE:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEQ_LEN="${SEQ_LEN:-512}"
MBS="${MBS:-4}"
WARMUP="${WARMUP:-2}"
ITERS="${ITERS:-5}"
RAY_PORT="${RAY_PORT:-6379}"
NSIGHT=false
EP=false
LOG_TO_FILE=false
OUT_DIR="out/ec2"
REMOTE_OUTPUT_DIR="/tmp/piper/out"
USE_INDUCTOR_ARG=""
GRADIENT_ACCUMULATION_ARG="--gradient-accumulation"
AR_A2A_SAME_STREAM_ARG="--no-ar-a2a-same-stream"
OVERLAP_ZERO_OPS_ARG="--no-overlap-zero-ops"
OVERLAP_CHUNKS_ARG="--no-overlap-chunks"

SSH_BASE=(-i "$SSH_KEY" -o StrictHostKeyChecking=no)
PROXY="ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2";       shift 2 ;;
        --schedule)    SCHEDULE="$2";    shift 2 ;;
        --pp)          PP="$2";          shift 2 ;;
        --dp)          DP="$2";          shift 2 ;;
        --ep)          EP=true;          shift ;;
        --zero-stage)  ZERO_STAGE="$2";  shift 2 ;;
        --bucket-size) BUCKET_SIZE="$2"; shift 2 ;;
        --batch-size)  BATCH_SIZE="$2";  shift 2 ;;
        --seq-len)     SEQ_LEN="$2";     shift 2 ;;
        --mbs)         MBS="$2";         shift 2 ;;
        --gradient-accumulation) GRADIENT_ACCUMULATION_ARG="--gradient-accumulation"; shift ;;
        --no-gradient-accumulation) GRADIENT_ACCUMULATION_ARG="--no-gradient-accumulation"; shift ;;
        --ar-a2a-same-stream) AR_A2A_SAME_STREAM_ARG="--ar-a2a-same-stream"; shift ;;
        --no-ar-a2a-same-stream) AR_A2A_SAME_STREAM_ARG="--no-ar-a2a-same-stream"; shift ;;
        --overlap-zero-ops) OVERLAP_ZERO_OPS_ARG="--overlap-zero-ops"; shift ;;
        --no-overlap-zero-ops) OVERLAP_ZERO_OPS_ARG="--no-overlap-zero-ops"; shift ;;
        --overlap-chunks) OVERLAP_CHUNKS_ARG="--overlap-chunks"; shift ;;
        --no-overlap-chunks) OVERLAP_CHUNKS_ARG="--no-overlap-chunks"; shift ;;
        --warmup)      WARMUP="$2";      shift 2 ;;
        --iters)       ITERS="$2";       shift 2 ;;
        --ray-port)    RAY_PORT="$2";    shift 2 ;;
        --nsight)      NSIGHT=true;      shift ;;
        --use-inductor) USE_INDUCTOR_ARG="--use-inductor"; shift ;;
        --no-use-inductor) USE_INDUCTOR_ARG="--no-use-inductor"; shift ;;
        --log-to-file) LOG_TO_FILE=true; shift ;;
        --out-dir)     OUT_DIR="$2";     shift 2 ;;
        --remote-output-dir) REMOTE_OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

run_head_ssh() {
    ssh "${SSH_BASE[@]}" "ubuntu@$HEAD_PUBLIC_IP" "$@"
}

run_worker_ssh() {
    local worker_ip="$1"
    shift
    ssh "${SSH_BASE[@]}" -o "$PROXY" "ubuntu@$worker_ip" "$@"
}

snapshot_ray_worker_logs() {
    local node_kind="$1"
    local node_host="${2:-}"
    local snapshot_path="$3"
    local remote_cmd="docker exec piper_ray bash -lc 'for f in /tmp/piper/ray_tmp/session_latest/logs/worker-*.out /tmp/piper/ray_tmp/session_latest/logs/worker-*.err; do [ -f \"\$f\" ] || continue; stat -c \"%n|%s\" \"\$f\"; done'"

    if [[ "$node_kind" == "head" ]]; then
        run_head_ssh "$remote_cmd" > "$snapshot_path" || true
    else
        run_worker_ssh "$node_host" "$remote_cmd" > "$snapshot_path" || true
    fi
}

bundle_ray_worker_logs() {
    local node_label="$1"
    local node_kind="$2"
    local node_host="${3:-}"
    local before_snapshot="$4"
    local bundle_dir="$5"
    local after_snapshot
    local remote_cmd

    mkdir -p "$bundle_dir/$node_label"
    after_snapshot="$(mktemp)"
    snapshot_ray_worker_logs "$node_kind" "$node_host" "$after_snapshot"

    declare -A before_sizes=()
    while IFS='|' read -r path size; do
        [[ -n "$path" ]] || continue
        before_sizes["$path"]="$size"
    done < "$before_snapshot"

    while IFS='|' read -r path size; do
        [[ -n "$path" ]] || continue
        local before_size="${before_sizes[$path]:-0}"
        if [[ "$size" == "$before_size" ]]; then
            continue
        fi
        local fetch_from=1
        if [[ "$size" =~ ^[0-9]+$ ]] && [[ "$before_size" =~ ^[0-9]+$ ]] && (( size > before_size )); then
            fetch_from=$((before_size + 1))
        fi

        local dest="$bundle_dir/$node_label/$(basename "$path")"
        remote_cmd="docker exec piper_ray bash -lc 'tail -c +$fetch_from \"${path}\"'"
        if [[ "$node_kind" == "head" ]]; then
            run_head_ssh "$remote_cmd" > "$dest" || true
        else
            run_worker_ssh "$node_host" "$remote_cmd" > "$dest" || true
        fi
    done < "$after_snapshot"

    rm -f "$after_snapshot"
}

BUCKET_SIZE_LABEL=""
if [[ -n "$BUCKET_SIZE" ]]; then
    BUCKET_SIZE_LABEL="-bucket${BUCKET_SIZE}"
fi

NSIGHT_LABEL=""
if $NSIGHT; then
    NSIGHT_LABEL="-nsight1"
fi

GRADIENT_ACCUMULATION_LABEL=1
if [[ "$GRADIENT_ACCUMULATION_ARG" == "--no-gradient-accumulation" ]]; then
    GRADIENT_ACCUMULATION_LABEL=0
fi

AR_A2A_SAME_STREAM_LABEL=0
if [[ "$AR_A2A_SAME_STREAM_ARG" == "--ar-a2a-same-stream" ]]; then
    AR_A2A_SAME_STREAM_LABEL=1
fi

OVERLAP_ZERO_OPS_LABEL=0
if [[ "$OVERLAP_ZERO_OPS_ARG" == "--overlap-zero-ops" ]]; then
    OVERLAP_ZERO_OPS_LABEL=1
fi

OVERLAP_CHUNKS_LABEL=0
if [[ "$OVERLAP_CHUNKS_ARG" == "--overlap-chunks" ]]; then
    OVERLAP_CHUNKS_LABEL=1
fi

EP_LABEL=0
if $EP; then
    EP_LABEL=1
fi

EXPERIMENT_NAME="qwen${MODEL}-sched_${SCHEDULE}-pp${PP}-dp${DP}-ep${EP_LABEL}-zero${ZERO_STAGE}${BUCKET_SIZE_LABEL}${NSIGHT_LABEL}-bs${BATCH_SIZE}-sl${SEQ_LEN}-mbs${MBS}-ga${GRADIENT_ACCUMULATION_LABEL}-aras${AR_A2A_SAME_STREAM_LABEL}-ozo${OVERLAP_ZERO_OPS_LABEL}-och${OVERLAP_CHUNKS_LABEL}"

if $LOG_TO_FILE; then
    mkdir -p "$OUT_DIR"
    LOG_FILE="$OUT_DIR/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
    echo "Logging to $LOG_FILE"
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

ZERO_ARG=""
if [[ "$ZERO_STAGE" -gt 0 ]]; then
    ZERO_ARG="--zero-stage $ZERO_STAGE"
fi

BUCKET_SIZE_ARG=""
if [[ -n "$BUCKET_SIZE" ]]; then
    BUCKET_SIZE_ARG="--bucket-size $BUCKET_SIZE"
fi

NSIGHT_ARG=""
if $NSIGHT; then
    NSIGHT_ARG="--nsight"
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
REMOTE_RUN_DIR="${REMOTE_OUTPUT_DIR}/runner-fetch/${EXPERIMENT_NAME}_${RUN_ID}"
REMOTE_LOG="${REMOTE_RUN_DIR}/test_qwen.log"
LOCAL_CLUSTER_LOG="$OUT_DIR/${EXPERIMENT_NAME}_${RUN_ID}.cluster.log"
LOCAL_RAY_LOG_BUNDLE_DIR="$OUT_DIR/${EXPERIMENT_NAME}_${RUN_ID}.ray-logs"
SNAPSHOT_DIR="$(mktemp -d)"
HEAD_SNAPSHOT="$SNAPSHOT_DIR/head.snapshot"
WORKER_LOG_NODES=()

for worker_ip in "${WORKER1_PRIVATE_IP:-}" "${WORKER2_PRIVATE_IP:-}" "${WORKER3_PRIVATE_IP:-}"; do
    if [[ -n "$worker_ip" ]]; then
        WORKER_LOG_NODES+=("$worker_ip")
    fi
done

mkdir -p "$OUT_DIR"
echo "Experiment name: $EXPERIMENT_NAME"
echo "Remote run dir: $REMOTE_RUN_DIR"
echo "Remote test_qwen log: $REMOTE_LOG"
echo "Local fetched log: $LOCAL_CLUSTER_LOG"
echo "Local Ray worker log bundle: $LOCAL_RAY_LOG_BUNDLE_DIR"

snapshot_ray_worker_logs "head" "" "$HEAD_SNAPSHOT"
for worker_idx in "${!WORKER_LOG_NODES[@]}"; do
    snapshot_ray_worker_logs "worker" "${WORKER_LOG_NODES[$worker_idx]}" "$SNAPSHOT_DIR/worker$((worker_idx + 1)).snapshot"
done

set +e
run_head_ssh \
  "docker exec -e RAY_DEDUP_LOGS=0 -w /tmp/piper -e PYTHONPATH=/tmp/piper piper_ray \
   bash -lc 'mkdir -p $REMOTE_OUTPUT_DIR $REMOTE_RUN_DIR && python3 -m test.test_qwen \
     --model $MODEL \
     --schedule $SCHEDULE \
     --pp $PP \
     --dp $DP \
     $( $EP && echo --ep ) \
     $ZERO_ARG \
     $BUCKET_SIZE_ARG \
     --batch-size $BATCH_SIZE \
     --seq-len $SEQ_LEN \
     --mbs $MBS \
     $GRADIENT_ACCUMULATION_ARG \
     $AR_A2A_SAME_STREAM_ARG \
     $OVERLAP_ZERO_OPS_ARG \
     $OVERLAP_CHUNKS_ARG \
     --warmup $WARMUP \
     --iters $ITERS \
     --address $HEAD_PRIVATE_IP \
     --port $RAY_PORT \
     --output-dir $REMOTE_OUTPUT_DIR \
     $USE_INDUCTOR_ARG \
     $NSIGHT_ARG \
     --temp-dir /tmp/piper/ray_tmp > $REMOTE_LOG 2>&1'"
RUN_STATUS=$?

run_head_ssh "docker exec piper_ray bash -lc 'test -f $REMOTE_LOG && cat $REMOTE_LOG'" > "$LOCAL_CLUSTER_LOG"
FETCH_STATUS=$?

bundle_ray_worker_logs "head" "head" "" "$HEAD_SNAPSHOT" "$LOCAL_RAY_LOG_BUNDLE_DIR"
for worker_idx in "${!WORKER_LOG_NODES[@]}"; do
    bundle_ray_worker_logs \
        "worker$((worker_idx + 1))" \
        "worker" \
        "${WORKER_LOG_NODES[$worker_idx]}" \
        "$SNAPSHOT_DIR/worker$((worker_idx + 1)).snapshot" \
        "$LOCAL_RAY_LOG_BUNDLE_DIR"
done
set -e

rm -rf "$SNAPSHOT_DIR"

if [[ $FETCH_STATUS -ne 0 ]]; then
    echo "Failed to fetch remote log from $REMOTE_LOG" >&2
    exit $RUN_STATUS
fi

cat "$LOCAL_CLUSTER_LOG"
exit $RUN_STATUS
