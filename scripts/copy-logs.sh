#!/usr/bin/env bash
# Copy training logs and/or Nsight profiles from the EC2 cluster.
# Requires: SSH_KEY, HEAD_PUBLIC_IP, WORKER1_PRIVATE_IP, WORKER2_PRIVATE_IP, WORKER3_PRIVATE_IP

set -euo pipefail

NSIGHT=false
NSIGHT_ONLY=false
OUT_DIR="out/ec2"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nsight)       NSIGHT=true; shift ;;
        --nsight-only)  NSIGHT=true; NSIGHT_ONLY=true; shift ;;
        --out-dir)      OUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no"
PROXY="ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP"

mkdir -p "$OUT_DIR"

if ! $NSIGHT_ONLY; then
    # Copy training log files from head container.
    $SSH ubuntu@$HEAD_PUBLIC_IP \
      "rm -rf /tmp/piper-out && docker cp piper_ray:/tmp/piper/out/. /tmp/piper-out/"
    scp -r -i "$SSH_KEY" ubuntu@$HEAD_PUBLIC_IP:/tmp/piper-out/. "$OUT_DIR/"
fi

if $NSIGHT; then
    # Head node Nsight profile
    mkdir -p "$OUT_DIR/nsight-head"
    $SSH ubuntu@$HEAD_PUBLIC_IP \
      "rm -rf /tmp/nsight-head && mkdir -p /tmp/nsight-head && \
       docker cp piper_ray:/tmp/piper/ray_tmp/session_latest/logs/nsight/. /tmp/nsight-head/"
    scp -r -i "$SSH_KEY" ubuntu@$HEAD_PUBLIC_IP:/tmp/nsight-head/. "$OUT_DIR/nsight-head/"

    # Worker Nsight profiles
    WORKER_IDX=1
    for WORKER_IP in $WORKER1_PRIVATE_IP $WORKER2_PRIVATE_IP; do
        mkdir -p "$OUT_DIR/nsight-worker${WORKER_IDX}"
        $SSH -o "$PROXY" ubuntu@$WORKER_IP \
          "rm -rf /tmp/nsight-worker && mkdir -p /tmp/nsight-worker && \
           docker cp piper_ray:/tmp/piper/ray_tmp/session_latest/logs/nsight/. /tmp/nsight-worker/"
        scp -r -i "$SSH_KEY" \
          -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
          ubuntu@$WORKER_IP:/tmp/nsight-worker/. "$OUT_DIR/nsight-worker${WORKER_IDX}/"
        WORKER_IDX=$((WORKER_IDX + 1))
    done
fi
