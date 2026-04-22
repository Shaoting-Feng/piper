export PATH_TO_PIPER=/m-coriander/coriander/shubham/moe-scheduling/piper/piper
export IMAGE=mfris/piper:latest
# export SSH_KEY=/m-coriander/coriander/shubham/ray-autoscaler_us-east-2.pem
export SSH_KEY=/home/shubham/.ssh/mfris-us-west-2.pem

export HEAD_PUBLIC_IP=100.22.89.85
export HEAD_PRIVATE_IP=10.0.79.139
export WORKER1_PRIVATE_IP=10.0.71.178
export WORKER2_PRIVATE_IP=10.0.70.34
export WORKER3_PRIVATE_IP=10.0.69.189
export WORKERS="$WORKER1_PRIVATE_IP $WORKER2_PRIVATE_IP $WORKER3_PRIVATE_IP"


./scripts/resync.sh


python scripts/run_qwen_e2e_eval.py \
  --model 9B \
  --sweeps zero \
  --zero-stages 2 3 \
  --zero-grad-acc-settings off \
  --bucket-size 25 \
  --batch-size 4 \
  --zero-batch-sizes 2 4 8 12 16 24 28 32 \
  --seq-len 128 \
  --memory-profile \
  --pp 2 \
  --dp 4 \
  --warmup 2 \
  --iters 2

# python scripts/run_qwen_e2e_eval.py \
#   --model 9B \
#   --sweeps zero \
#   --zero-stages 2 \
#   --zero-grad-acc-settings off \
#   --bucket-size 25 \
#   --batch-size 4 \
#   --zero-batch-sizes 8 12 16 \
#   --seq-len 128 \
#   --memory-profile \
#   --pp 2 \
#   --dp 4 \
#   --warmup 2 \
#   --iters 2

# LOCAL_DIR=/m-coriander/coriander/shubham/moe-scheduling/piper/piper/e2e-eval/20260419_235909/logs/zero-sched_1f1b-pp4-dp2-ep0-zero3-bucket25-bs32-sl512-mbs8-ga0-aras0-ozo1-och0/memory_snapshots
# mkdir -p "$LOCAL_DIR"
# REMOTE_DIR=/tmp/piper/out/zero-sched_1f1b-pp4-dp2-ep0-zero3-bucket25-bs32-sl512-mbs8-ga0-aras0-ozo1-och0/memory_snapshots

# for ENTRY in "head:$HEAD_PUBLIC_IP" \
#              "worker1:$WORKER1_PRIVATE_IP" \
#              "worker2:$WORKER2_PRIVATE_IP" \
#              "worker3:$WORKER3_PRIVATE_IP"; do
#   LABEL=${ENTRY%%:*}; HOST=${ENTRY#*:}
#   [[ -z "$HOST" ]] && continue

#   if [[ "$LABEL" == "head" ]]; then
#     SSH_OPTS=()
#   else
#     SSH_OPTS=(-o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP")
#   fi

#   STAGE=/tmp/zero3_snaps_${LABEL}
#   echo "=== $LABEL ($HOST) ==="
#   ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "${SSH_OPTS[@]}" ubuntu@$HOST \
#       "rm -rf $STAGE && mkdir -p $STAGE && \
#        docker cp piper_ray:$REMOTE_DIR $STAGE/ 2>/dev/null; \
#        ls $STAGE/memory_snapshots/ 2>/dev/null"

#   scp -r -i "$SSH_KEY" -o StrictHostKeyChecking=no "${SSH_OPTS[@]}" \
#       "ubuntu@$HOST:$STAGE/memory_snapshots/." "$LOCAL_DIR/" 2>/dev/null \
#       && echo "  fetched from $LABEL" \
#       || echo "  no snapshots on $LABEL (skip)"
# done

# echo "=== Final ==="
# ls -la "$LOCAL_DIR"





