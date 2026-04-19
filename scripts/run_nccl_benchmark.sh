#!/usr/bin/env bash
# Run NCCL benchmark combinations and print a summary table.
#
# EFA experiments:    4 GPU configs × 2 NIC settings = 8 experiments
# Socket experiments: 4 GPU configs × 1              = 4 experiments
#   GPU counts:  4 (within-node), 8 (within-node), 12 (cross-node), 16 (cross-node)
#   EFA NICs:    FI_EFA_NUM_NETWORK_CARDS = 1, 4  (ignored for socket)
#
# Requires these env vars (set from aws-manual-setup.md Variables section):
#   SSH_KEY, HEAD_PUBLIC_IP, WORKER_PRIVATE_IP, HEAD_PRIVATE_IP
#
# Run from local machine:
#   ./scripts/run_nccl_benchmark.sh

set -euo pipefail

: "${SSH_KEY:?SSH_KEY must be set}"
: "${HEAD_PUBLIC_IP:?HEAD_PUBLIC_IP must be set}"
: "${WORKER_PRIVATE_IP:?WORKER_PRIVATE_IP must be set}"
: "${HEAD_PRIVATE_IP:?HEAD_PRIVATE_IP must be set}"

RDZV_PORT=29500
LOGFILE=$(mktemp /tmp/nccl_benchmark_XXXXXX.log)
echo "Raw output → $LOGFILE"
echo ""

# Write an SSH config so ProxyCommand quoting is clean
SSH_CFG=$(mktemp /tmp/ssh-cfg-XXXXXX)
trap "rm -f $SSH_CFG" EXIT
cat > "$SSH_CFG" <<EOF
Host head
  HostName $HEAD_PUBLIC_IP
  User ubuntu
  IdentityFile $SSH_KEY
  StrictHostKeyChecking no

Host worker
  HostName $WORKER_PRIVATE_IP
  User ubuntu
  IdentityFile $SSH_KEY
  StrictHostKeyChecking no
  ProxyCommand ssh -F $SSH_CFG -W %h:%p head
EOF

ssh_head()   { ssh -F "$SSH_CFG" head "$@"; }
ssh_worker() { ssh -F "$SSH_CFG" worker "$@"; }

# Results: indexed by [gpus,efa_cards,transport] → "a2a_algbw a2a_busbw ar_algbw ar_busbw"
declare -A RESULTS

BASE_ENV="-e NCCL_SOCKET_IFNAME=ens32 -e GLOO_SOCKET_IFNAME=ens32 -e NCCL_PROTO=simple"

# Build extra env vars for a given transport and NIC count.
# transport: "efa" or "socket"
# efa: NIC count (ignored for socket)
extra_env() {
    local transport=$1 efa=$2
    if [[ $transport == socket ]]; then
        echo "-e NCCL_NET=Socket"
    else
        echo "-e FI_EFA_NUM_NETWORK_CARDS=$efa"
    fi
}

run_single_node() {
    local gpus=$1 efa=$2 transport=$3
    local nic_label; [[ $transport == socket ]] && nic_label="-" || nic_label="$efa"
    printf "  %-16s %-7s NICs=%-2s  running..." "within-node ${gpus}G" "[$transport]" "$nic_label"
    out=$(ssh_head "docker exec \
        $BASE_ENV $(extra_env $transport $efa) \
        -w /tmp/piper piper_ray \
        torchrun --nproc_per_node=$gpus scripts/nccl_benchmark.py" 2>&1)
    echo "$out" >> "$LOGFILE"
    echo "  done"
    parse_results "$out" "$gpus" "$efa" "$transport"
}

run_multi_node() {
    local gpus_per_node=$1 efa=$2 transport=$3
    local total=$((gpus_per_node * 2))
    local nic_label; [[ $transport == socket ]] && nic_label="-" || nic_label="$efa"
    printf "  %-16s %-7s NICs=%-2s  running..." "cross-node ${total}G" "[$transport]" "$nic_label"

    local tmpout
    tmpout=$(mktemp)
    ssh_head "docker exec \
        $BASE_ENV $(extra_env $transport $efa) \
        -w /tmp/piper piper_ray \
        torchrun --nnodes=2 --nproc_per_node=$gpus_per_node \
        --node_rank=0 \
        --master_addr=$HEAD_PRIVATE_IP --master_port=$RDZV_PORT \
        scripts/nccl_benchmark.py" > "$tmpout" 2>&1 &
    HEAD_PID=$!

    ssh_worker "docker exec \
        $BASE_ENV $(extra_env $transport $efa) \
        -w /tmp/piper piper_ray \
        torchrun --nnodes=2 --nproc_per_node=$gpus_per_node \
        --node_rank=1 \
        --master_addr=$HEAD_PRIVATE_IP --master_port=$RDZV_PORT \
        scripts/nccl_benchmark.py" >/dev/null 2>&1 &
    WORKER_PID=$!

    wait $HEAD_PID
    out=$(cat "$tmpout")
    rm -f "$tmpout"
    echo "$out" >> "$LOGFILE"
    echo "  done"
    wait $WORKER_PID

    parse_results "$out" "$total" "$efa" "$transport"
    RDZV_PORT=$((RDZV_PORT + 1))
}

parse_results() {
    local out=$1 gpus=$2 efa=$3 transport=$4
    local a2a_time a2a_bw ar_time ar_bw
    a2a_time=$(echo "$out" | grep -A6 "All-to-All" | grep "time/iter" | awk '{print $2}')
    a2a_bw=$(  echo "$out" | grep -A6 "All-to-All" | grep "bandwidth" | awk '{print $2}')
    ar_time=$( echo "$out" | grep -A6 "All-Reduce"  | grep "time/iter" | awk '{print $2}')
    ar_bw=$(   echo "$out" | grep -A6 "All-Reduce"  | grep "bandwidth" | awk '{print $2}')
    RESULTS["${gpus},${efa},${transport}"]="${a2a_time:-N/A} ${a2a_bw:-N/A} ${ar_time:-N/A} ${ar_bw:-N/A}"
}

# ── Run all experiments ───────────────────────────────────────────────────────

echo "── EFA ──────────────────────────────────────────────────────────────────"
for efa in 1 4; do
    run_single_node  4  $efa  efa
    run_single_node  8  $efa  efa
    run_multi_node   6  $efa  efa   # 12 GPUs total
    run_multi_node   8  $efa  efa   # 16 GPUs total
done

echo ""
echo "── Socket ───────────────────────────────────────────────────────────────"
run_single_node  4  1  socket
run_single_node  8  1  socket
run_multi_node   6  1  socket   # 12 GPUs total
run_multi_node   8  1  socket   # 16 GPUs total

# ── Summary table ─────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  NCCL Benchmark Summary (GB/s)"
echo "═══════════════════════════════════════════════════════════════════════════════"
printf "  %-18s  %-7s  %-5s  %-22s  %-22s\n" \
    "config" "net" "NICs" "All-to-All (ms / GB/s)" "All-Reduce (ms / GB/s)"
echo "  ──────────────────  ───────  ─────  ──────────────────────  ──────────────────────"

for gpus in 4 8 12 16; do
    if   [[ $gpus -le 8 ]]; then label="within-node ${gpus}G"
    else                         label="cross-node  ${gpus}G"
    fi
    for efa in 1 4; do
        key="${gpus},${efa},efa"
        read -r a2a_time a2a_bw ar_time ar_bw <<< "${RESULTS[$key]:-N/A N/A N/A N/A}"
        printf "  %-18s  %-7s  %-5s  %-10s %-10s  %-10s %-10s\n" \
            "$label" "efa" "$efa" "$a2a_time" "$a2a_bw" "$ar_time" "$ar_bw"
    done
    key="${gpus},1,socket"
    read -r a2a_time a2a_bw ar_time ar_bw <<< "${RESULTS[$key]:-N/A N/A N/A N/A}"
    printf "  %-18s  %-7s  %-5s  %-10s %-10s  %-10s %-10s\n" \
        "$label" "socket" "-" "$a2a_time" "$a2a_bw" "$ar_time" "$ar_bw"
done
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  Full output: $LOGFILE"
