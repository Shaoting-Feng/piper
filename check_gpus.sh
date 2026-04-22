export SSH_KEY=/home/shubham/.ssh/mfris-us-west-2.pem
export HEAD_PUBLIC_IP=100.22.89.85
export HEAD_PRIVATE_IP=10.0.79.139
export WORKER1_PRIVATE_IP=10.0.71.178
export WORKER2_PRIVATE_IP=10.0.70.34
export WORKER3_PRIVATE_IP=10.0.69.189

for ENTRY in "head:$HEAD_PUBLIC_IP" \
             "worker1:$WORKER1_PRIVATE_IP" \
             "worker2:$WORKER2_PRIVATE_IP" \
             "worker3:$WORKER3_PRIVATE_IP"; do
  LABEL=${ENTRY%%:*}; HOST=${ENTRY#*:}

  if [[ "$LABEL" == "head" ]]; then
    SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no)
  else
    SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no \
              -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP")
  fi

  echo ""
  echo "============================================================"
  echo "=== $LABEL ($HOST) ==="
  echo "============================================================"
  ssh "${SSH_OPTS[@]}" ubuntu@$HOST "echo 'Hostname: \$(hostname)'; nvidia-smi"
done
