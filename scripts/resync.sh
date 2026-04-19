rsync -av --delete \
  --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
  --exclude='venv' --exclude='out' --exclude='e2e-eval' --exclude='ray_tmp' --exclude='aws' \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  . ubuntu@$HEAD_PUBLIC_IP:/tmp/piper/

for WORKER_IP in $WORKER1_PRIVATE_IP $WORKER2_PRIVATE_IP $WORKER3_PRIVATE_IP; do
  rsync -av --delete \
    --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
    --exclude='venv' --exclude='out' --exclude='e2e-eval' --exclude='ray_tmp' --exclude='aws' \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o 'ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP'" \
    . ubuntu@$WORKER_IP:/tmp/piper/
done

ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$HEAD_PUBLIC_IP \
  "docker cp /tmp/piper/. piper_ray:/tmp/piper/"

for WORKER_IP in $WORKER1_PRIVATE_IP $WORKER2_PRIVATE_IP $WORKER3_PRIVATE_IP; do
  ssh -i $SSH_KEY -o StrictHostKeyChecking=no \
    -o "ProxyCommand=ssh -i $SSH_KEY -o StrictHostKeyChecking=no -W %h:%p ubuntu@$HEAD_PUBLIC_IP" \
    ubuntu@$WORKER_IP \
    "docker cp /tmp/piper/. piper_ray:/tmp/piper/" &
done
wait
