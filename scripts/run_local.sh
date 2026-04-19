#!/bin/bash
# Run piper inside Docker locally. Usage:
#   ./run_local.sh test.test_env
#   ./run_local.sh test.test_mixtral --model tiny --schedule 1f1b --pp 2 --dp 1 --mbs 4

set -e

MODULE="$1"
shift

docker run --rm \
    --gpus all \
    --shm-size=10g \
    --network host \
    -v "$(pwd)":/tmp/piper \
    piper:latest \
    bash -c "cd /tmp/piper && ray start --head --num-cpus=$(nproc) --num-gpus=$(nvidia-smi -L | wc -l) && python -m $MODULE $*"
