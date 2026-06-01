# Piper

Piper is a user-controllable distributed training system that decouples distributed training strategy from runtime implementation. Piper lets users express high-level placement and low-level scheduling intent through lightweight model annotations and scheduling directives. The compiler lowers this intent into a unified global training DAG that explicitly represents computation, communication, data dependencies, temporal dependencies, device placement, and GPU stream assignment. A centralized scheduler then decomposes the DAG into per-device execution plans, and a Ray-based distributed runtime executes those plans on GPU workers while managing streams, communicators, and memory. This design allows Piper to match existing general-purpose training systems on common strategies while making it easier to express and optimize composed strategies such as PP x EP/DP DualPipe-style schedules.

![Piper architecture](figs/piper-architecture.png)

## Install

The following installation has been tested with Python 3.10 on Linux with CUDA 12.8 drivers. 

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Quickstart

Run the test harness with the `pp2` base schedule and a generated 1F1B order schedule:

```bash
python examples/test_harness.py \
  --test-file examples/test_qwen.py \
  --base-schedule examples/base-schedules/pp2.json \
  --schedule 1f1b \
  --ranks 2 \
  --mbs 4
```

Results and the generated full schedule are written under `out/<timestamp>/`. Add `--viz` to also render the generated schedule and per-rank TrainingDAGs.

## Citation

TBD

## Contact

TBD
