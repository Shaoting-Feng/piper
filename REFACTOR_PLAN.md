# Piper Refactor Plan

This document records the agreed refactor direction for the Piper implementation
under `src/`. It is a planning document, not an implementation checklist that
must be completed in one change.

## Goals

- Improve module boundaries and make each file responsible for one coherent
  part of the system.
- Remove `piper_` from implementation filenames. Only the main backend module
  should keep the name `piper.py`.
- Keep `src.piper` as the public user-facing API surface for `annotate`, backend
  registration, and execution entry points.
- Preserve the DAG as the single source of scheduling policy, including
  communication, ZeRO allocation/free behavior, and task order.
- Reduce `PiperActor` complexity by grouping state and extracting execution
  helpers while keeping the top-level DAG execution loop easy to read.
- Make illegal states fail early through schedule/API validation rather than
  through indirect runtime failures.

## Refactoring Principles

- Separate data structures, parsing, graph transforms, runtime orchestration,
  and distributed execution.
- Prefer mechanical moves before behavioral rewrites so diffs remain reviewable.
- Avoid compatibility shims unless explicitly required.
- Avoid parallel dictionaries where one typed state object can express ownership.
- Keep the schedule execution semantics visible: the top-level executor should
  still iterate over the sorted DAG and dispatch by task type.
- Keep ZeRO policy out of parameter storage. The DAG decides when to allocate,
  free, gather, scatter, and sync.

## Target Module Layout

### `piper.py`

Public API and backend registration only:

- `annotate`
- annotation state reset
- `piper` backend callback
- `piper_exec_dag`
- minimal public glue

`piper.py` should delegate DAG construction, schedule directive application,
ordering, visualization, and runtime setup to focused modules.

### `actor.py`

Ray actor API shell and lifecycle:

- actor construction
- process group setup
- loading inputs, labels, constants, stages, and DAGs
- profiler/NVTX public methods
- delegating task execution to `DagExecutor`

The actor should own composed state objects, not a large collection of unrelated
parallel dictionaries.

### `coordinator.py`

Distributed program coordination:

- `PiperProgramCoordinator`
- `run_dp_rank`
- placement group construction

### `compile.py`

Compile-time orchestration:

- `piper_setup`
- compile-time Ray stores
- model construction/tracing setup
- pushing compiled DAG data to actors

### `schedule.py`

Schedule JSON loading and schema-level validation:

- load schedule directives
- validate current JSON API shape
- derive schedule info such as PP/DP degrees and microbatch count

### `tasks.py`

Task type enum and task-level constants.

### `state.py`

Shared process-local state and logging:

- `PiperMetadata`
- global metadata instance
- `create_logger`

### `fx.py`

FX graph infrastructure:

- GraphModule serialization/deserialization
- annotation stack validation
- annotation-based FX splitting into segments

This combines the current FX serialization helpers and FX splitting logic,
because both are infrastructure around `torch.fx.GraphModule` representation.

### `bucket.py`

Parameter bucket planning and bucket GraphModule construction.

This remains separate from `fx.py` because bucketing has separate invariants and
is tied to optimizer/parameter ownership rather than generic FX transport.

### `dag.py`

Training DAG data structures, construction, and core graph utilities:

- `TrainingDAG`
- `TrainingDAGNode`
- `TrainingDAGEdge`
- construction of the initial TrainingDAG from annotated FX segments
- graph mutation helpers
- basic topological utilities that are not specifically ordering directives

### `directives.py`

Schedule directive parsing, matching, validation, and application.

This file should include communication insertion because communication nodes are
created as the effect of applying schedule directives:

- `place`
- `replicate`
- `shard`
- `split`
- `order` parsing handoff
- send/recv insertion
- reduce/all-reduce insertion
- all-gather insertion
- reduce-scatter insertion
- A2A insertion

There should not be a separate `comm_transforms.py`; communication transforms
are part of directive application.

### `ordering.py`

Ordering-specific graph logic:

- serial and per-stream topological ordering
- order directive sub-DAG matching
- temporal edge insertion for order directives
- stream total order resolution

### `zero.py`

ZeRO-specific DAG metadata and graph rewrites:

- ZeRO lifetime metadata pruning
- inter-chain temporal edge insertion around ZeRO lifetimes
- helper logic for full-param/full-grad lifetime correctness

This module should operate on the DAG and metadata. It should not own runtime
parameter storage policy.

### `visualization.py`

Debug/render output:

- TrainingDAG order dumps
- Graphviz rendering
- schedule visualization currently produced by `examples/build_schedule.py`
- dependency logging if it remains debug-output oriented

The example harness should expose a single `--viz` flag. That flag should guard
both schedule visualization and TrainingDAG visualization; there should not be
separate schedule-only or DAG-only example flags unless a later use case needs
that split.

### `backward.py`

Current `backward_utils.py` helpers for split backward/autograd graph traversal.

## Actor Design

`PiperActor` should become a thin Ray shell around typed runtime components.

### `DagExecutor`

Owns the top-level sorted-DAG execution loop.

This should remain the readable control plane:

```python
for node in sorted_nodes:
    match task_type:
        case TaskType.FWD:
            compute.forward(node)
        case TaskType.BWD:
            compute.backward(node)
        case TaskType.SEND:
            communication.send(node)
        case TaskType.RECV:
            communication.recv(node)
        case TaskType.ALL_GATHER:
            communication.all_gather(node)
        case TaskType.REDUCE_SCATTER:
            communication.reduce_scatter(node)
        case TaskType.A2A:
            communication.a2a(node)
```

`DagExecutor` also handles metadata-driven lifecycle hooks, because the DAG is
the source of policy:

```python
if node.meta.get("zero_alloc_full_grads_before"):
    stage_store.params.alloc_full_grads(...)

execute_task(node)

if node.meta.get("zero_free_full_params_after"):
    stage_store.params.free_full_params(...)
```

### `ComputeExecutor`

Executes compute tasks:

- forward
- fused backward
- backward input
- backward weight
- loss application
- optimizer/update execution where appropriate

It should use `StageStore`, `BufferStore`, and `ParamStorage` rather than owning
parameter or buffer dictionaries itself.

### `CommunicationExecutor`

Executes communication tasks:

- send
- recv
- all-reduce
- reduce-scatter
- all-gather
- A2A

Send and recv are generic communication tasks. There are no direction-specific
send/recv task types in the current implementation and the refactor should not
introduce them.

### `StageStore`

Owns loaded executable stage/bucket state:

- bucket modules
- graph args
- input and parameter indices
- optimizers
- bucket metadata
- parameter storage

### `BucketState`

One executable bucket:

- bucket id
- GraphModule/submodules
- placeholder metadata
- optimizer
- parameter metadata
- task/bucket mode metadata

### `ParamStorage`

Storage and tensor lookup only. It should not encode ZeRO scheduling policy.

Responsibilities:

- own parameter tensors or local shards
- expose args for forward/compute
- map placeholder names/indices to tensors
- store and clear full parameter materializations
- store and clear full gradient buffers
- expose local shards and grad shards
- provide trainable parameter views/layout metadata

The DAG and `DagExecutor` decide when to allocate/free/gather/scatter. This
avoids separate `Zero2ParamState` or `Zero3ParamState` policy classes, because
ZeRO behavior is already encoded by DAG nodes and metadata.

### `BufferStore`

Owns runtime buffers:

- forward outputs
- backward outputs
- recv payloads
- send payloads
- task buffers and refcounts

### `RuntimeState`

Owns runtime environment:

- ranks and degrees
- device
- process groups
- streams
- stream lookup
- profiler/NVTX helpers

## Staged Implementation Plan

1. Rename files mechanically:
   - `piper_actor.py` -> `actor.py`
   - `piper_compile.py` -> `compile.py`
   - `piper_coordinator.py` -> `coordinator.py`
   - `piper_exec.py` -> `tasks.py`
   - `piper_graph_transform.py` -> temporary `fx.py` or split later
   - `piper_schedule.py` -> `schedule.py`
   - `piper_utils.py` -> temporary `state.py` or split later
   - `backward_utils.py` -> `backward.py`

2. Update all imports, tests, and examples. Run unit tests and the Qwen
   verification matrix before doing semantic moves.

3. Split state and FX helpers:
   - move metadata/logging into `state.py`
   - move GraphModule serialization into `fx.py`
   - move annotation splitting into `fx.py`

4. Move bucket planning into `bucket.py`.

5. Move TrainingDAG dataclasses, initial DAG construction from annotated FX
   segments, and core graph utilities into `dag.py`.

6. Move schedule directive parsing/application and communication insertion into
   `directives.py`.

7. Move ordering-specific logic into `ordering.py`.

8. Move ZeRO DAG lifetime/metadata transforms into `zero.py`.

9. Move debug rendering, schedule visualization, and DAG order output into
   `visualization.py`. Rename the example visualization flag from `--save-viz`
   to `--viz` and use it to guard both schedule and DAG visualization.

10. Refactor `PiperActor` internally:
    - introduce `RuntimeState`
    - introduce `StageStore`
    - introduce `BucketState`
    - introduce `ParamStorage`
    - introduce `BufferStore`
    - introduce `ComputeExecutor`
    - introduce `CommunicationExecutor`
    - introduce `DagExecutor`

11. Keep the top-level DAG loop visible in `DagExecutor`; avoid replacing it
    with a registry of opaque callbacks unless there is a clear benefit.

12. After each stage:
    - run `python3 -m py_compile` on touched modules
    - run `.venv/bin/python -m pytest test`
    - run direct-file PP-only Qwen smoke tests with the current requested GPU
      mask (`CUDA_VISIBLE_DEVICES=0,2`) on:
      - PP2 topology with 1F1B
      - V-placement PP topology with DualPipeV

DualPipeV verification should use the corresponding V-placement base schedule,
because the generated DualPipeV order references virtual pipeline stages.

## Current Progress

- Stages 1-9 are implemented:
  - implementation modules have been renamed away from `piper_*`
  - state/logging, FX serialization/splitting, bucket planning, DAG
    construction, directive application, ordering, ZeRO DAG transforms, and
    visualization have been moved into focused modules
  - `examples/test_harness.py` uses `--viz` as the single visualization flag,
    and `piper_setup(..., visualize_dag=False)` is the default
- Stage 10 is implemented:
  - `RuntimeState` owns ranks, device selection, process groups, CUDA streams,
    and profiler/NVTX helpers
  - `BufferStore` owns per-iteration task outputs and refcounts
  - `EventStore` owns per-iteration non-compute CUDA events
  - `StageStore` owns loaded stage/bucket dictionaries and ZeRO bucket-mode sets
  - `BucketState` owns per-bucket forward args/functions, optimizer state, and
    ZeRO storage buffers, replacing the previous parallel `bucket_*` maps
  - `CommunicationExecutor` owns P2P send/recv, A2A, all-reduce, and
    reduce-scatter execution
  - `ComputeExecutor` owns bucket forward, fused backward, split backward-input,
    split backward-weight, and compute-loss debug logging
  - `ParamStorage` owns ZeRO full-param/full-grad storage, pending async frees,
    flat gradient accumulation, all-gather for sharded params, and ZeRO shard
    optimizer cleanup
  - `DagExecutor` owns the top-level sorted-DAG `match task_type` loop, as
    intended
- Stage 10 actor cleanup is implemented.

Latest verification after introducing `RuntimeState`:

- `.venv/bin/python -m py_compile src/*.py examples/*.py examples/models/*.py test/*.py scripts/test_env.py`
- `.venv/bin/python -m pytest test` (`15 passed`)
- `CUDA_VISIBLE_DEVICES=2,3` PP2 + 1F1B Qwen run:
  `out/20260601_203107/results.csv`
- `CUDA_VISIBLE_DEVICES=2,3` V-placement PP + DualPipeV Qwen run:
  `out/20260601_203147/results.csv`

Latest verification after introducing `BucketState`:

- `.venv/bin/python -m py_compile src/*.py examples/*.py examples/models/*.py test/*.py scripts/test_env.py`
- `.venv/bin/python -m pytest test` (`15 passed`)
- `CUDA_VISIBLE_DEVICES=2,3` PP2 + 1F1B Qwen run:
  `out/20260601_203620/results.csv`
- `CUDA_VISIBLE_DEVICES=2,3` V-placement PP + DualPipeV Qwen run:
  `out/20260601_203700/results.csv`

Latest verification after introducing `CommunicationExecutor`:

- `.venv/bin/python -m py_compile src/*.py examples/*.py examples/models/*.py test/*.py scripts/test_env.py`
- `.venv/bin/python -m pytest test` (`15 passed`)
- `CUDA_VISIBLE_DEVICES=0,2` PP2 + 1F1B Qwen run:
  `out/20260601_204331/results.csv`
- `CUDA_VISIBLE_DEVICES=0,2` V-placement PP + DualPipeV Qwen run:
  `out/20260601_204412/results.csv`

Latest verification after introducing `ComputeExecutor`:

- `.venv/bin/python -m py_compile src/*.py examples/*.py examples/models/*.py test/*.py scripts/test_env.py`
- `.venv/bin/python -m pytest test` (`15 passed`)
- `CUDA_VISIBLE_DEVICES=0,2` PP2 + 1F1B Qwen run:
  `out/20260601_204816/results.csv`
- `CUDA_VISIBLE_DEVICES=0,2` V-placement PP + DualPipeV Qwen run:
  `out/20260601_204855/results.csv`

Latest verification after introducing `ParamStorage`:

- `.venv/bin/python -m py_compile src/*.py examples/*.py examples/models/*.py test/*.py scripts/test_env.py`
- `.venv/bin/python -m pytest test` (`15 passed`)
- `CUDA_VISIBLE_DEVICES=0,2` PP2 + 1F1B Qwen run:
  `out/20260601_205514/results.csv`
- `CUDA_VISIBLE_DEVICES=0,2` V-placement PP + DualPipeV Qwen run:
  `out/20260601_205554/results.csv`

Latest verification after introducing `DagExecutor`:

- `.venv/bin/python -m py_compile src/*.py examples/*.py examples/models/*.py test/*.py scripts/test_env.py`
- `.venv/bin/python -m pytest test` (`15 passed`)
- Initial `CUDA_VISIBLE_DEVICES=0,2` PP2 + 1F1B Qwen run exposed an eager
  actor input-state bug after the loop move; fixed by initializing
  `PiperActor.inputs`
- `CUDA_VISIBLE_DEVICES=0,2` PP2 + 1F1B Qwen run after the fix:
  `out/20260601_210222/results.csv`
- `CUDA_VISIBLE_DEVICES=0,2` V-placement PP + DualPipeV Qwen run:
  `out/20260601_210306/results.csv`

## Resolved Decisions

- `visualization.py` should own schedule visualization as well as DAG
  visualization. The example `--viz` flag should guard both outputs.
- Send/recv task types remain generic; the refactor should not add directional
  send/recv variants.
- `piper_exec_dag` stays in `piper.py`.
