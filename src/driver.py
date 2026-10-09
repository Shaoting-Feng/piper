"""Single-driver runtime: placement, actors, process groups, program load, stepping and recovery."""
from __future__ import annotations

import dataclasses
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import ray
import torch
from ray.util.placement_group import placement_group, placement_group_table

from .actor import _bundle_index, _spawn_actor
from .config import ActorConfig, FTConfig
from .device import get_device
from .state import LOG_LEVEL, create_logger
from .topology import Topology

logger = create_logger("driver", LOG_LEVEL)


@dataclass
class Recovery:
    """How a failed step was recovered.

    failed: the ``(dp, pp)`` slot that failed.
    kind: ``"actor_died"`` or ``"task_error"``.
    replacement_rank: global rank of the standby now serving ``failed``.
    strategy: ``"transfer"`` or ``"checkpoint"``.
    next_iter: first iteration to (re-)execute.
    seconds: wall time from detection to a runnable runtime.
    """

    failed: tuple[int, int]
    kind: str
    replacement_rank: int
    strategy: str
    next_iter: int
    seconds: float


@dataclass
class StepResult:
    """Outcome of one ``Runtime.step``.

    losses: microbatch losses per DP slot (empty after a recovery).
    next_iter: iteration to run next.
    recovery: None, or how the step's failure was recovered.
    """

    losses: list[list[float]]
    next_iter: int
    recovery: Recovery | None = None


class StepFailed(RuntimeError):
    """A step task failed on ``slots``; ``kind`` is ``"actor_died"`` or ``"task_error"``."""

    def __init__(self, slots, kind: str, cause: BaseException | None = None, pending: list | None = None):
        super().__init__(f"step failed on slots {list(slots)} ({kind}): {cause}")
        self.slots = list(slots)
        self.kind = kind
        self.cause = cause
        self.pending = list(pending or [])


def create_placement_group(topology: Topology, config: ActorConfig):
    """One bundle per DP replica (per stage when ``pp_outer``), plus one per standby."""
    accel = get_device().accelerator_resource
    if config.pp_outer:
        per_bundle, num_bundles, strategy = topology.dp, topology.pp, "STRICT_SPREAD"
    else:
        per_bundle, num_bundles, strategy = topology.pp, topology.dp, "SPREAD"
    bundle = {"CPU": per_bundle, **({accel: per_bundle} if accel else {})}
    standby = {"CPU": 1, **({accel: 1} if accel else {})}
    return placement_group([bundle] * num_bundles + [standby] * topology.num_standby, strategy=strategy)


def _resolve_actor_config(topology: Topology, config: ActorConfig) -> ActorConfig:
    """``config`` as given, or with the settings standby mode requires when ``topology`` has standbys."""
    if topology.num_standby == 0:
        return config
    if topology.pp != 1 or config.pp_outer:
        raise NotImplementedError("standby ranks require pp == 1 and pp_inner placement")
    # The NCCL watchdog would answer the intentional abort by tearing down
    # every communicator.
    return dataclasses.replace(
        config,
        tensor_transport=True,
        env_vars={**config.env_vars, "TORCH_NCCL_ASYNC_ERROR_HANDLING": "0"},
    )


class Runtime:
    """Actors, process groups and the loaded program of one training job.

    actors: ``{(dp, pp): actor}`` for every trainer slot; a promoted standby
        replaces the failed slot's entry.
    standbys: spare actors that joined the world but hold no slot.
    groups: explicit rank lists of every process group.
    """

    def __init__(self, topology: Topology, ft: FTConfig, config: ActorConfig, pg):
        self.topology = topology
        self.ft = ft
        self.config = config
        self.pg = pg
        self.actors: dict[tuple[int, int], Any] = {}
        self.standbys: list[Any] = []
        self.groups = topology.groups()
        self.program = None
        self._slot_rank: dict[tuple[int, int], int] = {}
        self._standby_ranks: list[int] = []

    @property
    def all_actors(self) -> list[Any]:
        return list(self.actors.values()) + list(self.standbys)

    def _spawn_actors(self) -> None:
        topo, cfg = self.topology, self.config
        for slot in topo.slots():
            rank = topo.rank(*slot)
            self.actors[slot] = _spawn_actor(
                slot, rank, topo, cfg, pg=self.pg, bundle_index=_bundle_index(slot, cfg)
            )
            self._slot_rank[slot] = rank
        first_standby_bundle = topo.pp if cfg.pp_outer else topo.dp
        for i, rank in enumerate(topo.standby_ranks):
            self.standbys.append(
                _spawn_actor(None, rank, topo, cfg, pg=self.pg, bundle_index=first_standby_bundle + i)
            )
            self._standby_ranks.append(rank)

    def _join_groups(self) -> None:
        # Rank 0's node hosts the TCPStore every rank rendezvous on.
        master_addr, master_port = ray.get(self.actors[(0, 0)].get_node_ip_and_free_port.remote())
        logger.debug(f"process-group rendezvous at {master_addr}:{master_port}")
        ray.get([
            actor.join_groups.remote(master_addr, master_port, self.groups)
            for actor in self.all_actors
        ])

    # ---- program ----

    def load(self, program, optim_fn: Callable, const_attrs: dict | None = None) -> None:
        """Ship one program shard per PP rank to every replica and standby.

        program: a ``DagProgram`` (``shard_for``, ``input_pp_ranks``, ``loss_pp_ranks``).
        optim_fn: callable ``(params) -> Optimizer`` applied on each actor.
        const_attrs: non-trainable model attributes (e.g. rope caches) by bare name.
        """
        if program.pp != self.topology.pp:
            raise ValueError(f"program has {program.pp} PP shards but topology has pp={self.topology.pp}")

        attrs = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in (const_attrs or {}).items()
        }
        if attrs:
            ray.get([actor.load_const_attrs.remote(attrs) for actor in self.all_actors])

        refs = [
            actor.load_program.remote(program.shard_for(pp), program.kind, optim_fn)
            for (_dp, pp), actor in self.actors.items()
        ]
        refs += [
            actor.load_program.remote(program.shard_for(0), program.kind, optim_fn)
            for actor in self.standbys
        ]
        ray.get(refs)
        self.program = program
        self._prepare_standbys()

    def _prepare_standbys(self) -> None:
        """Ready the parked standbys for promotion; no-op without standbys."""
        if not self.standbys:
            return
        # Compile now rather than inside the recovery window.
        ray.get([actor.warm_stages.remote() for actor in self.standbys])
        if self.ft.recovery == "transfer":
            ray.get([actor.prepare_promotion_state.remote() for actor in self.all_actors])

    # ---- checkpoints ----

    def _ckpt_path(self, pp: int) -> str:
        return os.path.join(self.ft.ckpt_dir, f"pp{pp}.pt")

    def restore(self) -> int:
        """Load the latest checkpoint in ``ft.ckpt_dir`` into every trainer.

        Returns the iteration to run next: 0 when there is no checkpoint.
        """
        if not self.ft.ckpt_dir:
            return 0
        os.makedirs(self.ft.ckpt_dir, exist_ok=True)
        if not os.path.exists(self._ckpt_path(0)):
            return 0
        iters = ray.get([
            actor.load_checkpoint.remote(self._ckpt_path(pp))
            for (_dp, pp), actor in self.actors.items()
        ])
        if len(set(iters)) != 1:
            raise RuntimeError(f"checkpoint shards disagree on the resume iteration: {iters}")
        logger.info(f"restored checkpoint; resuming at iteration {iters[0]}")
        return iters[0]

    def _checkpoint_if_due(self, it: int) -> None:
        """Save a checkpoint after iteration ``it`` when checkpointing is on and the interval elapsed."""
        if not self.ft.ckpt_dir or (it + 1) % self.ft.ckpt_interval:
            return
        # DP replicas hold identical state; slot dp=0's copy suffices.
        ray.get([
            self.actors[(0, pp)].save_checkpoint.remote(self._ckpt_path(pp))
            for pp in range(self.topology.pp)
        ])

    # ---- stepping ----

    def step(self, it: int, inputs: list[list[torch.Tensor]], labels: list[torch.Tensor], loss_fn) -> StepResult:
        """Run iteration ``it`` on every DP slot; a failed slot is replaced by a
        standby when one exists, otherwise ``StepFailed`` is raised.

        inputs: one list of model inputs per DP slot.
        labels: one label tensor per DP slot.
        loss_fn: ``(output, labels) -> loss`` applied on the loss ranks.
        """
        if self.program is None:
            raise RuntimeError("Runtime.load must be called before step")
        dp = self.topology.dp
        if len(inputs) != dp or len(labels) != dp:
            raise ValueError(f"step expects {dp} input lists and {dp} label tensors, got {len(inputs)} and {len(labels)}")

        input_ranks, loss_ranks = set(self.program.input_pp_ranks), set(self.program.loss_pp_ranks)
        run_refs = {
            actor.run_step.remote(
                it,
                loss_fn,
                inputs[dp_rank] if pp in input_ranks else None,
                labels[dp_rank] if pp in loss_ranks else None,
            ): (dp_rank, pp)
            for (dp_rank, pp), actor in self.actors.items()
        }

        try:
            completed = self._wait_first_completed(run_refs, it)
        except StepFailed as failure:
            if not self.standbys:
                raise
            recovery = self._recover(failure, it)
            return StepResult(losses=[[] for _ in range(dp)], next_iter=recovery.next_iter, recovery=recovery)

        losses: list[list[float]] = [[] for _ in range(dp)]
        for ref, result in completed:
            if isinstance(result, dict):
                losses[run_refs[ref][0]].extend(result.get("losses", []))
        self._checkpoint_if_due(it)
        return StepResult(losses=losses, next_iter=it + 1)

    def _wait_first_completed(self, pending: dict[Any, tuple[int, int]], it: int) -> list[tuple[Any, Any]]:
        """Collect ``(ref, result)`` as tasks finish; the first failure raises ``StepFailed``."""
        refs = list(pending)
        timeout = self.ft.step_timeout
        completed = []
        while refs:
            done, refs = ray.wait(refs, num_returns=1, timeout=timeout)
            if not done:
                # A stuck rank and its peers blocked at the rendezvous time out
                # identically, so no local information names the culprit.
                logger.warning(
                    f"iteration {it} exceeded step_timeout={timeout:.1f}s; "
                    f"still waiting on slots {sorted({pending[r] for r in refs})}"
                )
                continue
            ref = done[0]
            try:
                result = ray.get(ref)
            except ray.exceptions.RayActorError as exc:
                raise StepFailed([pending[ref]], "actor_died", exc, pending=refs) from exc
            except ray.exceptions.RayError as exc:
                raise StepFailed([pending[ref]], "task_error", exc, pending=refs) from exc
            completed.append((ref, result))
        return completed

    # ---- recovery ----

    def _recover(self, failure: StepFailed, it: int) -> Recovery:
        """Replace the failed slot with a standby and bring every member to one state."""
        t0 = time.perf_counter()
        failed_slot = failure.slots[0]
        failed_dp = failed_slot[0]
        survivors = [dp for dp in range(self.topology.dp) if dp != failed_dp]
        standby = self.standbys.pop(0)
        standby_rank = self._standby_ranks.pop(0)
        logger.warning(
            f"iteration {it}: slot {failed_slot} failed ({failure.kind}: "
            f"{str(failure.cause).splitlines()[0][:200]}); promoting standby rank {standby_rank}"
        )

        # Survivors are blocked on the dead peer until their comms are aborted.
        ray.get([self.actors[(dp, 0)].abort_comms.remote() for dp in survivors], timeout=60)
        try:
            ray.kill(self.actors[failed_slot], no_restart=True)
        except Exception:
            logger.debug(f"failed actor of slot {failed_slot} already gone")
        self._drain(failure.pending)

        new_ranks = sorted([self._slot_rank[(dp, 0)] for dp in survivors] + [standby_rank])
        members = [self.actors[(dp, 0)] for dp in survivors] + [standby]
        ray.get([actor.regroup.remote(new_ranks) for actor in members], timeout=180)
        self.groups["dp"] = [new_ranks]

        if self.ft.recovery == "checkpoint":
            iters = ray.get([actor.load_checkpoint.remote(self._ckpt_path(0)) for actor in members], timeout=300)
            next_iter = iters[0]
            logger.info(f"recovery: all members rolled back to checkpoint; resuming at iteration {next_iter}")
        else:
            committed = ray.get([self.actors[(dp, 0)].get_last_committed.remote() for dp in survivors], timeout=60)
            earliest = min(committed)
            source = self.actors[(survivors[committed.index(earliest)], 0)]
            receivers = [standby] + [
                self.actors[(dp, 0)] for dp, lc in zip(survivors, committed) if lc != earliest
            ]
            if len(receivers) > 1:
                logger.warning(f"recovery: survivors committed {committed}; rolling back to {earliest}")
            state_ref = ray.get(source.make_state_ref.remote(), timeout=120)
            # The source must not step its live tensors until every transfer has landed.
            for receiver in receivers:
                ray.get(receiver.load_state.remote([state_ref]), timeout=300)
            del state_ref
            next_iter = earliest + 1
            logger.info(f"recovery: transferred state to {len(receivers)} actor(s); resuming at iteration {next_iter}")

        self.actors[failed_slot] = standby
        self._slot_rank[failed_slot] = standby_rank
        return Recovery(
            failed=failed_slot,
            kind=failure.kind,
            replacement_rank=standby_rank,
            strategy=self.ft.recovery,
            next_iter=next_iter,
            seconds=time.perf_counter() - t0,
        )

    @staticmethod
    def _drain(refs: list, timeout: float = 120.0) -> None:
        """Wait for the fenced step's remaining tasks, ignoring their errors."""
        if not refs:
            return
        done, rest = ray.wait(refs, num_returns=len(refs), timeout=timeout)
        for ref in done:
            try:
                ray.get(ref)
            except Exception:
                pass
        if rest:
            logger.warning(f"{len(rest)} task(s) of the fenced step did not finish within {timeout:.0f}s")


def setup(topology: Topology, ft: FTConfig | None = None, actors: ActorConfig | None = None) -> Runtime:
    """Create the placement group, ``dp * pp + num_standby`` actors and their process groups.

    topology: parallel degrees and standby count.
    ft: fault-tolerance settings.
    actors: per-actor runtime settings.
    """
    ft = ft or FTConfig()
    config = _resolve_actor_config(topology, actors or ActorConfig())
    if not ray.is_initialized():
        ray.init()
    pg = create_placement_group(topology, config)
    ray.get(pg.ready(), timeout=600)
    logger.debug(placement_group_table(pg))
    rt = Runtime(topology, ft, config, pg)
    rt._spawn_actors()
    rt._join_groups()
    return rt
