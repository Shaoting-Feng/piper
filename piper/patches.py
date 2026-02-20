"""
Runtime monkey patches for PyTorch and Ray dependencies.

These patches enable Piper's RemoteTensor to work correctly with TorchDynamo
and Ray's tensor transport backends.

Import this module early in your application to apply patches:
    import src.piper_patches
"""

import functools
import logging

logger = logging.getLogger(__name__)

_patches_applied = False


def apply_patches():
    """Apply all required monkey patches to PyTorch and Ray."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    _patch_ray_actor_method()

    logger.info("Piper runtime patches applied successfully")


def _patch_ray_actor_method():
    """
    Patch ray.actor.ActorMethod._remote to support multiple return values
    with tensor transport backends.

    The default Ray implementation only supports 1 return value per task
    when using tensor transport. This patch removes that restriction and
    properly registers multiple ObjectRefs with the GPU object manager.
    """
    try:
        import ray
        import ray.actor as actor_module
        from ray._raylet import ObjectRef
    except ImportError as e:
        logger.warning(f"Could not import ray.actor: {e}")
        return

    # Get TensorTransportEnum for comparison
    try:
        from ray.actor import TensorTransportEnum
    except ImportError:
        logger.warning("Could not import TensorTransportEnum, skipping Ray patch")
        return

    original_remote = actor_module.ActorMethod._remote

    @functools.wraps(original_remote)
    def patched_remote(
        self,
        args=None,
        kwargs=None,
        name="",
        num_returns=None,
        max_task_retries=None,
        retry_exceptions=None,
        concurrency_group=None,
        _generator_backpressure_num_objects=None,
        enable_task_events=None,
        tensor_transport=None,
    ):
        if num_returns is None:
            num_returns = self._num_returns
        if max_task_retries is None:
            max_task_retries = self._max_task_retries
        if max_task_retries is None:
            max_task_retries = 0
        if retry_exceptions is None:
            retry_exceptions = self._retry_exceptions
        if enable_task_events is None:
            enable_task_events = self._enable_task_events
        if _generator_backpressure_num_objects is None:
            _generator_backpressure_num_objects = (
                self._generator_backpressure_num_objects
            )

        if tensor_transport is None:
            tensor_transport = self._tensor_transport

        # PIPER MODIFICATION: Remove the num_returns != 1 check for tensor_transport
        # Original code would raise ValueError here if num_returns != 1
        if tensor_transport != TensorTransportEnum.OBJECT_STORE.name:
            # Skip the num_returns check - allow multiple returns
            if not self._actor._ray_enable_tensor_transport:
                raise ValueError(
                    f'Currently, methods with .options(tensor_transport="{tensor_transport}") are not supported when enable_tensor_transport=False. '
                    "Please set @ray.remote(enable_tensor_transport=True) on the actor class definition."
                )
            gpu_object_manager = ray._private.worker.global_worker.gpu_object_manager
            if not gpu_object_manager.actor_has_tensor_transport(
                self._actor, tensor_transport
            ):
                raise ValueError(
                    f'{self._actor} does not have tensor transport {tensor_transport} available. If using a collective-based transport ("nccl" or "gloo"), please create a communicator with '
                    "`ray.experimental.collective.create_collective_group` "
                    "before calling actor tasks with non-default tensor_transport."
                )

        args = args or []
        kwargs = kwargs or {}

        def invocation(args, kwargs):
            dst_actor = self._actor
            if dst_actor is None:
                raise RuntimeError(
                    "Lost reference to actor. Actor handles must be stored as variables, e.g. `actor = MyActor.remote()` before calling methods."
                )

            gpu_object_manager = ray._private.worker.global_worker.gpu_object_manager
            gpu_object_manager.trigger_out_of_band_tensor_transfer(dst_actor, args)

            return dst_actor._actor_method_call(
                self._method_name,
                args=args,
                kwargs=kwargs,
                name=name,
                num_returns=num_returns,
                max_task_retries=max_task_retries,
                retry_exceptions=retry_exceptions,
                concurrency_group_name=concurrency_group,
                generator_backpressure_num_objects=(
                    _generator_backpressure_num_objects
                ),
                enable_task_events=enable_task_events,
                tensor_transport=tensor_transport,
            )

        # Apply the decorator if there is one.
        if self._decorator is not None:
            invocation = self._decorator(invocation)

        object_refs = invocation(args, kwargs)

        # PIPER MODIFICATION: Handle multiple return values with GPU object manager
        if tensor_transport != TensorTransportEnum.OBJECT_STORE.name:
            gpu_object_manager = ray._private.worker.global_worker.gpu_object_manager
            if isinstance(object_refs, ObjectRef):
                # Single return value
                object_ref = object_refs
                gpu_object_manager.add_gpu_object_ref(
                    object_ref, self._actor, tensor_transport
                )
            else:
                # Multiple return values - register each ObjectRef
                for object_ref in object_refs:
                    if isinstance(object_ref, ObjectRef):
                        gpu_object_manager.add_gpu_object_ref(
                            object_ref, self._actor, tensor_transport
                        )

        return object_refs

    actor_module.ActorMethod._remote = patched_remote
    logger.debug("Patched ray.actor.ActorMethod._remote")


apply_patches()