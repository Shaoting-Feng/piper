"""Smoke test to verify the Ray + CUDA environment is set up correctly."""
import sys
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("timed out")

def check(name, fn, timeout_sec=30):
    sys.stdout.flush()
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_sec)
        result = fn()
        signal.alarm(0)
        print(f"  [PASS] {name}: {result}", flush=True)
        return True
    except Exception as e:
        signal.alarm(0)
        print(f"  [FAIL] {name}: {e}", flush=True)
        return False

def main():
    ok = True

    print("Python", flush=True)
    ok &= check("version", lambda: sys.version.split()[0])

    print("PyTorch", flush=True)
    import torch
    ok &= check("version", lambda: torch.__version__)
    ok &= check("CUDA available", lambda: torch.cuda.is_available())
    ok &= check("CUDA version", lambda: torch.version.cuda)
    ok &= check("GPU count", lambda: torch.cuda.device_count())
    if torch.cuda.is_available():
        ok &= check("GPU name", lambda: torch.cuda.get_device_name(0))
        ok &= check("tensor on GPU", lambda: torch.zeros(1, device="cuda").device)

    print("NCCL", flush=True)
    ok &= check("available", lambda: torch.cuda.nccl.is_available(torch.randn(1, device="cuda")))
    ok &= check("version", lambda: ".".join(str(v) for v in torch.cuda.nccl.version()))

    print("CuPy", flush=True)
    import cupy
    ok &= check("version", lambda: cupy.__version__)

    print("EFA", flush=True)
    import os, subprocess
    def check_efa_devices():
        devs = os.listdir("/dev/infiniband") if os.path.isdir("/dev/infiniband") else []
        if not devs:
            raise RuntimeError("no devices under /dev/infiniband")
        return devs
    def check_fi_info():
        result = subprocess.run(["fi_info", "-p", "efa"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "fi_info -p efa returned non-zero")
        providers = [l for l in result.stdout.splitlines() if l.startswith("provider:")]
        return f"{len(providers)} EFA provider(s) found"
    ok &= check("devices", check_efa_devices)
    try:
        result = check_fi_info()
        print(f"  [PASS] fi_info: {result}", flush=True)
    except FileNotFoundError:
        print("  [WARN] fi_info: not installed in container (host-side tool); skipping", flush=True)
    except Exception as e:
        print(f"  [FAIL] fi_info: {e}", flush=True)
        ok = False
    print("  [INFO] To confirm NCCL uses EFA: set NCCL_DEBUG=INFO before running a job and look for", flush=True)
    print("         'NET/OFI Initializing aws-ofi-nccl' and 'Using EFA' in the output.", flush=True)

    print("Piper imports", flush=True)
    ok &= check("compile", lambda: (__import__("src.compile"), "ok")[1])
    ok &= check("tasks", lambda: (__import__("src.tasks"), "ok")[1])
    ok &= check("actor", lambda: (__import__("src.actor"), "ok")[1])

    print("Ray", flush=True)
    import ray
    ok &= check("version", lambda: ray.__version__)
    ok &= check("init", lambda: (ray.init(ignore_reinit_error=True), "connected")[1])
    ok &= check("resources", lambda: {k: v for k, v in ray.cluster_resources().items() if k in ("CPU", "GPU")})

    print("Multi-node", flush=True)
    from ray.util.placement_group import placement_group, remove_placement_group
    active_nodes = [n for n in ray.nodes() if n["Alive"]]
    print(f"  [INFO] {len(active_nodes)} active node(s) in cluster", flush=True)
    if len(active_nodes) < 2:
        print("  [SKIP] strict_spread: need 2 nodes, only 1 available", flush=True)
    else:
        @ray.remote
        def get_node_id():
            return ray.get_runtime_context().get_node_id()

        def test_strict_spread():
            pg = placement_group([{"CPU": 1}, {"CPU": 1}], strategy="STRICT_SPREAD")
            ray.get(pg.ready(), timeout=30)
            ids = ray.get([
                get_node_id.options(placement_group=pg, placement_group_bundle_index=0).remote(),
                get_node_id.options(placement_group=pg, placement_group_bundle_index=1).remote(),
            ])
            remove_placement_group(pg)
            if ids[0] == ids[1]:
                raise RuntimeError(f"both actors landed on the same node")
            return f"actors on 2 different nodes"

        ok &= check("strict_spread", test_strict_spread, timeout_sec=60)

        @ray.remote(num_gpus=1)
        class NCCLWorker:
            def get_ip(self):
                return ray.util.get_node_ip_address()

            def init_and_allreduce(self, rank, world_size, master_addr, master_port):
                import torch
                import torch.distributed as dist
                import os
                os.environ["MASTER_ADDR"] = master_addr
                os.environ["MASTER_PORT"] = str(master_port)
                os.environ.setdefault("NCCL_SOCKET_IFNAME", "ens32")
                os.environ.setdefault("GLOO_SOCKET_IFNAME", "ens32")
                os.environ.setdefault("NCCL_PROTO", "simple")
                # Send value (rank+1) so rank 0 sends 1.0 and rank 1 sends 2.0;
                # after all-reduce the sum should be 3.0 on both ranks.
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    world_size=world_size,
                    rank=rank,
                )
                t = torch.full((4,), float(rank + 1), device="cuda")
                dist.all_reduce(t)
                result = t[0].item()
                dist.destroy_process_group()
                return result

        def test_cross_node_nccl():
            pg = placement_group([{"CPU": 1, "GPU": 1}, {"CPU": 1, "GPU": 1}], strategy="STRICT_SPREAD")
            ray.get(pg.ready(), timeout=30)
            w0 = NCCLWorker.options(placement_group=pg, placement_group_bundle_index=0).remote()
            w1 = NCCLWorker.options(placement_group=pg, placement_group_bundle_index=1).remote()
            master_addr = ray.get(w0.get_ip.remote())
            master_port = 29500
            results = ray.get([
                w0.init_and_allreduce.remote(0, 2, master_addr, master_port),
                w1.init_and_allreduce.remote(1, 2, master_addr, master_port),
            ], timeout=90)
            remove_placement_group(pg)
            expected = 3.0  # 1.0 + 2.0
            if results[0] != expected or results[1] != expected:
                raise RuntimeError(f"unexpected all-reduce results: {results}, expected {expected} on both ranks")
            return f"all-reduce across 2 nodes: {results[0]} == {results[1]} == {expected}"

        ok &= check("cross_node_nccl", test_cross_node_nccl, timeout_sec=120)

    ray.shutdown()

    print(flush=True)
    print("ALL PASSED" if ok else "SOME CHECKS FAILED", flush=True)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
