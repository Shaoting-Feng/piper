"""Builds src._nccl, the Python binding for ncclWarmup.

NCCL_HOME: a built checkout of github.com/Shaoting-Feng/nccl.
"""
import os

from setuptools import Extension, setup

nccl_home = os.environ.get("NCCL_HOME")
if not nccl_home:
    raise SystemExit("set NCCL_HOME to a built checkout of github.com/Shaoting-Feng/nccl")
lib = os.path.join(nccl_home, "build", "lib", "libnccl.so.2")
if not os.path.exists(lib):
    raise SystemExit(f"{lib} not found: run `make -j src.build` in {nccl_home} first")

setup(
    ext_modules=[
        Extension(
            "src._nccl",
            sources=["src/csrc/nccl_warmup.c"],
            # The build has no libnccl.so dev symlink, so link by path.
            extra_link_args=[lib],
        )
    ]
)
