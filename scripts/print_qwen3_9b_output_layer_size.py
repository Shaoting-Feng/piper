#!/usr/bin/env python3
"""Read the local Qwen3 9B config and print output-layer size in bf16."""

from __future__ import annotations

import sys
import ast
from pathlib import Path

import torch
from torch import nn


def _add_repo_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sibling_torchtitan = repo_root.parent / "torchtitan"

    for path in (repo_root, sibling_torchtitan):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


_add_repo_paths()


def _load_local_qwen3_9b_dims() -> tuple[int, int]:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "test" / "models" / "qwen3.py"
    module = ast.parse(config_path.read_text(encoding="utf-8"), filename=str(config_path))

    in_9b_case = False
    vocab_size = None
    dim = None

    for node in ast.walk(module):
        if not isinstance(node, ast.Match):
            continue
        for case in node.cases:
            pattern = case.pattern
            if isinstance(pattern, ast.MatchValue) and isinstance(pattern.value, ast.Constant):
                in_9b_case = pattern.value.value == "9B"
            else:
                in_9b_case = False
            if not in_9b_case:
                continue
            for stmt in case.body:
                if not isinstance(stmt, ast.Return) or not isinstance(stmt.value, ast.Call):
                    continue
                for kw in stmt.value.keywords:
                    if kw.arg == "vocab_size" and isinstance(kw.value, ast.Constant):
                        vocab_size = int(kw.value.value)
                    elif kw.arg == "dim" and isinstance(kw.value, ast.Constant):
                        dim = int(kw.value.value)
            break

    if vocab_size is None or dim is None:
        raise RuntimeError(f"Failed to read vocab_size/dim for 9B from {config_path}")
    return vocab_size, dim


def main() -> None:
    vocab_size, dim = _load_local_qwen3_9b_dims()
    with torch.device("meta"):
        output_layer = nn.Linear(dim, vocab_size, bias=False, dtype=torch.bfloat16)
    weight = output_layer.weight
    num_params = weight.numel()
    size_bytes = num_params * torch.tensor([], dtype=torch.bfloat16).element_size()

    print(f"model=Qwen3 9B dtype={torch.bfloat16}")
    print("source=local-config")
    print(f"output_layer_type={type(output_layer).__name__}")
    print(f"output_weight_shape={tuple(weight.shape)}")
    print(f"output_weight_numel={num_params}")
    print(f"output_weight_size_bytes={size_bytes}")
    print(f"output_weight_size_mib={size_bytes / (1024 ** 2):.2f}")


if __name__ == "__main__":
    main()
