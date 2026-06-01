import json
import os
from typing import Any


def load_schedule_directives(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        directives = json.load(f)
    if not isinstance(directives, list):
        raise ValueError(f"schedule directives file must contain a list, got: {type(directives)}")

    for directive in directives:
        if not isinstance(directive, dict):
            continue
        if "filter" in directive:
            directive["filter"] = _normalize_filter_list(directive["filter"])
        if "filters" in directive and isinstance(directive["filters"], list):
            directive["filters"] = _normalize_order_filters(directive["filters"])
        if directive.get("op") == "split" and directive.get("num_microbatches") == "__MBS__":
            raise ValueError(
                "split.num_microbatches must be encoded in the schedule JSON; "
                "'__MBS__' is no longer supported"
            )
    return directives


def load_schedule_info(path: str) -> dict[str, Any]:
    return derive_schedule_info(load_schedule_directives(path), path)


def derive_schedule_info(directives: list[dict], schedule_path: str) -> dict[str, Any]:
    pp_to_devices: dict[int, list[int]] = {}
    num_microbatches = None

    for directive in directives:
        if not isinstance(directive, dict):
            continue
        op = directive.get("op")
        if op == "place":
            pp_idx = _filter_value(directive.get("filter"), "PP")
            devices = directive.get("devices", directive.get("device"))
            if pp_idx is None or not isinstance(devices, list) or not devices:
                raise ValueError(f"place directive must include PP filter and non-empty devices: {directive}")
            pp_to_devices[int(pp_idx)] = [int(d) for d in devices]
        elif op == "split":
            n = int(directive.get("num_microbatches", 0))
            if n <= 0:
                raise ValueError(f"split directive requires num_microbatches > 0: {directive}")
            if num_microbatches is not None and num_microbatches != n:
                raise ValueError(
                    f"multiple split directives disagree on num_microbatches: "
                    f"{num_microbatches} vs {n}"
                )
            num_microbatches = n

    if not pp_to_devices:
        raise ValueError("schedule JSON must include place directives with PP filters")
    pp_indices = sorted(pp_to_devices)
    expected_pp = list(range(len(pp_indices)))
    if pp_indices != expected_pp:
        raise ValueError(f"PP indices must be contiguous from 0, got {pp_indices}")
    device_counts = {len(devices) for devices in pp_to_devices.values()}
    if len(device_counts) != 1:
        raise ValueError(f"all PP place directives must use the same device count, got {pp_to_devices}")
    device_keys = sorted({tuple(devices) for devices in pp_to_devices.values()})
    if num_microbatches is None:
        raise ValueError("schedule JSON must include a split directive with num_microbatches")

    return {
        "name": os.path.splitext(os.path.basename(schedule_path))[0],
        "path": schedule_path,
        "num_stages": len(pp_indices),
        "pp_degree": len(device_keys),
        "dp_degree": next(iter(device_counts)),
        "num_microbatches": num_microbatches,
    }


def _normalize_filter_list(spec):
    # JSON encodes tuples as lists; the schedule normalizer expects list[tuple[tag, value]].
    if isinstance(spec, list):
        out = []
        for item in spec:
            if (
                isinstance(item, list)
                and len(item) == 2
                and isinstance(item[0], str)
            ):
                out.append((item[0], item[1]))
            else:
                out.append(item)
        return out
    return spec


def _is_filter_list(spec):
    return isinstance(spec, list) and all(
        isinstance(item, list)
        and len(item) == 2
        and isinstance(item[0], str)
        for item in spec
    )


def _normalize_order_filters(filters):
    normalized = []
    for item in filters:
        if _is_filter_list(item):
            normalized.append(_normalize_filter_list(item))
        elif isinstance(item, list):
            normalized.append([
                _normalize_filter_list(f) if _is_filter_list(f) else f
                for f in item
            ])
        else:
            normalized.append(item)
    return normalized


def _filter_value(filter_spec, key: str):
    if isinstance(filter_spec, dict):
        return filter_spec.get(key)
    if isinstance(filter_spec, list):
        for item in filter_spec:
            if isinstance(item, tuple) and len(item) == 2 and item[0] == key:
                return item[1]
    return None
