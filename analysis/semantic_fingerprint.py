#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _update_object(hasher: Any, value: Any, name: str) -> None:
    import torch

    hasher.update(name.encode("utf-8"))
    hasher.update(type(value).__name__.encode("ascii"))
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        hasher.update(str(tensor.dtype).encode("ascii"))
        hasher.update(str(tuple(tensor.shape)).encode("ascii"))
        if tensor.ndim == 0:
            hasher.update(tensor.numpy().tobytes())
            return
        for start in range(0, len(tensor), 1024):
            hasher.update(tensor[start : start + 1024].numpy().tobytes())
        return
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            _update_object(hasher, value[key], f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _update_object(hasher, item, f"{name}[{index}]")
        return
    hasher.update(repr(value).encode("utf-8"))


def semantic_digest(path: Path) -> str:
    import torch

    payload = torch.load(path, map_location="cpu")
    hasher = hashlib.sha256()
    _update_object(hasher, payload, "root")
    return hasher.hexdigest()


def _sequence_label_map(path: Path) -> dict[str, set[int]]:
    import torch

    payload = torch.load(path, map_location="cpu")
    inputs = payload["input_ids"].detach().cpu()
    masks = payload.get("attention_mask")
    if masks is not None:
        masks = masks.detach().cpu()
    labels = payload["labels"].detach().cpu()
    out: dict[str, set[int]] = {}
    for index in range(len(inputs)):
        row = inputs[index]
        if masks is not None:
            row = row[masks[index].bool()]
        digest = hashlib.blake2b(row.contiguous().numpy().tobytes(), digest_size=16).hexdigest()
        out.setdefault(digest, set()).add(int(labels[index].item()))
    return out


def _overlap_report(split_maps: dict[str, dict[str, set[int]]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    names = ["train", "val", "test"]
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = set(split_maps[left]) & set(split_maps[right])
            conflicts = 0
            for digest in shared:
                if split_maps[left][digest] != split_maps[right][digest]:
                    conflicts += 1
            report[f"{left}_{right}"] = {
                "overlap_count": len(shared),
                "conflicting_label_count": conflicts,
            }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute serialization-independent LRA fingerprints.")
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--tasks", default="listops,text,pathfinder")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    result: dict[str, Any] = {"version": 1, "roots": {}, "tasks": {}}
    for root in args.roots:
        result["roots"][str(root)] = {}
        for task in tasks:
            task_result: dict[str, Any] = {"splits": {}}
            split_maps: dict[str, dict[str, set[int]]] = {}
            for split in ["train", "val", "test"]:
                path = root / task / f"{split}.pt"
                digest = semantic_digest(path)
                task_result["splits"][split] = {
                    "path": str(path),
                    "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "semantic_sha256": digest,
                }
                split_maps[split] = _sequence_label_map(path)
            task_result["cross_split_overlap"] = _overlap_report(split_maps)
            result["roots"][str(root)][task] = task_result

    for task in tasks:
        semantic_sets = {
            split: sorted(
                {
                    result["roots"][str(root)][task]["splits"][split]["semantic_sha256"]
                    for root in args.roots
                }
            )
            for split in ["train", "val", "test"]
        }
        result["tasks"][task] = {
            "semantic_sha256": semantic_sets,
            "semantic_match_across_roots": all(len(values) == 1 for values in semantic_sets.values()),
            "accepted_serialized_fingerprints": [],
            "cross_split_overlap": result["roots"][str(args.roots[0])][task]["cross_split_overlap"],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
