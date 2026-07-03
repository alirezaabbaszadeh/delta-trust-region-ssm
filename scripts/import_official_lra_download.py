from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

TASKS = ("listops", "text", "pathfinder")
SPLITS = ("train", "val", "test")

TASK_ALIASES = {
    "listops": ["listops"],
    "text": ["text", "imdb", "imdb_lra"],
    "pathfinder": ["pathfinder", "pathfinder_lra"],
}

SPLIT_ALIASES = {
    "train": ["train"],
    "val": ["val", "valid", "validation", "dev"],
    "test": ["test"],
}


# Pickle compatibility stubs for common community dumps.
class ListOpsDataset:
    pass


class IMDbDataset:
    pass


class PathfinderDataset:
    pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Import official LRA download into canonical raw split files + manifest.")
    p.add_argument("--input", required=True, help="File or directory under data/raw/lra_official/downloads")
    p.add_argument("--out-root", default="data/raw/lra_official")
    p.add_argument("--manifest", default="configs/datasets/lra_official_manifest.yaml")
    p.add_argument("--task-map", choices=["auto", "manual"], default="auto")
    p.add_argument("--manual-map-json", default="", help="JSON mapping used when --task-map=manual")
    p.add_argument("--strict", action="store_true", default=False)
    p.add_argument("--overwrite", action="store_true", default=False)
    return p.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw if isinstance(raw, dict) else {}


def _counter_to_hist(labels: list[int]) -> dict[str, int]:
    c = Counter(int(v) for v in labels)
    return {str(k): int(v) for k, v in sorted(c.items())}


def _to_label_list(labels: Any) -> list[int]:
    if isinstance(labels, torch.Tensor):
        return [int(v) for v in labels.detach().cpu().tolist()]
    return [int(v) for v in labels]


def _tokens_to_rows(tokens: Any) -> list[list[int]]:
    if isinstance(tokens, torch.Tensor):
        t = tokens.detach().cpu()
        if t.ndim != 2:
            raise ValueError(f"Expected 2D token tensor, got shape={tuple(t.shape)}")
        return [[int(v) for v in row] for row in t.tolist()]

    rows: list[list[int]] = []
    for row in tokens:
        if isinstance(row, torch.Tensor):
            rows.append([int(v) for v in row.detach().cpu().tolist()])
        else:
            rows.append([int(v) for v in row])
    return rows


def _extract_split_payload(payload: Any) -> tuple[list[list[int]], list[int]]:
    # Dict-style split payloads.
    if isinstance(payload, dict):
        tokens = payload.get("input_ids")
        if tokens is None:
            tokens = payload.get("tokens")
        labels = payload.get("labels")
        if tokens is not None and labels is not None:
            rows = _tokens_to_rows(tokens)
            labs = _to_label_list(labels)
            if len(rows) != len(labs):
                raise ValueError("tokens/labels size mismatch in split payload")
            return rows, labs

        data = payload.get("data")
        if isinstance(data, list):
            rows = [[int(v) for v in item[0]] for item in data]
            labs = [int(item[1]) for item in data]
            return rows, labs

    # Object-style split payloads used by some dumps.
    if hasattr(payload, "data"):
        rows: list[list[int]] = []
        labs: list[int] = []
        for tokens, label in payload.data:
            rows.append([int(v) for v in tokens])
            labs.append(int(label))
        return rows, labs

    if hasattr(payload, "tokens") and hasattr(payload, "labels"):
        rows = _tokens_to_rows(getattr(payload, "tokens"))
        labs = _to_label_list(getattr(payload, "labels"))
        if len(rows) != len(labs):
            raise ValueError("tokens/labels size mismatch in object payload")
        return rows, labs

    # List of row dicts.
    if isinstance(payload, list):
        rows: list[list[int]] = []
        labs: list[int] = []
        for item in payload:
            if isinstance(item, dict):
                tokens = item.get("input_ids", item.get("tokens"))
                label = item.get("label", item.get("labels"))
                if tokens is None or label is None:
                    raise ValueError("list payload row missing tokens/label")
                rows.append([int(v) for v in tokens])
                labs.append(int(label))
            else:
                raise ValueError(f"Unsupported list row type: {type(item)!r}")
        return rows, labs

    raise ValueError(f"Unsupported split payload type: {type(payload)!r}")


def _find_file_by_names(root: Path, names: list[str]) -> Path | None:
    # Prefer top-level matches for determinism.
    for name in names:
        p = root / name
        if p.exists() and p.is_file():
            return p

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name.lower() in {n.lower() for n in names}:
            return p
    return None


def _find_split_file(root: Path, task: str, split: str) -> Path | None:
    task_names = TASK_ALIASES[task]
    split_names = SPLIT_ALIASES[split]

    candidates: list[str] = []
    for t in task_names:
        for s in split_names:
            candidates.extend(
                [
                    f"{t}/{s}.pt",
                    f"{t}_{s}.pt",
                    f"{s}_{t}.pt",
                    f"{t}-{s}.pt",
                    f"{s}-{t}.pt",
                ]
            )
    # Flatten paths into name hints for recursive matcher.
    direct = []
    leaf_names = []
    for c in candidates:
        if "/" in c:
            direct.append(c)
            leaf_names.append(Path(c).name)
        else:
            leaf_names.append(c)

    for rel in direct:
        p = root / rel
        if p.exists() and p.is_file():
            return p

    return _find_file_by_names(root, leaf_names)


def _detect_split_ready_dir(root: Path) -> dict[str, dict[str, Path]] | None:
    out: dict[str, dict[str, Path]] = {}
    for task in TASKS:
        out[task] = {}
        for split in SPLITS:
            p = _find_split_file(root, task, split)
            if p is None:
                return None
            out[task][split] = p
    return out


def _detect_merged_task_files(root: Path) -> dict[str, Path] | None:
    names = {
        "listops": ["listops.pt"],
        "text": ["imdb_lra.pt", "imdb.pt", "text.pt"],
        "pathfinder": ["pathfinder_lra.pt", "pathfinder.pt"],
    }
    out: dict[str, Path] = {}
    for task in TASKS:
        p = _find_file_by_names(root, names[task])
        if p is None:
            return None
        out[task] = p
    return out


def _extract_archive(path: Path, temp_dir: Path) -> Path:
    lower = path.name.lower()

    # torch.save files are often ZIP containers; prefer loading as torch payload first.
    try:
        _ = torch.load(path, map_location="cpu")
        return path
    except Exception:
        pass

    if zipfile.is_zipfile(path):
        extract_root = temp_dir / "zip_extracted"
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(extract_root)
        return extract_root

    if tarfile.is_tarfile(path):
        extract_root = temp_dir / "tar_extracted"
        extract_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "r:*") as tf:
            tf.extractall(extract_root)
        return extract_root

    if lower.endswith(".gz"):
        out_name = path.name[:-3] or "decompressed"
        out_path = temp_dir / out_name
        with gzip.open(path, "rb") as src, out_path.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
        return out_path

    return path


def _load_torch_file(path: Path) -> Any:
    return torch.load(path, map_location="cpu")


def _extract_task_from_any_payload(payload: Any, task: str) -> dict[str, Any] | None:
    # Case A: payload itself is split map {train,val,test}.
    if isinstance(payload, dict) and all(k in payload for k in SPLITS):
        return {s: payload[s] for s in SPLITS}

    if isinstance(payload, dict):
        # Case B: payload has task namespaces.
        aliases = TASK_ALIASES[task]
        for alias in aliases:
            sub = payload.get(alias)
            if isinstance(sub, dict) and all(k in sub for k in SPLITS):
                return {s: sub[s] for s in SPLITS}

        # Case C: nested under datasets/task/data fields.
        for key in ("tasks", "datasets", "data"):
            sub = payload.get(key)
            if isinstance(sub, dict):
                for alias in TASK_ALIASES[task]:
                    tsub = sub.get(alias)
                    if isinstance(tsub, dict) and all(k in tsub for k in SPLITS):
                        return {s: tsub[s] for s in SPLITS}

    return None


def _save_task_splits(
    *,
    task: str,
    split_payloads: dict[str, Any],
    out_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    task_dir = out_root / task
    task_dir.mkdir(parents=True, exist_ok=True)

    task_report = {"task": task, "splits": {}}
    for split in SPLITS:
        split_payload = split_payloads[split]
        rows, labels = _extract_split_payload(split_payload)

        out_path = task_dir / f"{split}.pt"
        if out_path.exists() and not overwrite:
            existing = torch.load(out_path, map_location="cpu")
            ex_rows, ex_labels = _extract_split_payload(existing)
            rows = ex_rows
            labels = ex_labels
        else:
            torch.save({"input_ids": rows, "labels": labels}, out_path)

        task_report["splits"][split] = {
            "path": str(out_path),
            "sha256": sha256_file(out_path),
            "num_samples": int(len(labels)),
            "label_hist": _counter_to_hist(labels),
        }

    return task_report


def _import_from_split_map(
    split_map: dict[str, dict[str, Path]],
    *,
    out_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {"mode": "split_ready", "tasks": {}}
    for task in TASKS:
        split_payloads = {split: _load_torch_file(split_map[task][split]) for split in SPLITS}
        report["tasks"][task] = _save_task_splits(
            task=task,
            split_payloads=split_payloads,
            out_root=out_root,
            overwrite=overwrite,
        )
    return report


def _import_from_merged_files(
    merged_files: dict[str, Path],
    *,
    out_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {"mode": "merged_task_files", "tasks": {}}
    for task in TASKS:
        payload = _load_torch_file(merged_files[task])
        splits = _extract_task_from_any_payload(payload, task)
        if splits is None:
            raise ValueError(f"Could not extract splits for task={task} from {merged_files[task]}")
        report["tasks"][task] = _save_task_splits(
            task=task,
            split_payloads=splits,
            out_root=out_root,
            overwrite=overwrite,
        )
    return report


def _import_from_combined_payload(
    payload: Any,
    *,
    out_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {"mode": "combined_bundle", "tasks": {}}
    for task in TASKS:
        splits = _extract_task_from_any_payload(payload, task)
        if splits is None:
            raise ValueError(f"Could not extract task={task} from combined payload")
        report["tasks"][task] = _save_task_splits(
            task=task,
            split_payloads=splits,
            out_root=out_root,
            overwrite=overwrite,
        )
    return report


def _auto_import(input_path: Path, *, out_root: Path, overwrite: bool, strict: bool) -> dict[str, Any]:
    if input_path.is_dir():
        split_map = _detect_split_ready_dir(input_path)
        if split_map is not None:
            return _import_from_split_map(split_map, out_root=out_root, overwrite=overwrite)

        merged = _detect_merged_task_files(input_path)
        if merged is not None:
            return _import_from_merged_files(merged, out_root=out_root, overwrite=overwrite)

        # Try known bundle file names inside directory.
        for name in ("lra_release", "lra_release.pt", "lra_release.gz", "lra_release.tar", "lra_release.zip"):
            candidate = input_path / name
            if candidate.exists() and candidate.is_file():
                return _auto_import(candidate, out_root=out_root, overwrite=overwrite, strict=strict)

        if strict:
            raise ValueError(f"Could not auto-detect split-ready or merged inputs in directory: {input_path}")
        raise SystemExit(f"No recognized dataset layout in: {input_path}")

    with tempfile.TemporaryDirectory(prefix="lra_import_") as tmp:
        extracted = _extract_archive(input_path, Path(tmp))
        if extracted != input_path:
            return _auto_import(extracted, out_root=out_root, overwrite=overwrite, strict=strict)

        payload = _load_torch_file(input_path)
        return _import_from_combined_payload(payload, out_root=out_root, overwrite=overwrite)


def _manual_import(
    *,
    manual_map_json: Path,
    out_root: Path,
    overwrite: bool,
    strict: bool,
) -> dict[str, Any]:
    raw = json.loads(manual_map_json.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("manual-map-json must be a JSON object")

    report: dict[str, Any] = {"mode": "manual", "tasks": {}}
    for task in TASKS:
        if task not in raw:
            raise ValueError(f"manual-map-json missing key: {task}")

        value = raw[task]
        if isinstance(value, str):
            src_path = Path(value)
        elif isinstance(value, dict) and "path" in value:
            src_path = Path(str(value["path"]))
        else:
            raise ValueError(f"manual-map-json entry for task={task} must be string or object with path")

        if not src_path.is_absolute():
            src_path = (Path.cwd() / src_path).resolve()

        # Reuse auto logic per task source then keep only this task.
        if src_path.is_dir():
            split_map = _detect_split_ready_dir(src_path)
            if split_map is not None:
                split_payloads = {split: _load_torch_file(split_map[task][split]) for split in SPLITS}
                report["tasks"][task] = _save_task_splits(
                    task=task,
                    split_payloads=split_payloads,
                    out_root=out_root,
                    overwrite=overwrite,
                )
                continue

        extracted_input = src_path
        with tempfile.TemporaryDirectory(prefix="lra_import_manual_") as tmp:
            extracted = _extract_archive(src_path, Path(tmp))
            if extracted != src_path:
                extracted_input = extracted

            if extracted_input.is_dir():
                split_map = _detect_split_ready_dir(extracted_input)
                if split_map is not None:
                    split_payloads = {split: _load_torch_file(split_map[task][split]) for split in SPLITS}
                    report["tasks"][task] = _save_task_splits(
                        task=task,
                        split_payloads=split_payloads,
                        out_root=out_root,
                        overwrite=overwrite,
                    )
                    continue

                merged = _detect_merged_task_files(extracted_input)
                if merged is not None:
                    payload = _load_torch_file(merged[task])
                else:
                    # try combined bundle in directory
                    bundle = None
                    for name in ("lra_release", "lra_release.pt"):
                        c = extracted_input / name
                        if c.exists() and c.is_file():
                            bundle = c
                            break
                    if bundle is None:
                        if strict:
                            raise ValueError(f"Could not parse manual source for task={task}: {src_path}")
                        raise SystemExit(f"Unsupported manual source for task={task}: {src_path}")
                    payload = _load_torch_file(bundle)
            else:
                payload = _load_torch_file(extracted_input)

            splits = _extract_task_from_any_payload(payload, task)
            if splits is None:
                raise ValueError(f"Could not extract task={task} from manual source: {src_path}")
            report["tasks"][task] = _save_task_splits(
                task=task,
                split_payloads=splits,
                out_root=out_root,
                overwrite=overwrite,
            )

    return report


def _update_manifest(
    *,
    manifest_path: Path,
    out_root: Path,
    import_report: dict[str, Any],
    input_path: Path,
) -> dict[str, Any]:
    manifest = _load_yaml(manifest_path)
    if not manifest:
        manifest = {
            "manifest_version": 1,
            "preprocess_version": "v1",
            "defaults": {"pad_token_id": 0, "truncation_side": "right"},
            "tasks": {},
        }

    source = manifest.setdefault("source", {})
    source["name"] = "Long Range Arena official release import"
    source["reference_url"] = "https://storage.googleapis.com/long-range-arena/lra_release"
    source["official_reference"] = "https://github.com/google-research/long-range-arena"
    source["imported_from"] = str(input_path)
    source["imported_at_utc"] = utc_now_iso()

    tasks_obj = manifest.setdefault("tasks", {})
    for task in TASKS:
        task_obj = tasks_obj.setdefault(task, {})
        task_obj.setdefault("num_classes", 2)
        task_obj.setdefault("vocab_size", 32000)

        expected = task_obj.setdefault("expected_splits", {})
        splits = task_obj.setdefault("splits", {})

        task_rep = import_report["tasks"][task]
        for split in SPLITS:
            split_rep = task_rep["splits"][split]
            split_path = Path(split_rep["path"])
            if split_path.is_absolute():
                try:
                    rel_path = split_path.relative_to(Path.cwd())
                    split_path_str = rel_path.as_posix()
                except Exception:
                    split_path_str = split_path.as_posix()
            else:
                split_path_str = split_path.as_posix()

            splits[split] = {
                "path": split_path_str,
                "url": "",
                "format": "pt",
                "sha256": split_rep["sha256"],
            }
            expected[split] = {
                "num_samples": split_rep["num_samples"],
                "label_hist": split_rep["label_hist"],
            }

    _save_yaml(manifest_path, manifest)
    return manifest


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"--input does not exist: {input_path}")

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (Path.cwd() / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (Path.cwd() / manifest_path).resolve()

    if args.task_map == "manual":
        if not args.manual_map_json:
            raise ValueError("--manual-map-json is required when --task-map=manual")
        manual_map = Path(args.manual_map_json)
        if not manual_map.is_absolute():
            manual_map = (Path.cwd() / manual_map).resolve()
        if not manual_map.exists():
            raise FileNotFoundError(f"manual-map-json does not exist: {manual_map}")

        import_report = _manual_import(
            manual_map_json=manual_map,
            out_root=out_root,
            overwrite=bool(args.overwrite),
            strict=bool(args.strict),
        )
    else:
        import_report = _auto_import(
            input_path,
            out_root=out_root,
            overwrite=bool(args.overwrite),
            strict=bool(args.strict),
        )

    manifest = _update_manifest(
        manifest_path=manifest_path,
        out_root=out_root,
        import_report=import_report,
        input_path=input_path,
    )

    out = {
        "input": str(input_path),
        "out_root": str(out_root),
        "manifest": str(manifest_path),
        "import_mode": import_report.get("mode", "unknown"),
        "source": manifest.get("source", {}),
        "tasks": import_report.get("tasks", {}),
    }

    report_path = out_root / "import_report.json"
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "report": str(report_path), "mode": out["import_mode"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
