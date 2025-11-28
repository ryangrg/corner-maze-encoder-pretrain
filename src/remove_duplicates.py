#!/usr/bin/env python3
"""
remove_duplicates.py

Generate a deduplicated .pt dataset bundle (matching the structure produced by
create_dataset.py) while keeping only the first sample from each duplicate set.
Additionally, emit a CSV file listing the duplicate label groups so downstream
pipelines can align feature vectors or embeddings.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch

# =======================
# === USER SETTINGS ====
# =======================

# Input dataset bundle to inspect (defaults to the same path used by find_duplicate_pairs.py).
BUNDLE_PATH: Union[str, Path, None] = Path(
    "/Users/ryangrgurich/VS Code Local/corner-maze-encoder-pretrain/data/pt-files/corner-maze-render-base-images-duplicate-groups-52-1.pt"
)

# Destination for the deduplicated .pt bundle.
DEDUP_OUTPUT_PATH: Union[str, Path] = Path(
    "/Users/ryangrgurich/VS Code Local/corner-maze-encoder-pretrain/data/pt-files/dedup-corner-maze-render-base-images-dataset-52-1.pt"
)

# CSV export path (defaults to sharing the deduplicated bundle's stem if left as None).
DUPLICATES_CSV_PATH: Union[str, Path, None] = None
OVERWRITE_OUTPUTS: bool = True


def load_dataset(
    bundle_path: Union[str, Path], map_location: str = "cpu"
) -> Tuple[Dict[str, Any], torch.Tensor, List[int], List[str], Dict[Union[str, int], Any]]:
    """
    Load the .pt dataset bundle and return payload plus label metadata.
    """
    path = Path(bundle_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset bundle not found: {path}")

    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload, received {type(payload)!r}")

    if "x" not in payload:
        raise KeyError("Dataset payload missing 'x' tensor")

    tensor = payload["x"]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected 'x' to be torch.Tensor, received {type(tensor)!r}")

    label_catalog = payload.get("label_catalog", {})

    def _catalog_lookup(label_id: int, fallback: str) -> str:
        entry = label_catalog.get(label_id) or label_catalog.get(str(label_id))
        if isinstance(entry, dict):
            descriptions = entry.get("descriptions")
            if descriptions:
                return str(descriptions[0])
        return fallback

    raw_labels = payload.get("labels")
    if raw_labels is None:
        if "y" in payload:
            labels_raw_iter = payload["y"]
            if isinstance(labels_raw_iter, torch.Tensor):
                label_ids = [int(v) for v in labels_raw_iter.detach().cpu().tolist()]
            else:
                label_ids = [int(v) for v in labels_raw_iter]
        else:
            label_ids = list(range(tensor.shape[0]))
    elif isinstance(raw_labels, torch.Tensor):
        label_ids = [int(v) for v in raw_labels.detach().cpu().tolist()]
    else:
        label_ids = [int(v) for v in raw_labels]

    if len(label_ids) != tensor.shape[0]:
        raise ValueError(
            f"Label id count ({len(label_ids)}) does not match tensor length ({tensor.shape[0]})."
        )

    label_names_raw = payload.get("label_names")
    if label_names_raw is not None and len(label_names_raw) == tensor.shape[0]:
        label_names = [str(name) for name in label_names_raw]
    else:
        label_names = [_catalog_lookup(label_id, f"label_{label_id}") for label_id in label_ids]

    return payload, tensor, label_ids, label_names, label_catalog


def group_indices_by_label(
    label_ids: Sequence[int],
) -> Dict[int, List[int]]:
    """
    Group dataset indices by their integer label ID.
    """
    groups: Dict[int, List[int]] = {}
    for idx, label_id in enumerate(label_ids):
        bucket = groups.setdefault(int(label_id), [])
        bucket.append(idx)
    return groups


def _build_deduplication_plan(
    dataset_size: int,
    labels: Sequence[str],
    duplicate_index_groups: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    """
    Build bookkeeping describing which entries are kept vs removed when deduplicating.
    """
    removed_to_kept: Dict[int, int] = {}
    for group in duplicate_index_groups:
        if not group:
            continue
        sorted_group = sorted(group)
        keep_idx = sorted_group[0]
        for idx in sorted_group[1:]:
            removed_to_kept[idx] = keep_idx

    removed_indices = set(removed_to_kept)
    keep_indices = [idx for idx in range(dataset_size) if idx not in removed_indices]
    original_to_new = {orig_idx: new_idx for new_idx, orig_idx in enumerate(keep_indices)}

    removed_entries = [
        {
            "removed_index": int(removed_idx),
            "removed_label": labels[removed_idx],
            "kept_original_index": int(kept_idx),
            "kept_label": labels[kept_idx],
            "kept_new_index": int(original_to_new[kept_idx]),
        }
        for removed_idx, kept_idx in sorted(removed_to_kept.items())
    ]

    kept_entries = [
        {
            "original_index": int(orig_idx),
            "new_index": int(new_idx),
            "label": labels[orig_idx],
        }
        for orig_idx, new_idx in sorted(original_to_new.items(), key=lambda item: item[1])
    ]

    group_details = []
    for group in duplicate_index_groups:
        if not group:
            continue
        sorted_group = sorted(group)
        keep_idx = sorted_group[0]
        group_details.append(
            {
                "kept_original_index": int(keep_idx),
                "kept_new_index": int(original_to_new[keep_idx]),
                "kept_label": labels[keep_idx],
                "members": [
                    {
                        "original_index": int(idx),
                        "label": labels[idx],
                        "is_kept": idx == keep_idx,
                        "new_index": int(original_to_new[idx]) if idx in original_to_new else None,
                    }
                    for idx in sorted_group
                ],
            }
        )

    return {
        "keep_indices": [int(idx) for idx in keep_indices],
        "kept_entries": kept_entries,
        "removed_entries": removed_entries,
        "duplicate_groups": group_details,
    }


def deduplicate_dataset(
    bundle_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Write a new dataset bundle with duplicates removed, keeping the first occurrence in each group.
    Returns metadata describing the deduplication plan.
    """
    bundle_path = Path(bundle_path).expanduser().resolve()
    payload, stack, label_ids_list, label_names_list, label_catalog = load_dataset(bundle_path)

    label_groups_map = group_indices_by_label(label_ids_list)
    duplicate_index_groups = [
        sorted(indices) for indices in label_groups_map.values() if len(indices) > 1
    ]
    duplicate_label_groups = [
        [label_names_list[idx] for idx in indices] for indices in duplicate_index_groups
    ]
    duplicate_member_indices = {idx for group in duplicate_index_groups for idx in group}
    singleton_label_groups = [
        [label_names_list[idx]]
        for idx in range(stack.shape[0])
        if idx not in duplicate_member_indices
    ]

    if not duplicate_index_groups:
        return {
            "output_path": None,
            "removed": 0,
            "kept": stack.shape[0],
            "plan": None,
            "duplicate_groups": [],
            "singleton_groups": singleton_label_groups,
        }

    plan = _build_deduplication_plan(stack.shape[0], label_names_list, duplicate_index_groups)
    keep_indices = plan["keep_indices"]

    index_tensor = torch.tensor(keep_indices, dtype=torch.long, device=stack.device)
    dedup_stack = stack.index_select(0, index_tensor)

    label_ids_kept = [label_ids_list[idx] for idx in keep_indices]
    label_names_kept = [label_names_list[idx] for idx in keep_indices]

    catalog_builder: Dict[int, set] = defaultdict(set)
    catalog_meta: Dict[int, Dict[str, Any]] = defaultdict(dict)

    def _merge_metadata(target: Dict[str, Any], source: Dict[str, Any]) -> None:
        for key, value in source.items():
            if key == "descriptions":
                continue
            target.setdefault(key, value)

    def _ingest_catalog(source: Dict) -> None:
        for key, value in source.items():
            try:
                label_id = int(key)
            except (TypeError, ValueError):
                continue
            descriptions = []
            if isinstance(value, dict):
                descriptions = value.get("descriptions") or []
                _merge_metadata(catalog_meta[label_id], value)
            if isinstance(descriptions, (list, tuple, set)):
                for desc in descriptions:
                    catalog_builder[label_id].add(str(desc))

    if isinstance(label_catalog, dict):
        _ingest_catalog(label_catalog)

    for label_id, description in zip(label_ids_list, label_names_list):
        catalog_builder[int(label_id)].add(str(description))

    dedup_label_catalog = {
        label_id: {
            **catalog_meta.get(label_id, {}),
            "descriptions": sorted(descriptions),
        }
        for label_id, descriptions in catalog_builder.items()
    }

    label2idx = {
        description: label_id
        for label_id, info in dedup_label_catalog.items()
        for description in info["descriptions"]
    }
    idx2label = {
        label_id: info["descriptions"][0]
        for label_id, info in dedup_label_catalog.items()
    }

    y_tensor = torch.tensor(label_ids_kept, dtype=torch.long)

    dedup_payload = dict(payload)
    dedup_payload["x"] = dedup_stack
    dedup_payload["y"] = y_tensor
    dedup_payload["labels"] = label_ids_kept
    dedup_payload["label_names"] = label_names_kept
    dedup_payload["label_catalog"] = dedup_label_catalog
    dedup_payload["label2idx"] = label2idx
    dedup_payload["idx2label"] = idx2label
    dedup_payload["dedup_info"] = plan

    meta = dict(payload.get("meta", {}))
    meta.update(
        {
            "count": len(keep_indices),
            "deduplicated_from": str(bundle_path),
            "dedup_removed": len(plan["removed_entries"]),
            "catalog_size": len(dedup_label_catalog),
        }
    )
    meta.setdefault("label_dtype", str(y_tensor.dtype))
    meta["deduplication"] = {
        "removed": len(plan["removed_entries"]),
        "kept": len(keep_indices),
    }
    dedup_payload["meta"] = meta

    out_path = Path(output_path).expanduser().resolve()
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dedup_payload, out_path)

    return {
        "output_path": str(out_path),
        "removed": len(plan["removed_entries"]),
        "kept": len(keep_indices),
        "plan": plan,
        "duplicate_groups": duplicate_label_groups,
        "singleton_groups": singleton_label_groups,
    }


def _resolve_paths(
    bundle: Union[str, Path],
    output: Union[str, Path],
    csv_path: Union[str, Path, None],
) -> Tuple[Path, Path, Path]:
    bundle_path = Path(bundle).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if csv_path is None:
        csv_path = output_path.with_suffix(".csv")
    csv_path = Path(csv_path).expanduser().resolve()
    return bundle_path, output_path, csv_path


def _write_duplicate_csv(
    csv_path: Path,
    duplicate_groups: Sequence[Sequence[str]],
    singleton_groups: Sequence[Sequence[str]] | None = None,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        rows: List[Sequence[str]] = list(duplicate_groups)
        if singleton_groups:
            rows.extend(singleton_groups)
        if not rows:
            writer.writerow([])
            return
        for group in rows:
            writer.writerow(list(group))


def remove_duplicates(
    bundle: Union[str, Path],
    output_path: Union[str, Path],
    csv_path: Union[str, Path, None] = None,
    *,
    overwrite: bool = False,
) -> Tuple[dict, Union[Path, None]]:
    """
    Produce a deduplicated dataset bundle and accompanying CSV of duplicate label groups.
    Returns a tuple of (dedup_result, csv_path).
    """
    bundle_path, output_path, csv_target = _resolve_paths(bundle, output_path, csv_path)

    dedup_result = deduplicate_dataset(
        bundle_path,
        output_path,
        overwrite=overwrite,
    )

    duplicate_groups = dedup_result.get("duplicate_groups") or []
    singleton_groups = dedup_result.get("singleton_groups") or []
    if csv_target is not None and (duplicate_groups or singleton_groups):
        _write_duplicate_csv(csv_target, duplicate_groups, singleton_groups)
        csv_written: Union[Path, None] = csv_target
    else:
        csv_written = None

    return dedup_result, csv_written


def main() -> None:
    if BUNDLE_PATH is None:
        raise ValueError("Bundle path is required. Set BUNDLE_PATH before running this script.")
    if DEDUP_OUTPUT_PATH is None:
        raise ValueError("Output path is required. Set DEDUP_OUTPUT_PATH before running this script.")

    dedup_result, csv_path = remove_duplicates(
        bundle=BUNDLE_PATH,
        output_path=DEDUP_OUTPUT_PATH,
        csv_path=DUPLICATES_CSV_PATH,
        overwrite=OVERWRITE_OUTPUTS,
    )

    if dedup_result["removed"] == 0:
        print("No duplicates found; no files were written.")
    else:
        print(
            f"Deduplicated dataset written to {dedup_result['output_path']} "
            f"(kept {dedup_result['kept']} entries, removed {dedup_result['removed']})."
        )
        if csv_path is not None:
            print(f"Duplicate groups saved to {csv_path}.")


if __name__ == "__main__":
    main()
