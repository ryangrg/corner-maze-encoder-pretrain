#!/usr/bin/env python3
"""
apply_regroupings.py

Load a dataset_io bundle plus a regrouped CSV (each line is a comma-delimited set
of labels that should be merged) and rewrite the dataset so the grouped tensors
share the same label IDs. The bundle is expected to follow the lean payload format
produced by create_dataset.group_duplicates.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch

import dataset_io

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT_DIR / "data/datasets/corner-maze-render-base-images-ds"
DEFAULT_REGROUP_CSV = ROOT_DIR / "data/csv/corner-maze-render-base-images-duplicate-groups-partitioned-52-1-duplicate-groups-52-1-manual-check-acute.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_DIR.parent / f"corner-maze-render-base-images-regrouped-acute-ds"


def _load_regroup_csv(csv_path: Path) -> List[List[str]]:
    """
    Parse regrouped CSV file into a list of label-name groups.
    """
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Regroup CSV not found: {csv_path}")

    groups: List[List[str]] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            tokens = [token.strip() for token in line.strip().split(",") if token.strip()]
            if tokens:
                groups.append(tokens)

    if not groups:
        raise RuntimeError(f"No regrouping rows found in {csv_path}")

    return groups


def _coerce_label_names(payload: Dict[str, Any], size: int) -> List[str]:
    """
    Extract label names from payload, falling back to numeric labels if needed.
    """
    def _convert(values: Iterable) -> List[str]:
        seq = list(values)
        if len(seq) != size:
            raise ValueError("Label name sequence length does not match tensor length.")
        return [str(item) for item in seq]

    names = payload.get("label_names")
    if isinstance(names, torch.Tensor):
        return _convert(names.detach().cpu().tolist())
    if isinstance(names, Sequence):
        return _convert(names)

    labels = payload.get("labels")
    if isinstance(labels, torch.Tensor):
        return [str(int(v)) for v in labels.detach().cpu().tolist()]
    if isinstance(labels, Sequence):
        return [str(item) for item in labels]

    raise KeyError("Dataset payload missing label_names/labels for regrouping.")


def apply_regroupings(
    bundle_dir: Path,
    regroup_csv: Path,
    *,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Update dataset bundle to respect CSV-based regroupings.
    """
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    if bundle_dir.is_file():
        raise ValueError("bundle_dir must point to a dataset directory created by dataset_io.")

    payload = dataset_io.load_bundle(bundle_dir)
    stack = payload.get("x")
    if not isinstance(stack, torch.Tensor):
        raise TypeError("Dataset payload missing tensor 'x'.")
    size = stack.shape[0]

    label_names = _coerce_label_names(payload, size)
    label_to_index = {label: idx for idx, label in enumerate(label_names)}

    groups = _load_regroup_csv(regroup_csv)

    missing_labels = sorted({label for group in groups for label in group if label not in label_to_index})
    if missing_labels:
        raise KeyError(
            f"{len(missing_labels)} labels from regroup CSV were not found in the dataset. "
            f"Examples: {missing_labels[:5]}"
        )

    raw_labels = payload.get("labels")
    if raw_labels is None:
        label_ids = list(range(size))
    elif isinstance(raw_labels, torch.Tensor):
        label_ids = [int(v) for v in raw_labels.detach().cpu().tolist()]
    else:
        label_ids = [int(v) for v in raw_labels]

    updated = False
    for group in groups:
        indices = [label_to_index[label] for label in group]
        canonical = min(label_ids[idx] for idx in indices)
        for idx in indices:
            if label_ids[idx] != canonical:
                label_ids[idx] = canonical
                updated = True

    catalog_builder: Dict[int, set] = defaultdict(set)

    temp_builder: Dict[int, set] = defaultdict(set)
    for label_id, name in zip(label_ids, label_names):
        temp_builder[int(label_id)].add(str(name))
    labels2label_names = {
        label_id: sorted(descriptions)
        for label_id, descriptions in temp_builder.items()
    }

    y_tensor = torch.tensor(label_ids, dtype=torch.long, device=stack.device)
    payload["labels"] = label_ids
    payload["labels2label_names"] = labels2label_names
    payload["y"] = y_tensor
    payload.setdefault("label_names", label_names)
    meta = dict(payload.get("meta", {}))
    meta.update(
        {
            "stack_len": size,
            "label_names_count": len(label_names),
            "regrouping": {
                "csv": str(Path(regroup_csv).resolve()),
                "groups": len(groups),
                "labels_changed": int(updated),
            },
        }
    )
    payload["meta"] = meta

    destination = output_dir if output_dir is not None else bundle_dir
    destination = Path(destination).expanduser().resolve()

    if destination.exists() and not overwrite and destination != bundle_dir:
        raise FileExistsError(f"Destination {destination} already exists (set overwrite=True to replace).")

    destination.mkdir(parents=True, exist_ok=True)
    dataset_io.save_bundle(payload, destination)

    csv_source = Path(regroup_csv).expanduser().resolve()
    csv_name = destination.name[:-3] if destination.name.endswith("-ds") else destination.name
    csv_target = destination / f"{csv_name}.csv"
    csv_target.write_text(csv_source.read_text())

    return {
        "output_path": destination,
        "groups_processed": len(groups),
        "labels_changed": int(updated),
    }


def main() -> None:
    result = apply_regroupings(
        DEFAULT_DATASET_DIR,
        DEFAULT_REGROUP_CSV,
        output_dir=DEFAULT_OUTPUT_DIR,
        overwrite=True,
    )
    print(
        f"Regrouped dataset saved to {result['output_path']} "
        f"(groups={result['groups_processed']}, labels_changed={result['labels_changed']})."
    )


if __name__ == "__main__":
    main()
