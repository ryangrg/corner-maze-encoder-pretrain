#!/usr/bin/env python3
"""
Test helper for generating a deduplicated dataset bundle.

Rather than reporting duplicate groupings, this script writes a new `.pt` file
mirroring the structure produced by `create_dataset.py` but with every duplicate
sample removed (keeping the first occurrence in each group).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Union

from src.find_duplicate_pairs import BUNDLE_PATH as DEFAULT_BUNDLE_PATH
from src.remove_duplicates import deduplicate_dataset


def deduplicate_bundle(
    bundle_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    report_path: Union[str, Path, None] = None,
    overwrite: bool = False,
) -> Dict[str, Union[str, int, None]]:
    """
    Generate a deduplicated dataset bundle (keeping the first occurrence of each label_id)
    and optionally write a JSON report describing the removed entries.
    """
    result = deduplicate_dataset(
        bundle_path,
        output_path,
        overwrite=overwrite,
    )

    duplicate_groups = result.get("duplicate_groups") or []
    if report_path and duplicate_groups:
        report_path = Path(report_path).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "duplicate_groups": duplicate_groups,
            "plan": result.get("plan"),
        }
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        result["report_path"] = str(report_path)
    else:
        result["report_path"] = None

    return result


def parse_args(argv: Union[None, list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deduplicated dataset bundle suitable for embedding or feature extraction tests.",
    )
    parser.add_argument(
        "--bundle",
        default=DEFAULT_BUNDLE_PATH,
        help="Path to the input dataset bundle (.pt). Defaults to the bundle configured in src.find_duplicate_pairs.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination path for the deduplicated dataset bundle (.pt).",
    )
    parser.add_argument(
        "--report",
        help="Optional JSON file capturing which entries were removed and their mapping to the kept sample.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output or report files.",
    )
    return parser.parse_args(argv)


def main(argv: Union[None, list[str]] = None) -> None:
    args = parse_args(argv)
    result = deduplicate_bundle(
        bundle_path=args.bundle,
        output_path=args.output,
        report_path=args.report,
        overwrite=args.overwrite,
    )

    if result["removed"] == 0:
        print("No duplicates found; no files were written.")
    else:
        print(
            f"Deduplicated dataset written to {result['output_path']} "
            f"(kept {result['kept']} entries, removed {result['removed']})."
        )
        if result.get("report_path"):
            print(f"Duplicate mapping saved to {result['report_path']}.")


if __name__ == "__main__":
    main()
