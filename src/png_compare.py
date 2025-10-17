import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

try:
    from PIL import Image, ImageChops, ImageStat
except ImportError as exc:  # pragma: no cover - communicates dependency requirement
    raise ImportError(
        "png-compare requires Pillow. Install it with `pip install pillow`."
    ) from exc


@dataclass
class _Group:
    representative: Path
    names: List[str]


def _images_similar(
    path_a: Path,
    path_b: Path,
    per_channel_tolerance: int,
    mean_tolerance: float,
) -> bool:
    """Return True when two images match within the provided tolerances."""
    with Image.open(path_a) as img_a, Image.open(path_b) as img_b:
        rgba_a = img_a.convert("RGBA")
        rgba_b = img_b.convert("RGBA")
        if rgba_a.size != rgba_b.size:
            return False

        diff = ImageChops.difference(rgba_a, rgba_b)

        if per_channel_tolerance >= 0:
            extrema = diff.getextrema()
            if any(channel_max > per_channel_tolerance for _, channel_max in extrema):
                return False

        if mean_tolerance >= 0:
            stats = ImageStat.Stat(diff)
            mean_diff = sum(stats.mean) / len(stats.mean)
            if mean_diff > mean_tolerance:
                return False

    return True


def group_similar_pngs(
    directory: str,
    include_singletons: bool = False,
    per_channel_tolerance: int = 0,
    mean_tolerance: float = 0.0,
) -> List[List[str]]:
    """
    Group PNG files that have similar pixel data within the provided tolerances.

    Args:
        directory: Path to a folder containing PNG images.
        include_singletons: When True, include one-item groups for files with no close match.
        per_channel_tolerance: Maximum allowed absolute per-channel difference (0-255).
            Use a negative value to skip this check.
        mean_tolerance: Maximum allowed average per-channel difference across all pixels.
            Use a negative value to skip this check.

    Returns:
        A list of lists where each inner list contains the file names of PNGs deemed similar.
        Duplicate groups are always included; singleton groups depend on include_singletons.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"{directory!r} is not a directory")

    groups: List[_Group] = []

    for entry in sorted(dir_path.iterdir()):
        if not entry.is_file() or entry.suffix.lower() != ".png":
            continue

        placed = False
        for grp in groups:
            if _images_similar(
                entry,
                grp.representative,
                per_channel_tolerance=per_channel_tolerance,
                mean_tolerance=mean_tolerance,
            ):
                grp.names.append(entry.name)
                placed = True
                break

        if not placed:
            groups.append(_Group(representative=entry, names=[entry.name]))

    grouped_names: List[List[str]] = []
    for grp in groups:
        if include_singletons or len(grp.names) > 1:
            grouped_names.append(sorted(grp.names))

    grouped_names.sort(key=lambda items: (len(items) * -1, items[0]))
    return grouped_names


def _format_groups(groups: Iterable[Iterable[str]]) -> str:
    """Convert groups of names to a readable multi-line string."""
    lines = []
    for idx, group in enumerate(groups, start=1):
        joined = ", ".join(group)
        lines.append(f"{idx}. {joined}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group PNG files whose pixel data matches within configurable tolerances."
    )
    parser.add_argument(
        "directory",
        help="Directory containing PNG images to compare.",
    )
    parser.add_argument(
        "--include-singletons",
        action="store_true",
        help="Show groups for files that have no duplicates.",
    )
    parser.add_argument(
        "--per-channel-tolerance",
        type=int,
        default=0,
        help=(
            "Maximum allowed absolute per-channel difference (0-255). "
            "Set negative to ignore this constraint."
        ),
    )
    parser.add_argument(
        "--mean-tolerance",
        type=float,
        default=0.0,
        help=(
            "Maximum allowed average per-channel difference across all pixels. "
            "Set negative to ignore this constraint."
        ),
    )
    args = parser.parse_args()

    groups = group_similar_pngs(
        args.directory,
        include_singletons=args.include_singletons,
        per_channel_tolerance=args.per_channel_tolerance,
        mean_tolerance=args.mean_tolerance,
    )
    if not groups:
        print("No matching PNG files found.")
        return

    print(_format_groups(groups))


if __name__ == "__main__":
    main()
