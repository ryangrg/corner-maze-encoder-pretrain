"""
Sort x,y coordinate pairs within a text file.

Usage:
    python test/coordinate_sort.py positions.txt

Each non-empty line must contain a comma-separated pair (e.g. "5,7" or "3.2, -1").
The script sorts lines numerically by x then y and overwrites the original file.
Blank lines at the end of the file are preserved.
"""

import argparse
from pathlib import Path


def _parse_pair(line: str, line_no: int) -> tuple[float, float]:
    try:
        x_str, y_str = line.split(",", 1)
    except ValueError as exc:
        raise ValueError(f"Line {line_no}: expected 'x,y' format, got {line!r}") from exc

    try:
        return float(x_str.strip()), float(y_str.strip())
    except ValueError as exc:
        raise ValueError(f"Line {line_no}: non-numeric coordinate in {line!r}") from exc


def sort_coordinate_file(path: Path) -> int:
    lines = path.read_text().splitlines()

    coord_lines = [line for line in lines if line.strip()]
    trailing_blank_count = len(lines) - len(coord_lines)

    coords: list[tuple[float, float]] = []
    for idx, line in enumerate(coord_lines, start=1):
        coords.append(_parse_pair(line, idx))

    coords.sort(key=lambda pair: (pair[0], pair[1]))

    sorted_lines = [f"{x:g},{y:g}" for x, y in coords]
    if trailing_blank_count:
        sorted_lines.extend([""] * trailing_blank_count)

    path.write_text("\n".join(sorted_lines) + "\n")
    return len(coords)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="coordinate_sort",
        description="Sort x,y coordinate pairs numerically and overwrite the source file.",
    )
    parser.add_argument("file", type=Path, help="Path to the coordinate text file.")
    args = parser.parse_args()

    target = args.file
    if not target.is_file():
        raise FileNotFoundError(f"{target} does not exist or is not a file.")

    count = sort_coordinate_file(target)
    print(f"Sorted {count} coordinate pair(s) in {target}.")


if __name__ == "__main__":
    main()
