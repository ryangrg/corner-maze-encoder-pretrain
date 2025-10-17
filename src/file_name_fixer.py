"""
Rename stereo capture files by remapping their direction tokens.

Usage:
    python test/file_name_fixer.py SOURCE_DIR TARGET_DIR

For each PNG named like `configN_x_y_direction_eye.png`, the script copies it to
TARGET_DIR while remapping the `direction` part as follows:
    E(east) -> N(north), N(north) -> W(west), S(south) -> E(east), W(west) -> S(south),
    NE -> SE, NW -> NE, SW -> NW, SE -> SW.

Files that do not match the pattern are skipped. Existing files in TARGET_DIR
are overwritten.
"""

import argparse
import re
import shutil
from pathlib import Path

DIRECTION_MAP = {
    "EAST": "NORTH",
    "NORTH": "WEST",
    "SOUTH": "EAST",
    "WEST": "SOUTH",
    "NE": "NW",
    "NW": "SW",
    "SW": "SE",
    "SE": "NE"
}

FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>.+)_(?P<x>-?\d+)_(?P<y>-?\d+)_(?P<direction>[A-Za-z]+)_(?P<eye>left|right)\.png$",
    re.IGNORECASE,
)


def remap_file(source: Path, destination_dir: Path) -> bool:
    match = FILENAME_PATTERN.match(source.name)
    if not match:
        return False

    direction = match.group("direction")
    remapped_upper = DIRECTION_MAP.get(direction.upper())
    if remapped_upper is None:
        return False

    if direction.islower():
        remapped = remapped_upper.lower()
    elif direction.isupper():
        remapped = remapped_upper.upper()
    elif direction.istitle():
        remapped = remapped_upper.title()
    else:
        remapped = remapped_upper

    new_name = (
        f"{match.group('prefix')}_{match.group('x')}_{match.group('y')}_"
        f"{remapped}_{match.group('eye').lower()}.png"
    )
    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination_dir / new_name)
    return True


def process_directory(source_dir: Path, target_dir: Path) -> tuple[int, int]:
    total = 0
    renamed = 0
    for path in sorted(source_dir.glob("*.png")):
        total += 1
        if remap_file(path, target_dir):
            renamed += 1
    return renamed, total


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rename stereo capture files by remapping direction tokens "
            "and writing into a new directory."
        ),
    )
    parser.add_argument("source_dir", type=Path, help="Directory containing the PNG files.")
    parser.add_argument("target_dir", type=Path, help="Directory to write remapped files.")
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        raise NotADirectoryError(f"{args.source_dir} is not a directory.")
    args.target_dir.mkdir(parents=True, exist_ok=True)

    renamed, total = process_directory(args.source_dir, args.target_dir)
    print(
        f"Processed {total} file(s); remapped {renamed} direction-labelled PNG(s) "
        f"into {args.target_dir}."
    )


if __name__ == "__main__":
    main()
