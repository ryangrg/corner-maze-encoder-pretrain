"""
Simple utility to walk a directory tree and normalize direction tokens in
file/folder names.

Configure `CONFIG_DIR` below to point at the directory that contains the
folders you want to process. For every immediate sub-directory the script will
recursively rename nested files/folders so direction segments are lowercase
(L->l, N->n, NW->nw, etc.). Use with caution and ensure you have backups or
version control.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict

CONFIG_DIR = Path("~/VS Code Local/corner-maze-encoder-pretrain/data/images/change/")  # <-- update this before running

_DIRECTION_REPLACEMENTS: Dict[str, str] = {
    "L": "l",
    "R": "r",
    "N": "n",
    "E": "e",
    "S": "s",
    "W": "w",
    "NW": "nw",
    "NE": "ne",
    "SE": "se",
    "SW": "sw",
}
# Compile a regex that matches complete direction tokens (e.g. "N", "north", "NW"),
# case-insensitively and only when they are not embedded inside larger alphanumeric words.
_DIRECTION_WORD_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(left|right|north|east|south|west)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_DIRECTION_PATTERN = re.compile(
    # (?<![A-Za-z0-9])  => negative lookbehind: ensure the match is not immediately preceded by an alphanumeric char.
    r"(?<![A-Za-z0-9])("
    # The alternation of all direction tokens. We:
    #  - escape each token so any special regex chars are treated literally,
    #  - sort tokens by length (descending) so longer tokens (e.g. "north") are matched before shorter ones (e.g. "n")
    #    to avoid partial/ambiguous matches.
    + "|".join(
        re.escape(token)
        for token in sorted(_DIRECTION_REPLACEMENTS.keys(), key=len, reverse=True)
    )
    # )(?![A-Za-z0-9]) => close the capturing group and use a negative lookahead to ensure the match
    # is not immediately followed by an alphanumeric char.
    + r")(?![A-Za-z0-9])",
    re.IGNORECASE,  # case-insensitive matching so "N", "n", "North", "NORTH", etc. all match
)


def replace_direction_tokens(name: str) -> str:
    """Replace direction tokens (words, single letters, compass pairs) with lowercase forms."""

    def _word_replacement(match: re.Match[str]) -> str:
        mapping = {
            "left": "L",
            "right": "R",
            "north": "N",
            "east": "E",
            "south": "S",
            "west": "W",
        }
        return mapping[match.group(0).lower()]

    name_with_word_tokens = _DIRECTION_WORD_PATTERN.sub(_word_replacement, name)

    def _replacement(match: re.Match[str]) -> str:
        token = match.group(0)
        lookup_key = token.upper()
        if lookup_key not in _DIRECTION_REPLACEMENTS:
            raise KeyError(f"Unexpected direction token '{token}' encountered.")
        return _DIRECTION_REPLACEMENTS[lookup_key]

    return _DIRECTION_PATTERN.sub(_replacement, name_with_word_tokens)


def rename_entry(src: Path, dst: Path) -> None:
    if src == dst:
        return

    def _temporary_path(path: Path, attempt: int) -> Path:
        suffix = f".rename_tmp_{attempt}"
        return path.with_name(path.name + suffix)

    if dst.exists():
        try:
            same_file = src.samefile(dst)
        except FileNotFoundError:
            same_file = False

        if same_file:
            attempt = 0
            temp_path = _temporary_path(src, attempt)
            while temp_path.exists():
                attempt += 1
                temp_path = _temporary_path(src, attempt)
            src.rename(temp_path)
            temp_path.rename(dst)
            print(f"Renamed {src} -> {dst}")
            return

        print(f"Skipping rename {src} -> {dst}: destination already exists.")
        return

    src.rename(dst)
    print(f"Renamed {src} -> {dst}")


def rename_tree(base_dir: Path) -> None:
    if not base_dir.exists():
        raise FileNotFoundError(f"Target directory '{base_dir}' does not exist.")
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Target path '{base_dir}' is not a directory.")

    for current_root, dirs, files in os.walk(base_dir, topdown=False):
        current_path = Path(current_root)

        for file_name in files:
            updated_name = replace_direction_tokens(file_name)
            if updated_name == file_name:
                continue
            rename_entry(current_path / file_name, current_path / updated_name)

        for dir_name in dirs:
            updated_name = replace_direction_tokens(dir_name)
            if updated_name == dir_name:
                continue
            rename_entry(current_path / dir_name, current_path / updated_name)


def rename_children(base_dir: Path) -> None:
    if not base_dir.exists():
        raise FileNotFoundError(f"Provided directory '{base_dir}' does not exist.")
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Provided path '{base_dir}' is not a directory.")

    child_dirs = [path for path in sorted(base_dir.iterdir()) if path.is_dir()]
    if not child_dirs:
        print(f"No sub-directories found inside {base_dir}")
        return

    for child in child_dirs:
        print(f"Processing {child}")
        rename_tree(child)


def main() -> None:
    if CONFIG_DIR == Path("/path/to/config/directories"):
        raise ValueError(
            "Please update CONFIG_DIR in test/rename_files.py to point at the directory you want to process."
        )
    target_dir = CONFIG_DIR.expanduser().resolve()
    rename_children(target_dir)


if __name__ == "__main__":
    main()
