import argparse
from pathlib import Path

try:
    from PIL import Image, ImageFilter
except ImportError as exc:  # pragma: no cover - communicates dependency requirement
    raise ImportError(
        "png-smoother requires Pillow. Install it with `pip install pillow`."
    ) from exc

"""
Utility for smoothing PNG images with a Gaussian blur.

Command line usage:
    python test/png_smoother.py DIRECTORY [--radius FLOAT] [--overwrite] [--output-dir NAME]

Inputs and options:
    DIRECTORY     Path containing the PNG files to process.
    --radius      Blur radius (float, default 1.0).
    --overwrite   Replace each source image instead of writing a new file.
    --output-dir  Name of the folder (created next to this script) for blurred copies;
                  ignored when --overwrite is set.

By default the script writes blurred copies named *_smooth.png into the folder
`smoothed-image` inside the `test/` directory.
"""


def smooth_pngs(
    directory: str,
    radius: float = 1.0,
    overwrite: bool = False,
    output_dir: str | None = None,
) -> int:
    """
    Apply a Gaussian blur to every PNG in the given directory.

    Args:
        directory: Path to a folder containing PNG images.
        radius: Gaussian blur radius in pixels.
        overwrite: When True, replace the original image; otherwise write alongside it
            or inside output_dir if provided.
        output_dir: Optional directory name (created inside the Test folder) for smoothed images.
            Ignored when overwrite=True.

    Returns:
        The number of PNG files processed.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"{directory!r} is not a directory")

    if output_dir and overwrite:
        raise ValueError("Cannot specify output_dir when overwrite=True.")

    script_dir = Path(__file__).resolve().parent
    target_dir = dir_path
    if not overwrite:
        target_dir_name = output_dir or "smoothed-image"
        target_dir = script_dir / target_dir_name
        target_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for entry in sorted(dir_path.iterdir()):
        if not entry.is_file() or entry.suffix.lower() != ".png":
            continue

        with Image.open(entry) as img:
            blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))

            if overwrite:
                blurred.save(entry)
                out_path = entry
            else:
                out_name = f"{entry.stem}_smooth{entry.suffix}"
                out_path = target_dir / out_name
                blurred.save(out_path)

            print(f"Smoothed {entry.name} -> {out_path.name}")
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a Gaussian blur to PNG files within a directory."
    )
    parser.add_argument("directory", help="Directory containing PNG images.")
    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="Gaussian blur radius in pixels (default: 1.0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the original files instead of writing *_smooth.png copies.",
    )
    parser.add_argument(
        "--output-dir",
        default="smoothed-image",
        help=(
            "Directory (created inside the Test folder) for smoothed files "
            "(ignored when --overwrite is used)."
        ),
    )
    args = parser.parse_args()

    processed = smooth_pngs(
        args.directory,
        radius=args.radius,
        overwrite=args.overwrite,
        output_dir=None if args.overwrite else args.output_dir,
    )
    if processed == 0:
        print("No PNG files found.")
    else:
        print(f"Processed {processed} PNG file(s).")


if __name__ == "__main__":
    main()
