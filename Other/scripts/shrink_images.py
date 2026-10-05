"""Pre-resize line crops to the largest size any Model122 config can consume.

Both the height-normalized run (target_height 128, longest_edge 2800) and the
native control (processor cap longest_edge 1540) downscale anything larger, so
capping height at 176px and the longest edge at 2800px is near-lossless while
making the upload to a rented GPU feasible on a slow uplink.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

MAX_HEIGHT = 176
MAX_EDGE = 2800
QUALITY = 92


def convert(job: tuple[Path, Path]) -> tuple[int, int]:
    source, destination = job
    with Image.open(source) as image:
        image = image.convert("RGB")
        width, height = image.size
        scale = min(1.0, MAX_HEIGHT / height, MAX_EDGE / max(width, height))
        if scale < 1.0:
            image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                                 Image.Resampling.LANCZOS)
        image.save(destination, "JPEG", quality=QUALITY, optimize=True)
    return source.stat().st_size, destination.stat().st_size


def _init(max_height: int, max_edge: int, quality: int) -> None:
    """Workers are separate processes on Windows, so the caps are set here."""
    global MAX_HEIGHT, MAX_EDGE, QUALITY
    MAX_HEIGHT, MAX_EDGE, QUALITY = max_height, max_edge, quality


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-height", type=int, default=MAX_HEIGHT)
    parser.add_argument("--max-edge", type=int, default=MAX_EDGE)
    parser.add_argument("--quality", type=int, default=QUALITY)
    args = parser.parse_args()
    source_dir, out_dir = Path(args.source), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(path, out_dir / path.name) for path in sorted(source_dir.glob("*.jpg"))]
    before = after = 0
    _init(args.max_height, args.max_edge, args.quality)
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(args.max_height, args.max_edge, args.quality)) as pool:
        for old, new in pool.map(convert, jobs, chunksize=32):
            before += old
            after += new
    print(f"{len(jobs)} images: {before/2**20:.1f} MiB -> {after/2**20:.1f} MiB")


if __name__ == "__main__":
    main()
