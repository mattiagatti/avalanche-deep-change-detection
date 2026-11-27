#!/usr/bin/env python3
"""
Count positive/negative avalanche patches by event and patch size using rasterio,
and print a terminal report (no file output).

Expected layout:
<base>/<patch_size>/<event>/<patch_id>/mask.tif

Defaults:
  base = /home/jovyan/nfs/mgatti/datasets/Avalanches/patches
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import numpy as np
import rasterio

try:
    from tqdm import tqdm
except ImportError as e:
    print("[ERROR] tqdm is required. Install it with: pip install tqdm", file=sys.stderr)
    sys.exit(3)


def _is_positive_mask_rio(path: Path) -> bool:
    """True if the raster has at least one pixel > 0 across any band."""
    with rasterio.open(path) as ds:
        data = ds.read(masked=True)  # (bands, rows, cols), masked array
        filled = np.ma.filled(data, fill_value=0)
        return bool((filled > 0).any())


def _discover_masks(base: Path, sizes: List[int] | None) -> List[Tuple[int, str, Path]]:
    """Return list of (size, event, mask_path)."""
    if sizes is None:
        sizes = sorted([int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()])

    records: List[Tuple[int, str, Path]] = []
    for sz in sizes:
        size_dir = base / str(sz)
        if not size_dir.is_dir():
            continue
        for event_dir in sorted([p for p in size_dir.iterdir() if p.is_dir()]):
            event = event_dir.name
            for patch_dir in sorted([p for p in event_dir.iterdir() if p.is_dir()]):
                mask_path = patch_dir / "mask.tif"
                if mask_path.exists():
                    records.append((sz, event, mask_path))
    return records


def count_with_progress(base: Path, sizes: List[int] | None) -> Dict[int, Dict[str, Dict[str, int]]]:
    """Build counts[patch_size][event]['pos'|'neg'] with a single global tqdm."""
    recs = _discover_masks(base, sizes)
    if not recs:
        return {}

    sizes_sorted = sorted({sz for sz, _, _ in recs})
    events_sorted = sorted({ev for _, ev, _ in recs})
    counts: Dict[int, Dict[str, Dict[str, int]]] = {
        sz: {ev: {"pos": 0, "neg": 0} for ev in events_sorted} for sz in sizes_sorted
    }

    for sz, ev, mask_path in tqdm(recs, desc="Scanning masks", unit="mask", total=len(recs)):
        try:
            if _is_positive_mask_rio(mask_path):
                counts[sz][ev]["pos"] += 1
            else:
                counts[sz][ev]["neg"] += 1
        except Exception as e:
            print(f"[WARN] Skipping unreadable {mask_path}: {e}", file=sys.stderr)
            continue

    return counts


def print_terminal_report(counts: Dict[int, Dict[str, Dict[str, int]]]) -> None:
    """Pretty terminal table: Event | <size Pos> <size Neg> ... plus totals."""
    if not counts:
        print("[ERROR] No counts found. Check folder structure and sizes.", file=sys.stderr)
        return

    sizes_sorted = sorted(counts.keys())
    events = sorted({ev for s in sizes_sorted for ev in counts[s].keys()})

    colw_event = max(5, max(len(ev) for ev in events)) if events else 10
    header_cells: List[str] = []
    for s in sizes_sorted:
        header_cells += [f"{s} Pos", f"{s} Neg"]

    print("\nSummary (Pos/Neg) per event and size:\n")
    print(f"{'Event':<{colw_event}}  " + "  ".join([f"{h:>8}" for h in header_cells]))
    print("-" * (colw_event + 2 + 10 * len(header_cells)))

    for ev in events:
        row_vals: List[int] = []
        for s in sizes_sorted:
            row_vals.append(counts[s].get(ev, {}).get("pos", 0))
            row_vals.append(counts[s].get(ev, {}).get("neg", 0))
        print(f"{ev:<{colw_event}}  " + "  ".join([f"{v:>8d}" for v in row_vals]))

    total_cells = []
    for s in sizes_sorted:
        p = sum(counts[s].get(ev, {}).get("pos", 0) for ev in events)
        n = sum(counts[s].get(ev, {}).get("neg", 0) for ev in events)
        total_cells += [f"{p:>8d}", f"{n:>8d}"]
    print("-" * (colw_event + 2 + 10 * len(header_cells)))
    print(f"{'Total':<{colw_event}}  " + "  ".join(total_cells))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Count positive/negative avalanche patches with rasterio and print a terminal report."
    )
    parser.add_argument(
        "-b", "--base", type=Path,
        default=Path("/home/jovyan/nfs/mgatti/datasets/Avalanches/patches"),
        help="Base path to 'patches' directory."
    )
    parser.add_argument(
        "-s", "--sizes", type=int, nargs="*", default=None,
        help="Patch sizes to include (e.g. -s 32 64 128 256). Defaults to autodetect numeric subfolders."
    )
    args = parser.parse_args()

    base: Path = args.base
    if not base.exists():
        print(f"[ERROR] Base path does not exist: {base}", file=sys.stderr)
        sys.exit(1)

    counts = count_with_progress(base, sizes=args.sizes)
    if not counts:
        sys.exit(2)

    print_terminal_report(counts)


if __name__ == "__main__":
    main()