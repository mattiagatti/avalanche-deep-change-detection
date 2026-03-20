import numpy as np
import rasterio
from pathlib import Path
from PIL import Image

# Paths
ROOT = Path("output/test_blending")
GT_PATH = ROOT / "ground_truth.tif"
PRED_PATH = ROOT / "pred_max.tif"
OUT_PATH = ROOT / "images"

OUT_PATH.mkdir(exist_ok=True, parents=True)

# Constants
NODATA = 255
GT_POSITIVE = 1
PRED_POSITIVE = 1


def load_first_band(path: Path) -> np.ndarray:
    """Return the first band of a single-band raster as a NumPy array."""
    with rasterio.open(path) as src:
        return src.read(1)


def convert_all_tifs_to_png() -> None:
    for tif_path in sorted(ROOT.glob("*.tif")):
        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata

        if nodata is None:
            valid_mask = np.isfinite(arr)
        else:
            valid_mask = np.isfinite(arr) & (arr != nodata)

        if not np.any(valid_mask):
            print(f"Skipped {tif_path} (no valid pixels)")
            continue

        valid = arr[valid_mask]
        uniq = np.unique(valid)

        # Binary raster
        if np.all(np.isin(uniq, [0.0, 1.0])):
            out = np.zeros_like(arr, dtype=np.uint8)
            out[valid_mask] = (arr[valid_mask] * 255).astype(np.uint8)

        # Float/probability raster
        else:
            vmin = float(valid.min())
            vmax = float(valid.max())

            out = np.zeros_like(arr, dtype=np.uint8)
            if vmax > vmin:
                scaled = (arr[valid_mask] - vmin) / (vmax - vmin)
                out[valid_mask] = np.clip(scaled * 255, 0, 255).astype(np.uint8)

        out_path = OUT_PATH / f"{tif_path.stem}.png"
        Image.fromarray(out).save(out_path)
        print(f"Saved PNG -> {out_path} | min={valid.min():.6f}, max={valid.max():.6f}")


def generate_confusion_map() -> None:
    # 1. Read rasters
    gt = load_first_band(GT_PATH)
    pred = load_first_band(PRED_PATH)

    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: GT {gt.shape} vs PRED {pred.shape}")

    # 2. Build boolean masks for each outcome
    nodata_mask = (gt == NODATA) | (pred == NODATA)

    tp = (pred == PRED_POSITIVE) & (gt == GT_POSITIVE) & ~nodata_mask
    fp = (pred == PRED_POSITIVE) & (gt != GT_POSITIVE) & ~nodata_mask
    fn = (pred != PRED_POSITIVE) & (gt == GT_POSITIVE) & ~nodata_mask
    tn = (pred != PRED_POSITIVE) & (gt != GT_POSITIVE) & ~nodata_mask

    # 3. Assemble RGB image
    h, w = gt.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    rgb[tn] = (0, 0, 0)         # black   : true negative
    rgb[tp] = (0, 255, 0)       # green    : true positive
    rgb[fn] = (255, 0, 0)       # red     : false negative
    rgb[fp] = (255, 255, 0)     # yellow  : false positive
    rgb[nodata_mask] = (0, 0, 0)  # black  : nodata

    # 4. Save confusion map
    out_conf = OUT_PATH / "confusion_map.png"
    Image.fromarray(rgb).save(out_conf)
    print(f"Confusion map saved -> {out_conf}")


def main() -> None:
    generate_confusion_map()
    convert_all_tifs_to_png()


if __name__ == "__main__":
    main()