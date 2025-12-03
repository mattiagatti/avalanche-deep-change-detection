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


def convert_all_binary_tifs_to_grayscale() -> None:
    """Convert all binary .tif files in ROOT to grayscale PNGs in OUT_PATH."""
    for tif_path in ROOT.glob("*.tif"):
        arr = load_first_band(tif_path)

        # Handle nodata if present
        arr = np.where(arr == NODATA, 0, arr)

        # Convert binary values (0/1) to grayscale (0/255)
        if arr.max() == 1:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

        out_path = OUT_PATH / (tif_path.stem + ".png")
        Image.fromarray(arr).save(out_path)
        print(f"Saved grayscale image -> {out_path}")


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
    convert_all_binary_tifs_to_grayscale()


if __name__ == "__main__":
    main()