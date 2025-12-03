import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pathlib import Path
from PIL import Image


def normalize_image(img):
    """Percentile-based contrast-stretch to 0-255 uint8."""
    v_min, v_max = np.percentile(img, (2, 98))
    img = np.clip((img - v_min) / (v_max - v_min), 0, 1) * 255
    return img.astype(np.uint8)


def read_two_band(path):
    with rasterio.open(path) as src:
        return src.read(1), src.read(2)   # VV, VH


def read_single_band(path):
    with rasterio.open(path) as src:
        return src.read(1)


def save_png(arr, out_path, cmap=None):
    """Save 2-D array as PNG (optionally with matplotlib colormap)."""
    if cmap is None:
        Image.fromarray(arr).convert("L").save(out_path)
    else:
        normed = arr / 255.0
        cmap_fn = matplotlib.colormaps.get_cmap(cmap)
        colored = cmap_fn(normed)[:, :, :3]
        rgb = (colored * 255).astype(np.uint8)
        Image.fromarray(rgb).save(out_path)


def plot_grid(pre_vv, pre_vh, post_vv, post_vh, mask, slope, lia, out_path):
    imgs = [
        normalize_image(pre_vv), normalize_image(pre_vh),
        normalize_image(post_vv), normalize_image(post_vh),
        normalize_image(slope), (mask * 255).astype(np.uint8),
    ]
    titles = ["Pre VV", "Pre VH", "Post VV", "Post VH", "Slope", "LIA", "Mask"]
    cmaps = ["gray", "gray", "gray", "gray", "terrain", "viridis", "gray"]

    fig, axs = plt.subplots(2, 4, figsize=(14, 7))
    axs = axs.flatten()
    
    for i in range(7):
        axs[i].imshow(imgs[i], cmap=cmaps[i], vmin=0, vmax=255)
        axs[i].set_title(titles[i])
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    
    axs[7].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_sample(sample_dir: Path, out_dir: Path, mode="single"):
    """Write all visualisations for one sample directory."""
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_vv, pre_vh = read_two_band(sample_dir / "pre.tif")
    post_vv, post_vh = read_two_band(sample_dir / "post.tif")
    mask = read_single_band(sample_dir / "mask.tif")
    slope = read_single_band(sample_dir / "slope.tif")
    lia = read_single_band(sample_dir / "lia.tif")

    all_arrays = {
        "pre_vv": pre_vv,
        "pre_vh": pre_vh,
        "post_vv": post_vv,
        "post_vh": post_vh,
        "mask": mask,
        "slope": slope,
        "lia": lia
    }

    for name, arr in all_arrays.items():
        if np.isnan(arr).any():
            print(f"NaNs found in {name} of {sample_dir.name}")
            return

    if mode == "single":
        save_png(normalize_image(pre_vv),  out_dir / "pre_vv.png")
        save_png(normalize_image(post_vv), out_dir / "post_vv.png")
        save_png(normalize_image(pre_vh),  out_dir / "pre_vh.png")
        save_png(normalize_image(post_vh), out_dir / "post_vh.png")
        save_png((mask * 255).astype(np.uint8), out_dir / "mask.png")
        save_png(normalize_image(slope), out_dir / "slope.png", cmap="terrain")
        save_png(normalize_image(lia),   out_dir / "lia.png",   cmap="viridis")
    elif mode == "grid":
        out_path = out_dir / "sample_grid.png"
        plot_grid(pre_vv, pre_vh, post_vv, post_vh, mask, slope, lia, out_path)
    else:
        raise ValueError("mode must be 'single' or 'grid'")


def process_dataset(dataset_root: Path,
                    mode="single"):
    """
    Walk through every immediate sub-folder of *dataset_root*
    and export its visualisations.
    """
    outputs_root = Path(f"output/samples/{dataset_root.name}/{dataset_root.parent.name}")
    outputs_root.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    if not sample_dirs:
        print(f"No sub-folders found in {dataset_root}")
        return

    for d in sample_dirs:
        out_dir = outputs_root / d.name
        print(f"Processing {d.name} -> {out_dir}")
        try:
            save_sample(d, out_dir, mode=mode)
        except FileNotFoundError as e:
            print(f"  Skipped {d.name}: missing file – {e}")
        except Exception as e:
            print(f"  Error on {d.name}: {e}")


if __name__ == "__main__":
    SIZE = 128
    dataset_root = Path(
        f"/home/jovyan/nfs/mgatti/datasets/Avalanches/patches/{SIZE}/Livigno_20240403"
    )
    process_dataset(dataset_root, mode="single")   # or mode="grid"