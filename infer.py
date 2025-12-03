import argparse
import geopandas as gpd
import numpy as np
import rasterio
import re
import timeit
import torch

from datetime import datetime
from models.swinunet import ChangeDetectionSwinUNet
from pathlib import Path
from rasterio.features import shapes
from scipy.ndimage import binary_fill_holes
from shapely.geometry import shape
from skimage.util.shape import view_as_windows
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from utils.morph import morph_close
from utils.slope import ensure_slope


torch.set_num_threads(8)  # intra-op: threads inside a single op
torch.set_num_interop_threads(4)  # inter-op: parallelism across ops

parser = argparse.ArgumentParser(description="Run inference on a single event with selectable blending strategy.")
parser.add_argument("--event-path", type=str, required=True, help="Path to the event folder")
parser.add_argument("--model-ckpt", type=str, default="exp/swinunet_128/best_model.pth", help="Path to the model checkpoint")
parser.add_argument("--output-dir", type=str, default="output/infer", help="Directory to save output predictions")
parser.add_argument("--patch-size", type=int, default=128)
parser.add_argument("--stride", type=int, default=64)
parser.add_argument("--blending", type=str, choices=["mean", "max", "min", "gaussian", "center_crop"], default="center_crop")
parser.add_argument("--use-aux", action="store_true", help="Enable Auxiliary data usage.")
parser.add_argument("--force", action="store_true", help="Recompute even if outputs already exist.")

args = parser.parse_args()

PATCH_SIZE = (args.patch_size, args.patch_size)
STRIDE = (args.stride, args.stride)
USE_AUX = args.use_aux
event_path = Path(args.event_path)
model_ckpt = Path(args.model_ckpt)


def is_yyyymmdd(s: str) -> bool:
    if not re.fullmatch(r"\d{8}", s):
        return False
    try:
        datetime.strptime(s, "%Y%m%d")
        return True
    except ValueError:
        return False


area = event_path.parts[-2] if len(event_path.parts) >= 2 else "unknown"
date_candidate = event_path.name  # last path part

if is_yyyymmdd(date_candidate):
    output_dir = Path(args.output_dir) / area / date_candidate
else:
    output_dir = Path(args.output_dir) / area

output_dir.mkdir(parents=True, exist_ok=True)

NAN_PERCENT = 0.8


def get_raster_paths(event_path: Path, use_aux: bool):
    # Base SAR keys always required
    expected_keywords = ["preVH", "preVV", "postVH", "postVV"]
    raster_files = {key: None for key in expected_keywords}

    # If using AUX, also expect LIA and derive SLP
    if use_aux:
        expected_keywords.append("LIA")

    # Scan folder
    for tif_file in event_path.glob("*.tif"):
        for key in expected_keywords:
            if key in tif_file.name:
                raster_files[key] = tif_file

    missing = [k for k, v in raster_files.items() if v is None]
    if missing:
        raise FileNotFoundError(f"Missing expected raster files for: {missing}")

    if use_aux:
        # ensure_slope creates/returns the SLP .tif based on a SAR reference
        slope_path = ensure_slope(event_path, raster_files["preVH"])
        raster_files["SLP"] = Path(slope_path)
        ordered_keys = ["preVH", "preVV", "postVH", "postVV", "LIA", "SLP"]
    else:
        ordered_keys = ["preVH", "preVV", "postVH", "postVV"]

    return [raster_files[k] for k in ordered_keys]


def extract_patches_stack(raster_paths, patch_size, stride, region_mask=None):
    with rasterio.open(raster_paths[0]) as src:
        meta = src.meta.copy()
        height, width = src.shape
        num_bands = len(raster_paths)

        raster_array = []
        for path in raster_paths:
            with rasterio.open(path) as src:
                img = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    img = np.where(img == src.nodata, np.nan, img)
                # >>> mask out-of-region as NaN <<<
                if region_mask is not None:
                    img = np.where(region_mask == 1, img, np.nan)
                raster_array.append(img)
        raster_array = np.stack(raster_array, axis=0)

    patches = view_as_windows(raster_array, (num_bands, *patch_size), (1, *stride))
    grid_shape = patches.shape[1:3]
    return patches, grid_shape, (height, width), meta


class AvalancheDataset(Dataset):
    def __init__(self, patches, region_patches, stats, use_aux: bool):
        self.patches = patches
        self.region_patches = region_patches
        self.stats = stats
        self.use_aux = use_aux

        self.valid_indices = []
        for idx in range(len(self.patches)):
            region_patch = self.region_patches[idx]
            inside_region = np.all(region_patch == 1)

            # first 4 channels are SAR (always there)
            sar_patch = self.patches[idx][:4]
            nan_mask = np.isnan(sar_patch).any(axis=0)
            nan_fraction = nan_mask.mean()

            if inside_region or nan_fraction < NAN_PERCENT:
                self.valid_indices.append(idx)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        raw_patch = self.patches[idx]

        # sanitize SAR channels
        raw_patch[:4] = np.where(
            (~np.isfinite(raw_patch[:4])) | (raw_patch[:4] < -40.0) | (raw_patch[:4] > 20.0),
            np.nan,
            raw_patch[:4]
        )

        patch = torch.tensor(raw_patch, dtype=torch.float32)

        pre = patch[:2]
        post = patch[2:4]

        # --- normalization stats for SAR ---
        mean_img = self.stats["img_mean"].view(-1, 1, 1)
        std_img = self.stats["img_std"].view(-1, 1, 1)
        fill_img = -self.stats["sentinel_z_img"].view(-1, 1, 1)

        pre_z = (pre - mean_img) / std_img
        post_z = (post - mean_img) / std_img
        for c in range(pre_z.shape[0]):
            pre_z[c][~torch.isfinite(pre_z[c])] = fill_img[c].item()
            post_z[c][~torch.isfinite(post_z[c])] = fill_img[c].item()

        if self.use_aux:
            lia = patch[4]
            slope = patch[5]
            aux = torch.stack([lia, slope], dim=0)

            mean_aux = self.stats["aux_mean"].view(-1, 1, 1)
            std_aux = self.stats["aux_std"].view(-1, 1, 1)
            fill_aux = -self.stats["sentinel_z_aux"].view(-1, 1, 1)

            aux_z = (aux - mean_aux) / std_aux
            for c in range(aux_z.shape[0]):
                aux_z[c][~torch.isfinite(aux_z[c])] = fill_aux[c].item()
        else:
            # return a placeholder aux with right spatial dims to keep collate simple
            H = pre_z.shape[1]
            W = pre_z.shape[2]
            aux_z = torch.zeros(2, H, W, dtype=torch.float32)

        return pre_z, post_z, aux_z

    def get_valid_indices(self):
        return self.valid_indices


def merge_patches(patch_outputs, grid_shape, patch_size, stride, original_shape, valid_indices, best_threshold, mode="center_crop"):
    patch_outputs = np.concatenate(patch_outputs, axis=0)
    patch_height, patch_width = patch_size
    grid_rows, grid_cols = grid_shape
    full_height, full_width = original_shape

    valid_grid = np.zeros((grid_rows, grid_cols), dtype=bool)
    for idx in valid_indices:
        r, c = divmod(idx, grid_cols)
        valid_grid[r, c] = True

    if mode == "max":
        full_image = np.full((full_height, full_width), -np.inf, dtype=np.float32)
    elif mode == "min":
        full_image = np.full((full_height, full_width), np.inf, dtype=np.float32)
    else:
        full_image = np.zeros((full_height, full_width), dtype=np.float32)

    count_map = np.zeros((full_height, full_width), dtype=np.float32)

    patch_idx = 0
    valid_patch_idx = 0

    # put near the top of merge_patches, after patch_height/width, etc.
    overlap_y = max(0, patch_height - stride[0])
    overlap_x = max(0, patch_width - stride[1])

    # canonical split: half overlap to top/left, remainder to bottom/right
    half_y = overlap_y // 2
    half_x = overlap_x // 2
    rem_y = overlap_y - half_y   # handles odd overlaps
    rem_x = overlap_x - half_x

    if mode == "gaussian":
        y = np.linspace(-1, 1, patch_height)
        x = np.linspace(-1, 1, patch_width)
        xv, yv = np.meshgrid(x, y)
        gaussian_weights = np.exp(-(xv**2 + yv**2) / 0.5)
        gaussian_weights = gaussian_weights.astype(np.float32)
    else:
        gaussian_weights = None

    for i in range(grid_rows):
        for j in range(grid_cols):
            y, x = i * stride[0], j * stride[1]

            if patch_idx in valid_indices:
                patch = patch_outputs[valid_patch_idx]

                if mode == "center_crop":
                    # which neighbors exist & are valid?
                    top_ok = (i > 0) and valid_grid[i - 1, j]
                    bottom_ok = (i < grid_rows - 1) and valid_grid[i + 1, j]
                    left_ok = (j > 0) and valid_grid[i, j - 1]
                    right_ok = (j < grid_cols - 1) and valid_grid[i, j + 1]

                    # crop half-overlap toward neighbors; keep full area if neighbor missing
                    top_crop = half_y if top_ok else 0
                    bottom_crop = rem_y if bottom_ok else 0
                    left_crop = half_x if left_ok else 0
                    right_crop = rem_x if right_ok else 0

                    # apply crop and write
                    y0 = y + top_crop
                    x0 = x + left_crop
                    y1 = y + patch_height - bottom_crop
                    x1 = x + patch_width - right_crop

                    cropped = patch[top_crop: patch_height - bottom_crop,
                                    left_crop: patch_width - right_crop]

                    full_image[y0:y1, x0:x1] = cropped
                    count_map[y0:y1, x0:x1] += 1

                elif mode == "gaussian":
                    full_image[y:y + patch_height, x:x + patch_width] += patch * gaussian_weights
                    count_map[y:y + patch_height, x:x + patch_width] += gaussian_weights

                elif mode == "max":
                    full_image[y:y + patch_height, x:x + patch_width] = np.maximum(
                        full_image[y:y + patch_height, x:x + patch_width], patch
                    )
                    count_map[y:y + patch_height, x:x + patch_width] += 1

                elif mode == "min":
                    region = full_image[y:y + patch_height, x:x + patch_width]
                    mask = count_map[y:y + patch_height, x:x + patch_width] == 0
                    region[mask] = patch[mask]
                    full_image[y:y + patch_height, x:x + patch_width] = np.minimum(region, patch)
                    count_map[y:y + patch_height, x:x + patch_width] += 1

                else:
                    full_image[y:y + patch_height, x:x + patch_width] += patch
                    count_map[y:y + patch_height, x:x + patch_width] += 1

                valid_patch_idx += 1

            patch_idx += 1

    valid_mask = count_map > 0
    if mode in ["mean", "gaussian"]:
        full_image[valid_mask] /= count_map[valid_mask]

    full_image[~valid_mask] = 255
    return full_image


event_name = event_path.name
out_path = output_dir / f"{event_name}.tif"
output_gpkg_path = output_dir / f"{event_name}.gpkg"

if not args.force:
    tif_ok = out_path.exists() and out_path.stat().st_size > 0
    gpkg_ok = output_gpkg_path.exists() and output_gpkg_path.stat().st_size > 0
    if tif_ok and gpkg_ok:
        print(f"Outputs already exist: {out_path} and {output_gpkg_path}. Skipping. "
              f"(Use --force to recompute)")
        exit()

# Load model
model = ChangeDetectionSwinUNet(img_size=args.patch_size, use_aux=USE_AUX)
checkpoint = torch.load(model_ckpt, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
best_thr = checkpoint.get("best_threshold", 0.5)

# Load raster paths
raster_paths = get_raster_paths(event_path, USE_AUX)

# Build region mask
with rasterio.open(raster_paths[0]) as src_vh_pre:
    vh_pre = src_vh_pre.read(1)
with rasterio.open(raster_paths[1]) as src_vv_pre:
    vv_pre = src_vv_pre.read(1)
with rasterio.open(raster_paths[2]) as src_vh_post:
    vh_post = src_vh_post.read(1)
with rasterio.open(raster_paths[3]) as src_vv_post:
    vv_post = src_vv_post.read(1)

valid_mask = (~np.isnan(vh_pre) & ~np.isnan(vv_pre) & ~np.isnan(vh_post) & ~np.isnan(vv_post))
region_mask = binary_fill_holes(valid_mask).astype(np.uint8)

# Extract patches
stacked_patches, grid_shape, original_shape, meta = extract_patches_stack(raster_paths, PATCH_SIZE, STRIDE, region_mask)
patches = stacked_patches.reshape(-1, *stacked_patches.shape[3:])
region_patches = view_as_windows(region_mask, PATCH_SIZE, STRIDE).reshape(-1, *PATCH_SIZE)

# Load stats
stats_tensor = checkpoint["norm_stats"]

# Inference
dataset = AvalancheDataset(patches, region_patches, stats_tensor, USE_AUX)
loader = DataLoader(Subset(dataset, dataset.get_valid_indices()), batch_size=32, shuffle=False)

patch_outputs = []

start = timeit.default_timer()

with torch.no_grad():
    for pre_patch, post_patch, aux_patch in tqdm(loader, desc="Inferencing", ncols=100):
        pre_patch, post_patch = pre_patch.to(device), post_patch.to(device)
        if not USE_AUX:
            logits = model(pre_patch, post_patch)
        else:
            aux_patch = aux_patch.to(device)
            logits = model(pre_patch, post_patch)
        preds = torch.sigmoid(logits)
        patch_outputs.append(preds.cpu().numpy().squeeze(1))

# Stitch and save output
reconstructed = merge_patches(
    patch_outputs, grid_shape, PATCH_SIZE, STRIDE,
    original_shape, dataset.get_valid_indices(), best_thr, mode=args.blending
)

# Build valid mask (exclude nodata=255)
valid = reconstructed != 255

# ---- Threshold then morphological closing on the full map ----
# Set nodata to 0 so they don't create false dilations
prob_2d = reconstructed.copy().astype(np.float32)
prob_2d[~valid] = 0.0

# Binary map
binary = (prob_2d > best_thr).astype(np.uint8)

# Morphological closing (fill small holes / bridge small gaps)
# morph_close works on numpy (H, W) arrays
binary_closed = morph_close(binary, kernel_size=3, iterations=1).astype(np.uint8)

duration = timeit.default_timer() - start

print(f"Inference + reconstruction time: {duration:.2f} seconds")

# Restore nodata=255 outside valid area
binary_mask = np.full(binary_closed.shape, 255, dtype=np.uint8)
binary_mask[valid] = binary_closed[valid]

meta.update(
    driver="GTiff",
    count=1,
    dtype="uint8",
    nodata=255,
    tiled=True,
    blockxsize=256,
    blockysize=256,
    compress="LZW",
    BIGTIFF="IF_SAFER",
    NUM_THREADS="ALL_CPUS",
)
with rasterio.open(out_path, "w", **meta) as dst:
    dst.write(binary_mask, 1)

print(f"Saved output to: {out_path}")

# Extract polygons from raster
polygons = []
with rasterio.open(out_path) as src:
    image = src.read(1)
    mask = image != 255  # Ignore nodata (255)
    transform = src.transform

    for geom, val in shapes(image, mask=mask, transform=transform, connectivity=8):
        if val == 1:  # Keep only predicted class
            polygons.append(shape(geom))

    crs = src.crs  # Coordinate reference system

gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
gdf.to_file(output_gpkg_path, driver="GPKG")

print(f"Polygons saved to: {output_gpkg_path}")