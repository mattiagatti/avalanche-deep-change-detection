#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Avalanche CD – tiled inference + optional blending, with optional aux channels (LIA, SLP).

Key changes vs your original:
- Optional aux channels: pass --use-aux to include LIA & SLP. Default: off.
- get_raster_paths() no longer fails when aux is missing and aux is disabled.
- Consistent model calls with/without aux; model initialized with use_aux flag.
- Single flow that runs:
    * tiling -> patch inference -> stitching (multiple modes) -> metrics -> GeoTIFF
- "none" mode now truly writes a file (adjacent stitching only; no overlap).
- Safer NaN handling & stats application when aux is disabled.
- CLI via argparse; sensible defaults mirror your constants.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shutil
import torch
from rasterio.features import rasterize
from scipy.ndimage import binary_fill_holes
from skimage.util.shape import view_as_windows
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics.classification import Precision, Recall, F1Score, JaccardIndex
from tqdm import tqdm

# ---- your local modules ----
from models.swinunet import ChangeDetectionSwinUNet
from utils.morph import morph_close


# ------------------------------- #
# Defaults (can be overridden CLI)
# ------------------------------- #
DEFAULT_PATCH_SIZE = (128, 128)
DEFAULT_STRIDE = (64, 64)
DEFAULT_RASTERS_DIR = Path("output/test_blending")
DEFAULT_EVENT_PATH = Path("/home/jovyan/nfs/mgatti/datasets/Avalanches/AvalCD/Tromso_20241220/")
DEFAULT_STATS_PATH = Path("/home/jovyan/nfs/mgatti/datasets/Avalanches/patches/128/stats.json")
DEFAULT_MODEL_CKPT = Path("/home/jovyan/nfs/mgatti/python/avalanches/exp/swinunet_128_F2/best_model.pth")
DEFAULT_GPKG = Path("/home/jovyan/nfs/mgatti/datasets/Avalanches/AvalCD/Tromso_20241220/Tromso_20241220.gpkg")
DEFAULT_MIN_FRACTION_INSIDE = 0.5
DEFAULT_NAN_PERCENT = 0.8  # align with test.py (0.5) for sanity checks
DEFAULT_MODES = ["none", "mean", "max", "min", "gaussian", "center_crop"]


# -------------------- #
# Utility / IO helpers #
# -------------------- #
def get_raster_paths(event_path: Path, use_aux: bool) -> Dict[str, Optional[Path]]:
    """
    Return a dict of expected rasters from folder.
    Always requires SAR: preVH, preVV, postVH, postVV.
    Optionally requires LIA & SLP if use_aux=True.

    Raises if any required band is missing.
    """
    required = ["preVH", "preVV", "postVH", "postVV"]
    optional = ["LIA", "SLP"]
    expected = required + (optional if use_aux else [])

    found: Dict[str, Optional[Path]] = {k: None for k in expected}
    for tif in event_path.glob("*.tif"):
        for key in expected:
            if key in tif.name:
                found[key] = tif

    missing = [k for k, v in found.items() if v is None]
    if missing:
        raise FileNotFoundError(f"Missing rasters for: {missing}")

    # Also track optional when not used (handy for logging)
    if not use_aux:
        for k in optional:
            found[k] = next((t for t in event_path.glob("*.tif") if k in t.name), None)

    return found


def compute_polygon_hit_metrics_by_size(
    shapefile_path: Path,
    raster_path: Path,
    size_field: str = "size",
    classes: Sequence[int] = (2, 3, 4),
    min_fraction: float = 0.5,
):
    """
    For each size in `classes`, compute (hits, total, rate) where a polygon is a 'hit' if
    fraction(pred==1 within polygon, ignoring nodata) >= min_fraction.
    Returns dict: {size: (hits, total, rate)}. If no polygons for a size -> (0,0,0.0).
    """
    import warnings

    if shapefile_path is None or not shapefile_path.exists():
        warnings.warn("Shapefile path missing; skipping polygon metrics by size.")
        return {c: (0, 0, 0.0) for c in classes}

    with rasterio.open(raster_path) as src:
        pred = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata if src.nodata is not None else 255
        height, width = src.height, src.width

    valid = pred != nodata
    pred_pos = (pred == 1)

    gdf = gpd.read_file(shapefile_path)
    if gdf.empty:
        warnings.warn("Shapefile has no polygons; skipping polygon metrics by size.")
        return {c: (0, 0, 0.0) for c in classes}
    if gdf.crs is None:
        raise ValueError("Shapefile has no CRS. Define or reproject it.")
    if crs is None:
        raise ValueError("Prediction raster has no CRS. Cannot align polygons.")

    gdf = gdf.to_crs(crs)
    gdf = gdf[gdf.geometry.notnull() & gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]

    if size_field not in gdf.columns:
        warnings.warn(f"'{size_field}' field not found; skipping polygon metrics by size.")
        return {c: (0, 0, 0.0) for c in classes}

    # Normalize/clean the size field
    sizes_series = pd.to_numeric(gdf[size_field], errors="coerce").astype("Int64")
    gdf = gdf.loc[sizes_series.notna()].copy()
    gdf[size_field] = sizes_series.loc[gdf.index].astype(int)

    # Keep only requested classes
    gdf = gdf[gdf[size_field].isin(classes)]
    if gdf.empty:
        return {c: (0, 0, 0.0) for c in classes}

    gdf = gdf.reset_index(drop=True)
    sizes_arr = gdf[size_field].to_numpy(dtype=np.int32)

    # Rasterize polygon IDs
    shapes = [(geom, idx + 1) for idx, geom in enumerate(gdf.geometry)]
    poly_ids = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.int32,
        all_touched=False,
    )

    n_polys = len(shapes)
    ids_valid = poly_ids[valid]
    total_per_poly = np.bincount(ids_valid.ravel(), minlength=n_polys + 1)
    ids_pred_pos = poly_ids[valid & pred_pos]
    pos_per_poly = np.bincount(ids_pred_pos.ravel(), minlength=n_polys + 1)
    total_per_poly = total_per_poly[1:]
    pos_per_poly = pos_per_poly[1:]

    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(total_per_poly > 0, pos_per_poly / total_per_poly, 0.0)
    hit_mask = frac >= float(min_fraction)

    out = {}
    for cls in classes:
        sel = sizes_arr == cls
        total = int(sel.sum())
        hits = int((hit_mask & sel).sum())
        rate = (hits / total) if total > 0 else 0.0
        out[cls] = (hits, total, rate)
    return out


def extract_patches_stack(
    raster_paths_in_order: Sequence[Path],
    patch_size: Tuple[int, int],
    stride: Tuple[int, int],
    region_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], dict]:
    """
    Load bands in the given order (channels-first), optionally mask outside region as NaN,
    and return a 5D window tensor via view_as_windows plus grid/original shapes and raster meta.
    """
    with rasterio.open(raster_paths_in_order[0]) as src0:
        meta = src0.meta.copy()
        height, width = src0.shape

    raster_array = []
    for path in raster_paths_in_order:
        with rasterio.open(path) as src:
            img = src.read(1).astype(np.float32)
            if src.nodata is not None:
                img = np.where(img == src.nodata, np.nan, img)
            if region_mask is not None:
                if region_mask.shape != img.shape:
                    raise ValueError(f"region_mask shape {region_mask.shape} != raster {img.shape}")
                img = np.where(region_mask == 1, img, np.nan)
            raster_array.append(img)

    raster_array = np.stack(raster_array, axis=0)  # (C, H, W)
    num_bands = raster_array.shape[0]
    ph, pw = patch_size
    sh, sw = stride

    patches = view_as_windows(raster_array, (num_bands, ph, pw), (1, sh, sw))
    grid_shape = patches.shape[1:3]  # (rows, cols)
    return patches, grid_shape, (height, width), meta


# ---------------- #
# Stitching / Merg #
# ---------------- #
def merge_patches(
    patch_outputs: List[np.ndarray],
    grid_shape: Tuple[int, int],
    patch_size: Tuple[int, int],
    stride: Tuple[int, int],
    original_shape: Tuple[int, int],
    valid_indices: Sequence[int],
    best_threshold: float,
    mode: str = "center_crop",
) -> np.ndarray:
    """
    Merge per-patch probabilities into a full map with different blending modes.
    Invalid pixels set to 255 (nodata).
    - "none": adjacent-only stitching (no overlap). Writes the top-left stride-sized crop of each patch.
    """
    patch_outputs = np.concatenate(patch_outputs, axis=0)  # (N_valid, ph, pw)
    ph, pw = patch_size
    rows, cols = grid_shape
    H, W = original_shape

    valid_grid = np.zeros((rows, cols), dtype=bool)
    for idx in valid_indices:
        r, c = divmod(idx, cols)
        valid_grid[r, c] = True

    # --- "none" mode: adjacent-only stitching, no blending, no overlap ---
    if mode == "none":
        full = np.full((H, W), 255, dtype=np.float32)  # nodata by default
        written = np.zeros((H, W), dtype=bool)
        patch_idx = 0
        v_idx = 0
        for i in range(rows):
            for j in range(cols):
                y0, x0 = i * stride[0], j * stride[1]
                if patch_idx in valid_indices:
                    patch = patch_outputs[v_idx]

                    # Write only the stride-sized top-left crop so adjacent tiles abut with no overlap.
                    h_write = min(stride[0], H - y0)
                    w_write = min(stride[1], W - x0)
                    crop = patch[:h_write, :w_write]

                    tgt = full[y0:y0 + h_write, x0:x0 + w_write]
                    mask = ~written[y0:y0 + h_write, x0:x0 + w_write]
                    tgt[mask] = crop[mask]
                    full[y0:y0 + h_write, x0:x0 + w_write] = tgt
                    written[y0:y0 + h_write, x0:x0 + w_write] = True

                    v_idx += 1
                patch_idx += 1
        return full
    # --- end "none" ---

    if mode == "max":
        full = np.full((H, W), -np.inf, dtype=np.float32)
    elif mode == "min":
        full = np.full((H, W), np.inf, dtype=np.float32)
    else:
        full = np.zeros((H, W), dtype=np.float32)

    count_map = np.zeros((H, W), dtype=np.float32)

    if mode == "gaussian":
        y = np.linspace(-1, 1, ph)
        x = np.linspace(-1, 1, pw)
        xv, yv = np.meshgrid(x, y)
        weights = np.exp(-(xv**2 + yv**2) / 0.5).astype(np.float32)
    else:
        weights = None

    patch_idx = 0
    v_idx = 0
    for i in range(rows):
        for j in range(cols):
            y0, x0 = i * stride[0], j * stride[1]
            if patch_idx in valid_indices:
                patch = patch_outputs[v_idx]

                if mode == "center_crop":
                    my, mx = ph // 4, pw // 4
                    top_ok = (i > 0 and valid_grid[i - 1, j])
                    bot_ok = (i < rows - 1 and valid_grid[i + 1, j])
                    left_ok = (j > 0 and valid_grid[i, j - 1])
                    right_ok = (j < cols - 1 and valid_grid[i, j + 1])

                    tc = my if top_ok else 0
                    bc = my if bot_ok else 0
                    lc = mx if left_ok else 0
                    rc = mx if right_ok else 0

                    cropped = patch[tc: ph - bc, lc: pw - rc]
                    yy0, xx0 = y0 + tc, x0 + lc
                    yy1, xx1 = y0 + ph - bc, x0 + pw - rc
                    full[yy0:yy1, xx0:xx1] = cropped
                    count_map[yy0:yy1, xx0:xx1] += 1

                elif mode == "gaussian":
                    full[y0:y0 + ph, x0:x0 + pw] += patch * weights
                    count_map[y0:y0 + ph, x0:x0 + pw] += weights

                elif mode == "max":
                    region = full[y0:y0 + ph, x0:x0 + pw]
                    full[y0:y0 + ph, x0:x0 + pw] = np.maximum(region, patch)
                    count_map[y0:y0 + ph, x0:x0 + pw] += 1

                elif mode == "min":
                    region = full[y0:y0 + ph, x0:x0 + pw]
                    mask = count_map[y0:y0 + ph, x0:x0 + pw] == 0
                    region[mask] = patch[mask]
                    full[y0:y0 + ph, x0:x0 + pw] = np.minimum(region, patch)
                    count_map[y0:y0 + ph, x0:x0 + pw] += 1

                else:  # mean
                    full[y0:y0 + ph, x0:x0 + pw] += patch
                    count_map[y0:y0 + ph, x0:x0 + pw] += 1

                v_idx += 1
            patch_idx += 1

    valid = count_map > 0
    if mode in ["mean", "gaussian"]:
        full[valid] /= count_map[valid]
    full[~valid] = 255  # nodata
    return full


# ------------------------- #
# Dataset & preprocessing   #
# ------------------------- #
class AvalancheDataset(Dataset):
    """
    Patches are (C,H,W) with C = 4 (+2 aux if enabled).
    Applies stats (z-score) with per-channel fill (sentinel z) and yields:
        pre_z (2,H,W), post_z (2,H,W), aux_z (2,H,W or empty), label (1,H,W)
    """
    def __init__(
        self,
        patches: np.ndarray,
        region_patches: np.ndarray,
        gt_patches: np.ndarray,
        stats: dict,
        nan_percent: float,
        use_aux: bool,
    ):
        self.patches = patches  # (N, C, H, W)
        self.region_patches = region_patches  # (N, H, W)
        self.gt_patches = gt_patches  # (N, H, W)
        self.stats = stats
        self.nan_percent = float(nan_percent)
        self.use_aux = bool(use_aux)

        self.valid_indices: List[int] = []
        for idx in range(len(self.patches)):
            region_patch = self.region_patches[idx]
            inside_region = np.all(region_patch == 1)

            sar_patch = self.patches[idx][:4]  # SAR only for validity
            nan_mask = np.isnan(sar_patch).any(axis=0)  # union over 4 bands
            nan_fraction = float(nan_mask.mean())

            if inside_region or (nan_fraction < self.nan_percent):
                self.valid_indices.append(idx)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        raw = self.patches[idx].copy()  # (C,H,W)

        # Clean SAR (first 4 channels) BEFORE torch conversion
        sar = raw[:4]
        sar = np.where(
            (~np.isfinite(sar)) | (sar < -40.0) | (sar > 20.0),
            np.nan,
            sar,
        )
        raw[:4] = sar

        gt_patch = self.gt_patches[idx].astype(np.float32)  # (H,W)

        patch = torch.tensor(raw, dtype=torch.float32)
        pre = patch[:2]
        post = patch[2:4]

        if self.use_aux:
            lia = patch[4]
            slp = patch[5]
            aux = torch.stack([lia, slp], dim=0)
        else:
            aux = torch.empty(0)  # placeholder

        # Stats tensors
        mean_img = self.stats["img_mean"].view(-1, 1, 1)
        std_img = self.stats["img_std"].view(-1, 1, 1)
        fill_img = self.stats["sentinel_z_img"].view(-1, 1, 1)

        pre_z = (pre - mean_img) / std_img
        post_z = (post - mean_img) / std_img

        # Fill SAR NaNs with sentinel z
        for c in range(pre_z.shape[0]):
            pre_z[c][~torch.isfinite(pre_z[c])] = fill_img[c]
            post_z[c][~torch.isfinite(post_z[c])] = fill_img[c]

        if self.use_aux:
            mean_aux = self.stats["aux_mean"].view(-1, 1, 1)
            std_aux = self.stats["aux_std"].view(-1, 1, 1)
            fill_aux = self.stats["sentinel_z_aux"].view(-1, 1, 1)

            aux_z = (aux - mean_aux) / std_aux
            for c in range(aux_z.shape[0]):
                aux_z[c][~torch.isfinite(aux_z[c])] = fill_aux[c]
        else:
            # keep shape consistent when collated: provide (0,H,W) tensor
            H, W = pre_z.shape[1:]
            aux_z = torch.empty((0, H, W), dtype=torch.float32)

        label = (torch.tensor(gt_patch) == 1).float().unsqueeze(0)  # (1,H,W)
        return pre_z, post_z, aux_z, label

    def get_valid_indices(self) -> List[int]:
        return self.valid_indices


# -------------------- #
# Polygon hit metric   #
# -------------------- #
def compute_polygon_hit_metric(
    shapefile_path: Path, raster_path: Path, min_fraction: float = 0.5
) -> Tuple[int, int]:
    """
    Counts polygons detected by the prediction raster (uint8 {0,1}, nodata=255).
    Detection: fraction(pred==1 within polygon, ignoring nodata) >= min_fraction.
    Returns (hits, total_polygons).
    """
    if shapefile_path is None or not shapefile_path.exists():
        logging.warning("Shapefile path is None or missing; skipping polygon metric.")
        return 0, 0

    with rasterio.open(raster_path) as src:
        pred = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata if src.nodata is not None else 255
        height, width = src.height, src.width

    valid = pred != nodata
    pred_pos = (pred == 1)

    gdf = gpd.read_file(shapefile_path)
    if gdf.empty:
        logging.warning("Shapefile has no polygons; skipping polygon metric.")
        return 0, 0
    if gdf.crs is None:
        raise ValueError("Shapefile has no CRS. Define or reproject it.")
    if crs is None:
        raise ValueError("Prediction raster has no CRS. Cannot align polygons.")
    gdf = gdf.to_crs(crs)
    gdf = gdf[gdf.geometry.notnull() & gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if gdf.empty:
        logging.warning("No polygonal geometries after filtering; skipping polygon metric.")
        return 0, 0

    shapes = [(geom, idx + 1) for idx, geom in enumerate(gdf.geometry)]
    poly_ids = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.int32,
        all_touched=False,
    )

    n_polys = len(shapes)
    ids_valid = poly_ids[valid]
    total_per_poly = np.bincount(ids_valid.ravel(), minlength=n_polys + 1)
    ids_pred_pos = poly_ids[valid & pred_pos]
    pos_per_poly = np.bincount(ids_pred_pos.ravel(), minlength=n_polys + 1)

    total_per_poly = total_per_poly[1:]
    pos_per_poly = pos_per_poly[1:]

    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(total_per_poly > 0, pos_per_poly / total_per_poly, 0.0)
    hits = int((frac >= min_fraction).sum())
    return hits, n_polys


# -------------- #
# Main pipeline  #
# -------------- #
def run_inference(
    event_path: Path,
    rasters_dir: Path,
    stats_path: Path,
    model_ckpt: Path,
    patch_size: Tuple[int, int],
    stride: Tuple[int, int],
    use_aux: bool,
    nan_percent: float,
    shapefile_path: Optional[Path],
    min_fraction_inside: float,
    modes: Sequence[str],
):
    rasters_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Using event at: {event_path}")
    logging.info(f"Aux enabled: {use_aux}")

    # Load stats
    stats_json = json.load(open(stats_path))
    stats_tensor = {k: torch.tensor(v) for k, v in stats_json.items()}

    # Rasters
    paths = get_raster_paths(event_path, use_aux=use_aux)

    # Ground truth
    gt_path = next(event_path.glob("*_GT.tif"))
    with rasterio.open(gt_path) as gt_src:
        gt_mask = gt_src.read(1).astype(np.uint8)
    shutil.copy(gt_path, rasters_dir / "ground_truth.tif")

    # Region mask from SAR valid pixels only
    with rasterio.open(paths["preVH"]) as src_vh_pre, \
         rasterio.open(paths["preVV"]) as src_vv_pre, \
         rasterio.open(paths["postVH"]) as src_vh_post, \
         rasterio.open(paths["postVV"]) as src_vv_post:

        vh_pre = src_vh_pre.read(1)
        vv_pre = src_vv_pre.read(1)
        vh_post = src_vh_post.read(1)
        vv_post = src_vv_post.read(1)

        valid_mask = (
            ~np.isnan(vh_pre) &
            ~np.isnan(vv_pre) &
            ~np.isnan(vh_post) &
            ~np.isnan(vv_post)
        )
        region_mask = binary_fill_holes(valid_mask).astype(np.uint8)

    # Band order for extraction
    ordered_paths = [paths["preVH"], paths["preVV"], paths["postVH"], paths["postVV"]]
    if use_aux:
        ordered_paths += [paths["LIA"], paths["SLP"]]

    # Tile extraction
    stacked, grid_shape, original_shape, meta = extract_patches_stack(
        ordered_paths, patch_size, stride, region_mask
    )
    patches = stacked.reshape(-1, *stacked.shape[3:])  # (N, C, H, W)

    # Region & GT patches
    region_patches = view_as_windows(region_mask, patch_size, stride).reshape(-1, *patch_size)
    gt_patches = view_as_windows(gt_mask, patch_size, stride).reshape(-1, *patch_size)

    # Dataset & loader
    dataset = AvalancheDataset(
        patches, region_patches, gt_patches, stats_tensor,
        nan_percent=nan_percent, use_aux=use_aux
    )
    valid_indices = dataset.get_valid_indices()
    subset = Subset(dataset, valid_indices)

    # Model
    model = ChangeDetectionSwinUNet(img_size=patch_size[0], use_aux=use_aux)
    checkpoint = torch.load(model_ckpt, weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_thr = float(checkpoint.get("best_threshold", 0.5))
    logging.info(f"Decision threshold (from ckpt): {best_thr:.4f}")

    # Metrics (binary)
    metrics = {
        "recall":   Recall(task="binary", num_classes=1, threshold=best_thr).to(device),
        "precision": Precision(task="binary", num_classes=1, threshold=best_thr).to(device),
        "f1":       F1Score(task="binary", num_classes=1, threshold=best_thr).to(device),
        "iou":      JaccardIndex(task="binary", num_classes=1, threshold=best_thr).to(device),
    }

    loader = DataLoader(subset, batch_size=32, shuffle=False)

    # ---- sanity check: patch counts ----
    total_grid_patches = grid_shape[0] * grid_shape[1]
    print(f"[Sanity] grid_shape: {grid_shape[0]} x {grid_shape[1]} -> {total_grid_patches} tiles")
    print(f"[Sanity] extracted patches tensor: N={patches.shape[0]}, C={patches.shape[1]}, H={patches.shape[2]}, W={patches.shape[3]}")
    print(f"[Sanity] valid patches (used): {len(subset)} / {len(dataset)} (filtered out: {len(dataset) - len(subset)})")
    print(f"[Sanity] decision threshold: {best_thr:.4f}")

    patch_outputs = []

    # ---- Inference over valid patches ----
    with torch.no_grad():
        for pre, post, aux, gt in tqdm(loader, desc="Inferencing", ncols=100):
            pre, post, gt = pre.to(device), post.to(device), gt.to(device)
            if use_aux:
                aux = aux.to(device)
                logits = model(pre, post, aux)
            else:
                logits = model(pre, post)

            # 1) raw probabilities (keep these for stitching)
            prb = torch.sigmoid(logits)  # (B,1,H,W)
            patch_outputs.append(prb.squeeze(1).cpu().numpy())

            # 2) binarize for per-patch metrics
            bin_mask = (prb > best_thr).float()

            # 3) optional morphology to match test.py behavior (binary domain)
            bin_mask = morph_close(bin_mask, kernel_size=3, iterations=1)

            # 4) update metrics with the post-processed binary mask
            for m in metrics.values():
                m.update(bin_mask, gt)

    print("\nSanity check — metrics on raw patch-wise predictions (after bin+close, before blending):")
    for n, m in metrics.items():
        print(f"{n.capitalize()}: {m.compute().item():.4f}")

    # ----------------------- #
    # Per-mode reconstruction #
    # ----------------------- #
    for mode in modes:
        print(f"\n=== Testing blending mode: {mode} ===")
        reconstructed = merge_patches(
            patch_outputs, grid_shape, patch_size, stride,
            original_shape, valid_indices, best_thr, mode=mode
        )

        # Post-process for metrics
        nodata_val = 255
        valid_mask_full = reconstructed != nodata_val

        # Threshold THEN morphology on binary
        bin_full = np.zeros_like(reconstructed, dtype=np.uint8)
        bin_full[valid_mask_full] = (reconstructed[valid_mask_full] > best_thr).astype(np.uint8)

        bin_tensor = torch.tensor(bin_full, dtype=torch.float32, device=device)
        bin_tensor = morph_close(bin_tensor, kernel_size=3, iterations=1)

        # Vectorize valid pixels
        pred_vec = bin_tensor[torch.tensor(valid_mask_full, dtype=torch.bool, device=device)].unsqueeze(1)
        gt_vec = torch.tensor(gt_mask[valid_mask_full], dtype=torch.int64, device=device).unsqueeze(1)

        # Compute metrics
        for m in metrics.values():
            m.reset()
            m.update(pred_vec, gt_vec)
        final_metrics = {name: m.compute().item() for name, m in metrics.items()}
        for k, v in final_metrics.items():
            print(f"{k.capitalize()}: {v:.4f}")

        # Write binary GeoTIFF
        binary_mask = np.full(reconstructed.shape, nodata_val, dtype="uint8")
        binary_mask[valid_mask_full] = (reconstructed[valid_mask_full] > best_thr).astype("uint8")

        out_path = rasters_dir / f"pred_{mode}.tif"
        meta_out = meta.copy()
        meta_out.update({"count": 1, "dtype": "uint8", "nodata": nodata_val})
        with rasterio.open(out_path, "w", **meta_out) as dst:
            dst.write(binary_mask, 1)

        # Polygon-level metrics by avalanche size (2,3,4)
        if shapefile_path is not None and shapefile_path.exists():
            by_size = compute_polygon_hit_metrics_by_size(
                shapefile_path, out_path, size_field="size",
                classes=(2, 3, 4), min_fraction=min_fraction_inside
            )

            # ---- totals across sizes 2–4
            tot_hits = sum(v[0] for v in by_size.values())
            tot_total = sum(v[1] for v in by_size.values())
            tot_rate = (tot_hits / tot_total) if tot_total > 0 else 0.0
            print(f"[{mode}] Total hit rate (sizes 2–4, ≥{min_fraction_inside*100:.0f}% inside): "
                  f"{tot_hits}/{tot_total} ({tot_rate:.2%})")

            for cls in (2, 3, 4):
                hits, total, rate = by_size[cls]
                print(f"[{mode}] Hit rate size {cls} (≥{min_fraction_inside*100:.0f}% inside): "
                      f"{hits}/{total} ({rate:.2%})")

    print("\nDone.")


# -------- #
#   CLI    #
# -------- #
def parse_args():
    p = argparse.ArgumentParser(description="Avalanche CD inference with optional aux (LIA, SLP).")
    p.add_argument("--event-path", type=Path, default=DEFAULT_EVENT_PATH, help="Folder with rasters.")
    p.add_argument("--rasters-dir", type=Path, default=DEFAULT_RASTERS_DIR, help="Output directory.")
    p.add_argument("--stats-path", type=Path, default=DEFAULT_STATS_PATH, help="Stats JSON.")
    p.add_argument("--model-ckpt", type=Path, default=DEFAULT_MODEL_CKPT, help="Checkpoint path.")
    p.add_argument("--patch-size", type=int, nargs=2, default=DEFAULT_PATCH_SIZE, metavar=("H", "W"))
    p.add_argument("--stride", type=int, nargs=2, default=DEFAULT_STRIDE, metavar=("H", "W"))
    p.add_argument("--use-aux", action="store_true", help="Use LIA & SLP auxiliary channels.")
    p.add_argument("--nan-percent", type=float, default=DEFAULT_NAN_PERCENT,
                   help="Accept patch if inside region OR NaN fraction < this.")
    p.add_argument("--shapefile", type=Path, default=DEFAULT_GPKG, help="Polygons for hit metric.")
    p.add_argument("--min-fraction-inside", type=float, default=DEFAULT_MIN_FRACTION_INSIDE,
                   help="Min fraction of predicted pixels inside polygon to count as hit.")
    p.add_argument("--modes", type=str, nargs="+", default=DEFAULT_MODES,
                   choices=["none", "mean", "max", "min", "gaussian", "center_crop"],
                   help="Blending modes to evaluate.")
    p.add_argument("--loglevel", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()), format="%(levelname)s: %(message)s")
    run_inference(
        event_path=args.event_path,
        rasters_dir=args.rasters_dir,
        stats_path=args.stats_path,
        model_ckpt=args.model_ckpt,
        patch_size=tuple(args.patch_size),
        stride=tuple(args.stride),
        use_aux=args.use_aux,
        nan_percent=args.nan_percent,
        shapefile_path=args.shapefile,
        min_fraction_inside=args.min_fraction_inside,
        modes=args.modes,
    )


if __name__ == "__main__":
    main()