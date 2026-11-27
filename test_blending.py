#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Avalanche CD – tiled inference + optional blending, with optional aux channels (LIA, SLP).

Output strategy:
- Confusion results are vectorized (TP/FN/FP polygons; TN dropped as background) and
  written as layers into ONE GeoPackage per model run: <rasters_dir>/<model_name>/confusion.gpkg
  with one layer per blending mode (confusion_<mode>). This fixes categorical-raster
  rendering artifacts (orange fringes, oversampling/reprojection averaging) because
  vector polygons are not resampled, and keeps the file count low.
- Per-mode probability/binary/confusion GeoTIFFs are OPTIONAL (--save-prob, --save-tif).
- Polygon hit metrics are computed directly from the in-memory prediction, so no
  intermediate binary raster is written to disk just to score it.

Encoding of the confusion layers: 0=TN, 1=TP, 2=FN, 3=FP (255=nodata in raster form).

Shared tiling/dataset/merge logic lives in ``utils.tiling``.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.features import rasterize, shapes
from skimage.util.shape import view_as_windows
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import F1Score, JaccardIndex, Precision, Recall
from tqdm import tqdm

from models.build import build_model
from utils.morph import morph_close
from utils.tiling import (
    NODATA,
    TiledInferenceDataset,
    build_region_mask,
    extract_patches_stack,
    find_raster_paths,
    merge_patches,
    ordered_band_paths,
)

# ------------------------------- #
# Defaults (can be overridden CLI)
# ------------------------------- #
DEFAULT_PATCH_SIZE = (128, 128)
DEFAULT_STRIDE = (64, 64)
DEFAULT_RASTERS_DIR = Path("output/test_blending")
DEFAULT_EVENT_PATH = Path("/home/jovyan/nfs/mgatti/datasets/Avalanches/AvalCD/Tromso_20241220/")
DEFAULT_STATS_PATH = Path("/home/jovyan/nfs/mgatti/datasets/Avalanches/patches/128/stats.json")
DEFAULT_MODEL_CKPT = Path("/home/jovyan/nfs/mgatti/python/avalanches/exp/swinunet_128_F2/best_model.pth")
DEFAULT_GPKG = Path("/home/jovyan/nfs/mgatti/datasets/Avalanches/AvalCD/Tromso_20241220/Tromso_20241220_GT.gpkg")
DEFAULT_MIN_FRACTION_INSIDE = 0.5
DEFAULT_NAN_PERCENT = 0.8
DEFAULT_MODES = ["none", "mean", "max", "min", "gaussian", "center_crop"]

CLASS_LABELS = {0: "TN", 1: "TP", 2: "FN", 3: "FP"}


# ------------------------------- #
# Confusion vectorization helpers
# ------------------------------- #
def build_confusion(
    gt_mask: np.ndarray,
    binary_mask: np.ndarray,
    valid_mask_full: np.ndarray,
    nodata_val: int = NODATA,
) -> np.ndarray:
    """
    Build a confusion array from GT and a binary prediction.
    Encoding: 0=TN, 1=TP, 2=FN, 3=FP, nodata_val=invalid/outside region.
    """
    conf = np.full(gt_mask.shape, nodata_val, dtype=np.uint8)
    gt1 = gt_mask == 1
    pr1 = binary_mask == 1
    conf[valid_mask_full & ~gt1 & ~pr1] = 0  # TN
    conf[valid_mask_full &  gt1 &  pr1] = 1  # TP
    conf[valid_mask_full &  gt1 & ~pr1] = 2  # FN
    conf[valid_mask_full & ~gt1 &  pr1] = 3  # FP
    return conf


def confusion_to_gdf(
    conf: np.ndarray,
    transform,
    crs,
    drop_tn: bool = True,
    dissolve: bool = True,
) -> gpd.GeoDataFrame:
    """
    Vectorize the confusion array into polygons with 'class' and 'label' columns.
    TN is dropped by default (background you render transparent anyway).
    dissolve=True merges each class into a single (multi)polygon feature.
    Returns an (possibly empty) GeoDataFrame in the raster CRS.
    """
    keep = (1, 2, 3) if drop_tn else (0, 1, 2, 3)
    mask = np.isin(conf, keep)
    records = [
        {"geometry": geom, "properties": {"class": int(v)}}
        for geom, v in shapes(conf, mask=mask, transform=transform, connectivity=4)
    ]
    if not records:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    gdf = gpd.GeoDataFrame.from_features(records, crs=crs)
    if dissolve:
        gdf = gdf.dissolve(by="class", as_index=False)
    gdf["label"] = gdf["class"].map(CLASS_LABELS)
    return gdf


def write_layer(gdf: gpd.GeoDataFrame, gpkg_path: Path, layer: str) -> None:
    """
    Append a layer to a GeoPackage. Creates the file on the first write and
    appends subsequent layers (requires a recent geopandas + pyogrio/fiona).
    """
    gdf.to_file(
        gpkg_path,
        layer=layer,
        driver="GPKG",
        mode=("a" if Path(gpkg_path).exists() else "w"),
    )


def compute_polygon_hit_metrics_by_size(
    shapefile_path: Path,
    pred: np.ndarray,
    transform,
    crs,
    nodata: int = NODATA,
    size_field: str = "size",
    classes: Sequence[int] = (2, 3, 4),
    min_fraction: float = 0.5,
):
    """
    For each size in `classes`, compute (hits, total, rate) where a polygon is a 'hit' if
    fraction(pred==1 within polygon, ignoring nodata) >= min_fraction.
    Works directly on the in-memory prediction array (no raster file needed).
    Returns dict: {size: (hits, total, rate)}. If no polygons for a size -> (0,0,0.0).
    """
    import warnings

    if shapefile_path is None or not Path(shapefile_path).exists():
        warnings.warn("Shapefile path missing; skipping polygon metrics by size.")
        return {c: (0, 0, 0.0) for c in classes}

    if crs is None:
        raise ValueError("Prediction raster has no CRS. Cannot align polygons.")

    height, width = pred.shape
    valid = pred != nodata
    pred_pos = (pred == 1)

    gdf = gpd.read_file(shapefile_path)
    if gdf.empty:
        warnings.warn("Shapefile has no polygons; skipping polygon metrics by size.")
        return {c: (0, 0, 0.0) for c in classes}
    if gdf.crs is None:
        raise ValueError("Shapefile has no CRS. Define or reproject it.")

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
    shapes_iter = [(geom, idx + 1) for idx, geom in enumerate(gdf.geometry)]
    poly_ids = rasterize(
        shapes=shapes_iter,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.int32,
        all_touched=False,
    )

    n_polys = len(shapes_iter)
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


def compute_polygon_hit_metric(
    shapefile_path: Path, raster_path: Path, min_fraction: float = 0.5
) -> Tuple[int, int]:
    """
    Counts polygons detected by the prediction raster (uint8 {0,1}, nodata=255).
    Detection: fraction(pred==1 within polygon, ignoring nodata) >= min_fraction.
    Returns (hits, total_polygons).
    (Kept for standalone use; the main pipeline uses the in-memory by-size variant.)
    """
    if shapefile_path is None or not shapefile_path.exists():
        logging.warning("Shapefile path is None or missing; skipping polygon metric.")
        return 0, 0

    with rasterio.open(raster_path) as src:
        pred = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata if src.nodata is not None else NODATA
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

    shapes_iter = [(geom, idx + 1) for idx, geom in enumerate(gdf.geometry)]
    poly_ids = rasterize(
        shapes=shapes_iter,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.int32,
        all_touched=False,
    )

    n_polys = len(shapes_iter)
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
    save_prob: bool = False,
    save_tif: bool = False,
    copy_gt_tif: bool = True,
):
    # Model name = the checkpoint's parent folder
    # (e.g. ".../exp/swinunet_128_F2/best_model.pth" -> "swinunet_128_F2")
    model_name = model_ckpt.parent.name
    rasters_dir = rasters_dir / model_name
    rasters_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Model name (from ckpt): {model_name}")
    logging.info(f"Writing outputs to: {rasters_dir}")
    logging.info(f"Using event at: {event_path}")
    logging.info(f"Aux enabled: {use_aux}")

    # Single GeoPackage that will hold all vector confusion layers.
    gpkg_path = rasters_dir / "confusion.gpkg"
    if gpkg_path.exists():
        gpkg_path.unlink()  # start clean so re-runs don't stack layers
    logging.info(f"Confusion GeoPackage: {gpkg_path}")

    # Load stats
    stats_json = json.load(open(stats_path))
    stats_tensor = {k: torch.tensor(v) for k, v in stats_json.items()}

    # Rasters (SLP expected to already exist in the folder)
    paths = find_raster_paths(event_path, use_aux=use_aux)

    # Ground truth
    gt_path = next(event_path.glob("*_GT.tif"))
    with rasterio.open(gt_path) as gt_src:
        gt_mask = gt_src.read(1).astype(np.uint8)
    if copy_gt_tif:
        shutil.copy(gt_path, rasters_dir / "ground_truth.tif")

    # Band order for extraction
    ordered_paths = ordered_band_paths(paths, use_aux)

    # Region mask from SAR valid pixels only
    region_mask = build_region_mask(ordered_paths[:4])

    # Tile extraction
    stacked, grid_shape, original_shape, meta = extract_patches_stack(
        ordered_paths, patch_size, stride, region_mask
    )
    patches = stacked.reshape(-1, *stacked.shape[3:])  # (N, C, H, W)

    # Geospatial reference (used for both vectorization and hit metrics)
    out_transform = meta["transform"]
    out_crs = meta["crs"]

    # Region & GT patches
    region_patches = view_as_windows(region_mask, patch_size, stride).reshape(-1, *patch_size)
    gt_patches = view_as_windows(gt_mask, patch_size, stride).reshape(-1, *patch_size)

    # Dataset & loader
    dataset = TiledInferenceDataset(
        patches, region_patches, stats_tensor,
        use_aux=use_aux, nan_percent=nan_percent, gt_patches=gt_patches,
    )
    valid_indices = dataset.get_valid_indices()
    subset = Subset(dataset, valid_indices)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("swinunet", patch_size=patch_size[0], use_aux=use_aux, device=device)
    checkpoint = torch.load(model_ckpt, weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
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
        for batch in tqdm(loader, desc="Inferencing", ncols=100):
            pre = batch["pre"].to(device)
            post = batch["post"].to(device)
            gt = batch["label"].to(device)
            if use_aux:
                aux = batch["aux"].to(device)
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
            original_shape, valid_indices, mode=mode
        )

        nodata_val = NODATA
        valid_mask_full = reconstructed != nodata_val

        # Binary prediction (threshold; morphology used only for metrics, matching original)
        binary_mask = np.full(reconstructed.shape, nodata_val, dtype="uint8")
        binary_mask[valid_mask_full] = (reconstructed[valid_mask_full] > best_thr).astype("uint8")

        # --- metrics: threshold THEN morphology on binary ---
        bin_full = np.zeros_like(reconstructed, dtype=np.uint8)
        bin_full[valid_mask_full] = (reconstructed[valid_mask_full] > best_thr).astype(np.uint8)
        bin_tensor = torch.tensor(bin_full, dtype=torch.float32, device=device)
        bin_tensor = morph_close(bin_tensor, kernel_size=3, iterations=1)
        pred_vec = bin_tensor[torch.tensor(valid_mask_full, dtype=torch.bool, device=device)].unsqueeze(1)
        gt_vec = torch.tensor(gt_mask[valid_mask_full], dtype=torch.int64, device=device).unsqueeze(1)
        for m in metrics.values():
            m.reset()
            m.update(pred_vec, gt_vec)
        final_metrics = {name: m.compute().item() for name, m in metrics.items()}
        for k, v in final_metrics.items():
            print(f"{k.capitalize()}: {v:.4f}")

        # --- confusion -> vector layer in the shared GeoPackage ---
        conf = build_confusion(gt_mask, binary_mask, valid_mask_full, nodata_val)
        gdf_conf = confusion_to_gdf(conf, out_transform, out_crs, drop_tn=True, dissolve=True)
        if not gdf_conf.empty:
            write_layer(gdf_conf, gpkg_path, f"confusion_{mode}")
            print(f"[{mode}] wrote layer confusion_{mode} -> {gpkg_path.name}")
        else:
            print(f"[{mode}] no TP/FN/FP pixels; skipped vector layer")

        # --- optional rasters (off by default to keep file count low) ---
        if save_prob:
            prob_out = np.full(reconstructed.shape, nodata_val, dtype=np.float32)
            prob_out[valid_mask_full] = reconstructed[valid_mask_full].astype(np.float32)
            meta_prob = meta.copy()
            meta_prob.update({"count": 1, "dtype": "float32", "nodata": float(nodata_val)})
            with rasterio.open(rasters_dir / f"pred_{mode}_prob.tif", "w", **meta_prob) as dst:
                dst.write(prob_out, 1)

        if save_tif:
            meta_out = meta.copy()
            meta_out.update({"count": 1, "dtype": "uint8", "nodata": nodata_val})
            with rasterio.open(rasters_dir / f"pred_{mode}.tif", "w", **meta_out) as dst:
                dst.write(binary_mask, 1)
            with rasterio.open(rasters_dir / f"confusion_{mode}.tif", "w", **meta_out) as dst:
                dst.write(conf, 1)
            print(f"[{mode}] wrote pred_{mode}.tif and confusion_{mode}.tif")

        # --- polygon-level metrics by avalanche size (2,3,4), straight from memory ---
        if shapefile_path is not None and Path(shapefile_path).exists():
            by_size = compute_polygon_hit_metrics_by_size(
                shapefile_path,
                pred=binary_mask,
                transform=out_transform,
                crs=out_crs,
                nodata=nodata_val,
                size_field="size",
                min_fraction=min_fraction_inside,
            )

            tot_hits = sum(v[0] for v in by_size.values())
            tot_total = sum(v[1] for v in by_size.values())
            tot_rate = (tot_hits / tot_total) if tot_total > 0 else 0.0
            print(f"[{mode}] Total hit rate (sizes 2–4, ≥{min_fraction_inside*100:.0f}% inside): "
                  f"{tot_hits}/{tot_total} ({tot_rate:.2%})")

            for cls in by_size:
                hits, total, rate = by_size[cls]
                print(f"[{mode}] Hit rate size {cls} (≥{min_fraction_inside*100:.0f}% inside): "
                      f"{hits}/{total} ({rate:.2%})")
        else:
            print(f"[{mode}] Hit rate not reported: shapefile missing or invalid: {shapefile_path}")

    print(f"\nDone. Vector confusion layers written to: {gpkg_path}")


# -------- #
#   CLI    #
# -------- #
def parse_args():
    p = argparse.ArgumentParser(description="Avalanche CD inference with optional aux (LIA, SLP).")
    p.add_argument("--event-path", type=Path, default=DEFAULT_EVENT_PATH, help="Folder with rasters.")
    p.add_argument("--rasters-dir", type=Path, default=DEFAULT_RASTERS_DIR, help="Output directory (results go under <rasters-dir>/<model-name>/).")
    p.add_argument("--stats-path", type=Path, default=DEFAULT_STATS_PATH, help="Stats JSON.")
    p.add_argument("--model-ckpt", type=Path, default=DEFAULT_MODEL_CKPT, help="Checkpoint path. Its parent folder name is used as the model name.")
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
    p.add_argument("--save-prob", action="store_true", help="Also write per-mode probability GeoTIFFs.")
    p.add_argument("--save-tif", action="store_true", help="Also write per-mode binary + confusion GeoTIFFs.")
    p.add_argument("--no-gt-copy", action="store_true", help="Do not copy the ground-truth GeoTIFF into the output folder.")
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
        save_prob=args.save_prob,
        save_tif=args.save_tif,
        copy_gt_tif=not args.no_gt_copy,
    )


if __name__ == "__main__":
    main()
