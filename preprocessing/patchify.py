#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import subprocess

from pathlib import Path
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds
from rasterio.crs import CRS
from scipy.ndimage import binary_fill_holes
from skimage.util.shape import view_as_windows
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Patch extraction pipeline with optional AOI cropping and negative sampling.")
parser.add_argument("--patch-size", type=int, default=128, help="Patch size (default: 128)")
parser.add_argument("--stride", type=int, default=64, help="Stride (default: 64)")
parser.add_argument("--force", action="store_true",
                    help="Regenerate outputs even if they already exist.")
parser.add_argument("--force-resolution", type=int, default=0,
                    help="Force all rasters to a custom grid size in meters (default: 10). "
                         "Set to 0 to keep native resolution.")
args = parser.parse_args()

COMPRESSION = None
GT_COMPRESSION = "LZW"
FORCE_RESOLUTION = args.force_resolution
PIXEL_SIZE = FORCE_RESOLUTION if FORCE_RESOLUTION > 0 else None

NATIVE_EXTRA_PX = 160
PATCH_SIZE = args.patch_size
STRIDE = args.stride

root_dir = Path("/home/jovyan/nfs/mgatti/datasets/Avalanches/")
images_dir = root_dir / "images_raw"

resampled_images_dir = root_dir / (f"AvalCD_{FORCE_RESOLUTION}m" if FORCE_RESOLUTION > 0 else "AvalCD")
resampled_images_dir.mkdir(parents=True, exist_ok=True)
patches_root_dir = root_dir / (f"patches_{FORCE_RESOLUTION}m" if FORCE_RESOLUTION > 0 else "patches")
patches_dir = patches_root_dir / str(PATCH_SIZE)

events = [
    f.name
    for f in images_dir.iterdir()
    if f.is_dir() and not f.name.startswith((".", "_", "~"))
]

# -----------------------------------------------------------------------------
# CRS mapping
# -----------------------------------------------------------------------------
crs_csv = images_dir / "crs_mapping.csv"
if crs_csv.exists():
    crs_mapping = pd.read_csv(crs_csv).set_index("event")["crs"].to_dict()
else:
    crs_mapping = {}


def _crs_str(crs_obj) -> str | None:
    """Return a GDAL-friendly CRS string or None."""
    if not crs_obj:
        return None
    try:
        crs = CRS.from_user_input(crs_obj)
        s = crs.to_string()
        if s:
            return s
        epsg = crs.to_epsg()
        return f"EPSG:{epsg}" if epsg else None
    except Exception:
        return None


def resolve_target_crs(event_dir: Path, name: str, mapping: dict) -> str | None:
    """
    Try, in order:
      1) explicit mapping[name]
      2) <name>_preVH.tif CRS
      3) <name>_DEM.tif CRS
    """
    v = mapping.get(name)
    s = _crs_str(v)
    if s:
        return s

    pre_vh = event_dir / f"{name}_preVH.tif"
    if pre_vh.exists():
        try:
            with rasterio.open(pre_vh) as src:
                s = _crs_str(src.crs)
                if s:
                    return s
        except Exception:
            pass

    dem = event_dir / f"{name}_DEM.tif"
    if dem.exists():
        try:
            with rasterio.open(dem) as src:
                s = _crs_str(src.crs)
                if s:
                    return s
        except Exception:
            pass

    return None


# -----------------------------------------------------------------------------
# AOI helpers (single polygon)
# -----------------------------------------------------------------------------
def find_single_aoi(event_dir: Path) -> gpd.GeoDataFrame | None:
    """
    Look for a single AOI polygon file matching *_AOI_* in this priority:
    .gpkg > .geojson > .shp. Returns a GeoDataFrame with exactly one polygon.
    """
    patterns = ["*_AOI_*.gpkg", "*_AOI_*.geojson", "*_AOI_*.shp"]
    for pat in patterns:
        cands = sorted(event_dir.glob(pat))
        if not cands:
            continue
        # choose the first; warn if multiple but proceed
        path = cands[0]
        try:
            gdf = gpd.read_file(path)
            if gdf.empty:
                continue
            # dissolve all into one polygon if multiple features
            gdf = gdf[gdf.geometry.notnull()].copy()
            if len(gdf) > 1:
                gdf = gpd.GeoDataFrame(geometry=[gdf.unary_union], crs=gdf.crs)
            # ensure polygon
            geom = gdf.geometry.iloc[0]
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type not in ("Polygon", "MultiPolygon"):
                # try to polygonize bounds
                gdf = gpd.GeoDataFrame(geometry=[geom.envelope], crs=gdf.crs)
            return gdf
        except Exception:
            continue
    return None


def mask_raster_with_geom(raster_path: Path, geom_gdf: gpd.GeoDataFrame, prefer_nan_for_float=True) -> Path:
    """
    Mask (clip) raster by AOI polygon. Outside polygon -> nodata (or NaN for floats).
    Overwrites the file in place.
    """
    raster_path = Path(raster_path)
    with rasterio.open(raster_path) as src:
        dst_crs = src.crs
        # reproject AOI to raster CRS if needed
        if geom_gdf.crs and dst_crs and geom_gdf.crs != dst_crs:
            aoi = geom_gdf.to_crs(dst_crs)
        else:
            aoi = geom_gdf
        shapes = [aoi.geometry.iloc[0]]

        data = src.read()
        profile = src.profile.copy()

        # choose nodata policy
        dtype = np.dtype(src.dtypes[0])
        is_float = np.issubdtype(dtype, np.floating)
        new_nodata = src.nodata
        if is_float and prefer_nan_for_float:
            new_nodata = np.nan
            profile.update(nodata=np.nan, dtype="float32")
            data = data.astype("float32", copy=False)

        out, out_transform = rasterio.mask.mask(
            src, shapes, crop=True, filled=True,
            nodata=new_nodata
        )

        profile.update(
            height=out.shape[1], width=out.shape[2],
            transform=out_transform, compress=COMPRESSION
        )

    tmp = raster_path.with_suffix(raster_path.suffix + ".tmp")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(out)
    tmp.replace(raster_path)
    return raster_path


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def _expand_bounds(bounds, expand_m):
    l, b, r, t = bounds
    return (l - expand_m, b - expand_m, r + expand_m, t + expand_m)


def crop_native_with_margin(native_path, base_bounds, margin_m):
    """
    Crop `native_path` (already in target CRS) to base_bounds expanded by margin_m (meters),
    clamped to the dataset extent. Overwrites in place.
    """
    native_path = Path(native_path)
    with rasterio.open(native_path) as src:
        req_bounds = _expand_bounds(base_bounds, margin_m)
        l = max(req_bounds[0], src.bounds.left)
        b = max(req_bounds[1], src.bounds.bottom)
        r = min(req_bounds[2], src.bounds.right)
        t = min(req_bounds[3], src.bounds.top)
        ibounds = (l, b, r, t)

        win = from_bounds(*ibounds, transform=src.transform)
        win = Window(int(np.floor(win.col_off)),
                     int(np.floor(win.row_off)),
                     int(np.ceil(win.width)),
                     int(np.ceil(win.height)))
        win = win.intersection(Window(0, 0, src.width, src.height))

        data = src.read(window=win, boundless=True, fill_value=src.nodata)
        out_transform = src.window_transform(win)

        meta = src.meta.copy()
        meta.update(height=win.height, width=win.width,
                    transform=out_transform, compress=COMPRESSION)

    with rasterio.open(native_path, "w", **meta) as dst:
        dst.write(data)
    return native_path


def replace_nodata_with_nan(path) -> Path:
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with rasterio.open(path) as src:
            profile = src.profile.copy()
            src_nodata = src.nodata
            data = src.read().astype("float32", copy=False)
            if src_nodata is not None:
                if np.isnan(src_nodata):
                    mask = np.isnan(data)
                else:
                    mask = (data == src_nodata)
                data[mask] = np.nan
            profile.update(dtype="float32", nodata=np.nan)
            with rasterio.open(tmp_path, "w", **profile) as dst:
                dst.update_tags(**src.tags())
                dst.write(data)
        tmp_path.replace(path)
        return path
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def resample_raster(src_path, reference_path, target_crs):
    event_name = src_path.parent.name
    output_dir = resampled_images_dir / event_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / src_path.name

    with rasterio.open(reference_path) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height

    with rasterio.open(src_path) as src:
        src_data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
        src_dtype = np.dtype(src.dtypes[0])
        src_nodata = src.nodata

        is_dem = src_path.stem.lower().endswith("_dem")
        is_float_src = np.issubdtype(src_dtype, np.floating)

        if "lia" in src_path.name.lower() and np.nanmax(src_data) < 21.0:
            print(f"[{event_name}] Converting LIA from dB to linear scale...")
            src_data = 10 ** (src_data.astype("float32") / 10.0)
            is_float_src = True

        if is_float_src:
            out_dtype = np.dtype("float32")
            dst_nodata = np.nan
        else:
            out_dtype = src_dtype
            dst_nodata = src_nodata

        kwargs = src.meta.copy()
        kwargs.update({
            "driver": "GTiff",
            "crs": ref_crs,
            "transform": ref_transform,
            "width": ref_width,
            "height": ref_height,
            "count": 1,
            "dtype": out_dtype.name,
            "nodata": dst_nodata,
            "compress": COMPRESSION
        })

        if np.issubdtype(out_dtype, np.floating):
            resampled = np.full((ref_height, ref_width), np.nan, dtype=out_dtype)
        else:
            fill = src_nodata if src_nodata is not None else 0
            resampled = np.full((ref_height, ref_width), fill, dtype=out_dtype)

        kernel = Resampling.bilinear

        reproject(
            source=src_data,
            destination=resampled,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=kernel,
            src_nodata=src_nodata,
            dst_nodata=dst_nodata
        )

        with rasterio.open(output_path, "w", **kwargs) as dst:
            dst.write(resampled, 1)

    if is_dem:
        slope_path = output_dir / f"{src_path.stem.replace('_DEM', '_SLP')}.tif"
        aspect_path = output_dir / f"{src_path.stem.replace('_DEM', '_ASP')}.tif"

        subprocess.run([
            "gdaldem", "slope", str(output_path), str(slope_path),
            "-of", "GTiff", "-s", "1.0", "-compute_edges"
        ], check=True)
        subprocess.run([
            "gdaldem", "aspect", str(output_path), str(aspect_path),
            "-of", "GTiff", "-s", "1.0", "-compute_edges"
        ], check=True)

        replace_nodata_with_nan(slope_path)
        replace_nodata_with_nan(aspect_path)

        print(f"Slope generated for {event_name}: {slope_path.name}")
        print(f"Aspect written to: {aspect_path}")

    return output_path


def crop_gt_to_positive_area(gt_path, margin_left=32, margin_right=32,
                             margin_top=32, margin_bottom=32):
    with rasterio.open(gt_path) as src:
        data = src.read(1)
        rows, cols = np.where(data == 1)
        if rows.size == 0:
            print(f'No positives in {gt_path}')
            return gt_path, None

        row_min = max(rows.min() - margin_top, 0)
        row_max = min(rows.max() + margin_bottom, src.height - 1)
        col_min = max(cols.min() - margin_left, 0)
        col_max = min(cols.max() + margin_right, src.width - 1)

        window = Window.from_slices((row_min, row_max + 1),
                                    (col_min, col_max + 1))
        out_transform = src.window_transform(window)
        out_data = src.read(window=window)

        meta = src.meta.copy()
        meta.update(
            height=window.height,
            width=window.width,
            transform=out_transform,
            nodata=None,
            compress=GT_COMPRESSION
        )

    with rasterio.open(gt_path, "w", **meta) as dst:
        dst.write(out_data)
    return gt_path, window


def crop_by_window(raster_path, window):
    raster_path = Path(raster_path)
    if window is None:
        return raster_path
    with rasterio.open(raster_path) as src:
        data = src.read(window=window, boundless=True, fill_value=src.nodata)
        out_transform = src.window_transform(window)
        meta = src.meta.copy()
        meta.update(height=window.height, width=window.width,
                    transform=out_transform, compress=COMPRESSION)
    with rasterio.open(raster_path, "w", **meta) as dst:
        dst.write(data)
    return raster_path


def assert_same_shape(*paths):
    shapes = [get_raster_shape(p) for p in paths]
    first = shapes[0]
    for p, s in zip(paths, shapes):
        assert s == first, f"{p.name} shape {s} differs from {first}"


def prepare_reference_raster(reference_path, target_crs, event_name):
    """
    Reproject <reference_path> to <target_crs> if provided.
    If FORCE_RESOLUTION > 0, snap to N×N m grid (-tap -tr N N).
    """
    out_dir = resampled_images_dir / event_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / reference_path.name

    with rasterio.open(reference_path) as src:
        dtype = src.dtypes[0]
        resampling = "near" if np.issubdtype(np.dtype(dtype), np.integer) else "bilinear"
        src_nodata = src.nodata

    cmd = [
        "gdalwarp", "-overwrite",
        "-r", resampling,
        "-multi", "-wo", "NUM_THREADS=ALL_CPUS",
        "-of", "GTiff",
    ]

    if target_crs:
        cmd += ["-t_srs", str(target_crs)]

    if src_nodata is not None:
        cmd += ["-dstnodata", str(src_nodata)]

    if FORCE_RESOLUTION and FORCE_RESOLUTION > 0:
        cmd += ["-tap", "-tr", str(PIXEL_SIZE), str(PIXEL_SIZE)]

    cmd += [str(reference_path), str(out_path)]

    print("[gdalwarp -> REF]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    return out_path


def _clean_attributes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop 'id' and cast 'size'/'area' to integers (rounded, nullable)."""
    gdf = gdf.copy()
    # Drop 'id' if present
    if "id" in gdf.columns:
        gdf = gdf.drop(columns=["id"])
    # Cast size/area -> integer
    for col in ("size", "area"):
        if col in gdf.columns:
            s = pd.to_numeric(gdf[col], errors="coerce").round()
            # Nullable integer (keeps NaN as <NA>)
            gdf[col] = s.astype("Int64")
    return gdf


def save_polygons_gpkg(vector_path: Path, reference_raster_path: Path, out_gpkg_path: Path, aoi_gdf: gpd.GeoDataFrame | None = None) -> Path:
    """
    Read polygons from `vector_path` (SHP or GPKG), reproject to the CRS of `reference_raster_path`,
    optionally clip to AOI, clean attributes, and write to `out_gpkg_path` as GPKG.
    """
    if vector_path is None or not Path(vector_path).exists():
        return None

    with rasterio.open(reference_raster_path) as ref:
        ref_crs = ref.crs

    gdf = gpd.read_file(vector_path)
    if gdf.empty:
        # write empty file so downstream code still finds a gpkg
        gpd.GeoDataFrame(geometry=[], crs=ref_crs).to_file(out_gpkg_path, driver="GPKG")
        return out_gpkg_path

    # reproject to reference CRS if needed
    if gdf.crs != ref_crs:
        gdf = gdf.to_crs(ref_crs)

    # optional AOI clip (keeps only polygons inside the AOI)
    if aoi_gdf is not None and not aoi_gdf.empty:
        aoi = aoi_gdf
        if aoi.crs != ref_crs:
            aoi = aoi.to_crs(ref_crs)
        try:
            gdf = gdf.clip(aoi)
        except Exception:
            gdf = gdf[gdf.geometry.intersects(aoi.geometry.iloc[0])]

    # ---- Clean attributes: drop 'id', cast 'size'/'area' to integer
    gdf = _clean_attributes(gdf)

    out_gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_gpkg_path, driver="GPKG", index=False)
    return out_gpkg_path


def rasterize_shapefile(shapefile_path, reference_raster_path):
    event_name = shapefile_path.parent.name
    output_raster_path = resampled_images_dir / event_name / f"{shapefile_path.stem}_GT.tif"
    output_raster_path.parent.mkdir(parents=True, exist_ok=True)
    output_vector_path = resampled_images_dir / event_name / f"{shapefile_path.stem}_GT.gpkg"

    with rasterio.open(reference_raster_path) as src:
        meta = src.meta.copy()
        transform = src.transform
        width, height = src.width, src.height
        crs = src.crs

    gdf = gpd.read_file(shapefile_path)

    if gdf.crs != crs:
        print(f"Reprojecting shapefile from {gdf.crs} to {crs}")
        gdf = gdf.to_crs(crs)

    # ---- Clean attributes before saving the GeoPackage copy
    gdf_out = _clean_attributes(gdf)
    gdf_out.to_file(output_vector_path, driver="GPKG", index=False)
    print(f"Saved vector data as GeoPackage: {output_vector_path}")

    shapes = [(geom, 1) for geom in gdf.geometry]
    mask = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype=rasterio.uint8)

    meta.update({"driver": "GTiff", "count": 1, "dtype": rasterio.uint8, "nodata": 255, "compress": COMPRESSION})
    with rasterio.open(output_raster_path, "w", **meta) as dst:
        dst.write(mask, 1)

    return output_raster_path


def get_raster_shape(file_path):
    with rasterio.open(file_path) as dataset:
        return dataset.shape


def save_native_dem_products(dem_path, out_dir, target_crs):
    """
    If target_crs is provided -> warp DEM to it; else keep native CRS (copy/warp without -t_srs).
    Then compute SLOPE and ASPECT on the (possibly reprojected) DEM.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    event = dem_path.parent.name

    dem_native = out_dir / f"{event}_DEM_native.tif"
    slp_native = out_dir / f"{event}_SLP_native.tif"
    asp_native = out_dir / f"{event}_ASP_native.tif"

    if dem_native.exists() and slp_native.exists() and asp_native.exists() and not args.force:
        print(f"[{event}] Native DEM/SLP/ASP already exist, skipping.")
        return dem_native, slp_native, asp_native

    with rasterio.open(dem_path) as src:
        src_dtype = src.dtypes[0]
        is_float = np.issubdtype(np.dtype(src_dtype), np.floating)

    warp_cmd = [
        "gdalwarp", "-overwrite",
        "-r", "bilinear",
        "-multi", "-wo", "NUM_THREADS=ALL_CPUS",
        "-of", "GTiff",
    ]

    if target_crs:
        warp_cmd += ["-t_srs", str(target_crs)]

    if is_float:
        warp_cmd += ["-ot", "Float32", "-dstnodata", "nan"]

    if COMPRESSION:
        warp_cmd += ["-co", f"COMPRESS={COMPRESSION}"]

    warp_cmd += [str(dem_path), str(dem_native)]
    print("[gdalwarp -> DEM_native]", " ".join(map(str, warp_cmd)))
    subprocess.run(warp_cmd, check=True)

    slope_cmd = ["gdaldem", "slope", str(dem_native), str(slp_native), "-of", "GTiff", "-s", "1.0", "-compute_edges"]
    aspect_cmd = ["gdaldem", "aspect", str(dem_native), str(asp_native), "-of", "GTiff", "-s", "1.0", "-compute_edges"]
    if COMPRESSION:
        slope_cmd += ["-co", f"COMPRESS={COMPRESSION}"]
        aspect_cmd += ["-co", f"COMPRESS={COMPRESSION}"]

    subprocess.run(slope_cmd, check=True)
    subprocess.run(aspect_cmd, check=True)
    replace_nodata_with_nan(slp_native)
    replace_nodata_with_nan(asp_native)

    print(f"[{event}] Saved native DEM/SLP/ASP: {dem_native.name}, {slp_native.name}, {asp_native.name}")
    return dem_native, slp_native, asp_native


def extract_and_save_patches(vh_pre_path, vv_pre_path, vh_post_path, vv_post_path, lia_path, dem_path, slp_path, asp_path, gt_path, patch_size, stride, dem_native_path, slp_native_path, asp_native_path):
    output_dir = patches_dir / vh_pre_path.parent.stem

    if output_dir.exists() and not args.force:
        print(f"Patches already exist for {vh_pre_path.parent.stem}, skipping...")
        return

    print("Saving patches...")

    with rasterio.open(vh_pre_path) as src_vh_pre, \
         rasterio.open(vv_pre_path) as src_vv_pre, \
         rasterio.open(vh_post_path) as src_vh_post, \
         rasterio.open(vv_post_path) as src_vv_post, \
         rasterio.open(lia_path) as src_lia, \
         rasterio.open(dem_path) as src_dem, \
         rasterio.open(slp_path) as src_slp, \
         rasterio.open(asp_path) as src_asp, \
         rasterio.open(gt_path) as src_gt, \
         rasterio.open(dem_native_path) as src_dem_nat, \
         rasterio.open(slp_native_path) as src_slp_nat, \
         rasterio.open(asp_native_path) as src_asp_nat:

        vh_pre = src_vh_pre.read(1)
        vv_pre = src_vv_pre.read(1)
        vh_post = src_vh_post.read(1)
        vv_post = src_vv_post.read(1)
        lia = src_lia.read(1)
        dem = src_dem.read(1)
        slp = src_slp.read(1)
        gt = src_gt.read(1)
        asp = src_asp.read(1)

        valid_mask = (
            ~np.isnan(vh_pre) &
            ~np.isnan(vv_pre) &
            ~np.isnan(vh_post) &
            ~np.isnan(vv_post)
        )

        region_mask = binary_fill_holes(valid_mask).astype(np.uint8)

        meta = src_vh_pre.meta.copy()
        meta.update({"height": patch_size, "width": patch_size, "count": 2})

        vh_pre_patches = view_as_windows(vh_pre, (patch_size, patch_size), stride)
        vv_pre_patches = view_as_windows(vv_pre, (patch_size, patch_size), stride)
        vh_post_patches = view_as_windows(vh_post, (patch_size, patch_size), stride)
        vv_post_patches = view_as_windows(vv_post, (patch_size, patch_size), stride)
        lia_patches = view_as_windows(lia, (patch_size, patch_size), stride)
        dem_patches = view_as_windows(dem, (patch_size, patch_size), stride)
        slp_patches = view_as_windows(slp, (patch_size, patch_size), stride)
        asp_patches = view_as_windows(asp, (patch_size, patch_size), stride)
        gt_patches = view_as_windows(gt, (patch_size, patch_size), stride)
        region_patches = view_as_windows(region_mask.astype(np.uint8), (patch_size, patch_size), stride)

        num_patches_y, num_patches_x = vh_pre_patches.shape[:2]
        positives = []
        negatives = []

        for patch_idx in tqdm(range(num_patches_y * num_patches_x), desc="Processing patches", unit="patch"):
            i = patch_idx // num_patches_x
            j = patch_idx % num_patches_x

            vh_pre_patch = vh_pre_patches[i, j]
            vv_pre_patch = vv_pre_patches[i, j]
            vh_post_patch = vh_post_patches[i, j]
            vv_post_patch = vv_post_patches[i, j]
            lia_patch = lia_patches[i, j]
            dem_patch = dem_patches[i, j]
            slp_patch = slp_patches[i, j]
            asp_patch = asp_patches[i, j]
            gt_patch = gt_patches[i, j]
            region_patch = region_patches[i, j]

            inside_region = np.all(region_patch == 1)

            if not inside_region:
                nan_mask = (
                    np.isnan(vh_pre_patch) |
                    np.isnan(vv_pre_patch) |
                    np.isnan(vh_post_patch) |
                    np.isnan(vv_post_patch)
                )
                nan_fraction = nan_mask.mean()
                if nan_fraction >= 0.5:
                    continue

            patch_data = {
                "i": i,
                "j": j,
                "vh_pre": vh_pre_patch,
                "vv_pre": vv_pre_patch,
                "vh_post": vh_post_patch,
                "vv_post": vv_post_patch,
                "lia": lia_patch,
                "dem": dem_patch,
                "slp": slp_patch,
                "asp": asp_patch,
                "gt": gt_patch,
            }

            H = patch_size
            W = patch_size
            half_h = H // 2
            half_w = W // 2

            center_row_sar = i * stride + half_h
            center_col_sar = j * stride + half_w
            cx, cy = src_vh_pre.transform * (center_col_sar + 0.5, center_row_sar + 0.5)

            row_wide, col_wide = src_dem_nat.index(cx, cy)

            r0 = row_wide - half_h
            c0 = col_wide - half_w
            win_wide = Window(col_off=c0, row_off=r0, width=W, height=H)

            is_float_dem = np.issubdtype(np.dtype(src_dem_nat.dtypes[0]), np.floating)
            is_float_slp = np.issubdtype(np.dtype(src_slp_nat.dtypes[0]), np.floating)
            is_float_asp = np.issubdtype(np.dtype(src_asp_nat.dtypes[0]), np.floating)

            fill_dem = src_dem_nat.nodata if src_dem_nat.nodata is not None else (np.nan if is_float_dem else 0)
            fill_slp = src_slp_nat.nodata if src_slp_nat.nodata is not None else (np.nan if is_float_slp else 0)
            fill_asp = src_asp_nat.nodata if src_asp_nat.nodata is not None else (np.nan if is_float_asp else 0)

            dem_wide_patch = src_dem_nat.read(1, window=win_wide, boundless=True, fill_value=fill_dem)
            slp_wide_patch = src_slp_nat.read(1, window=win_wide, boundless=True, fill_value=fill_slp)
            asp_wide_patch = src_asp_nat.read(1, window=win_wide, boundless=True, fill_value=fill_asp)

            patch_transform_wide = src_dem_nat.window_transform(win_wide)

            patch_data.update({
                "dem_wide": dem_wide_patch,
                "slp_wide": slp_wide_patch,
                "asp_wide": asp_wide_patch,
                "transform_wide": patch_transform_wide,
            })

            if np.max(gt_patch) >= 1:
                positives.append(patch_data)
            else:
                negatives.append(patch_data)

        print(f"Saving {len(positives)} positive and {len(negatives)} negative patches...")

        output_dir.mkdir(exist_ok=True, parents=True)

        def save_patch(patch_data, patch_id):
            i = patch_data["i"]; j = patch_data["j"]
            window = Window(j * stride, i * stride, patch_size, patch_size)
            patch_transform_sar = src_vh_pre.window_transform(window)

            patch_folder = output_dir / str(patch_id)
            patch_folder.mkdir(exist_ok=True)

            meta_sar = src_vh_pre.meta.copy()
            meta_sar.update({"driver": "GTiff", "transform": patch_transform_sar,
                             "height": patch_size, "width": patch_size})

            patch_transform_nat = patch_data["transform_wide"]
            meta_nat = src_dem_nat.meta.copy()
            meta_nat.update({"driver": "GTiff", "transform": patch_transform_nat,
                             "height": patch_size, "width": patch_size})

            items = [
                ("pre.tif", [patch_data["vh_pre"], patch_data["vv_pre"]], 2, meta_sar),
                ("post.tif", [patch_data["vh_post"], patch_data["vv_post"]], 2, meta_sar),
                ("lia.tif", [patch_data["lia"]], 1, meta_sar),
                ("dem.tif", [patch_data["dem"]], 1, meta_sar),
                ("slope.tif", [patch_data["slp"]], 1, meta_sar),
                ("aspect.tif", [patch_data["asp"]], 1, meta_sar),
                ("mask.tif", [patch_data["gt"]], 1, meta_sar),

                ("dem_wide.tif", [patch_data["dem_wide"]], 1, meta_nat),
                ("slope_wide.tif", [patch_data["slp_wide"]], 1, meta_nat),
                ("aspect_wide.tif", [patch_data["asp_wide"]], 1, meta_nat),
            ]

            for name, bands, count, meta_here in items:
                meta_here = meta_here.copy()
                meta_here.update({"count": count, "compress": None})
                with rasterio.open(patch_folder / name, "w", **meta_here) as dst:
                    for b, arr in enumerate(bands, start=1):
                        dst.write(arr, b)

        patch_id = 1
        for patch_data in positives:
            save_patch(patch_data, patch_id)
            patch_id += 1
        for patch_data in negatives:
            save_patch(patch_data, patch_id)
            patch_id += 1

        print(f"Saved {patch_id - 1} patches in {output_dir}")


def ensure_gt_mask(reference_raster_path, vector_path, out_gt_path):
    """
    If `vector_path` is None or empty, write a 0-filled GT aligned to `reference_raster_path`.
    Otherwise rasterize it to a uint8 mask (1=positive, 0=background).
    """
    with rasterio.open(reference_raster_path) as ref:
        h, w = ref.height, ref.width
        transform, crs = ref.transform, ref.crs
        meta = ref.meta.copy()
    meta.update(driver="GTiff", count=1, dtype=rasterio.uint8, nodata=255, transform=transform, crs=crs, compress=COMPRESSION)

    if vector_path is None:
        mask = np.zeros((h, w), dtype=np.uint8)
    else:
        gdf = gpd.read_file(vector_path)
        if gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
        if not shapes:
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            mask = rasterize(shapes, out_shape=(h, w), transform=transform, fill=0, dtype=rasterio.uint8)

    out_gt_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_gt_path, "w", **meta) as dst:
        dst.write(mask, 1)
    return out_gt_path


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    for event in events:
        print(f"\n=== Processing {event} ===")
        event_dir = images_dir / event
        output_dir = patches_dir / event

        pre_vh_path = event_dir / f"{event}_preVH.tif"
        pre_vv_path = event_dir / f"{event}_preVV.tif"
        post_vh_path = event_dir / f"{event}_postVH.tif"
        post_vv_path = event_dir / f"{event}_postVV.tif"
        lia_path = event_dir / f"{event}_LIA.tif"
        dem_path = event_dir / f"{event}_DEM.tif"
        slp_path = event_dir / f"{event}_SLP.tif"
        asp_path = event_dir / f"{event}_ASP.tif"
        shapefile_path = event_dir / f"{event}.shp"

        # Resolve CRS (robust)
        target_crs = resolve_target_crs(event_dir, event, crs_mapping)
        print(f"[{event}] Target CRS: {target_crs or '(keep native)'}")

        # Single AOI polygon if present
        aoi_gdf = find_single_aoi(event_dir)
        if aoi_gdf is None:
            print(f"[{event}] No AOI polygon found (pattern *_AOI_*).")
        else:
            print(f"[{event}] AOI polygon found and will be applied.")

        # Native-grid products (DEM, then SLP/ASP on the native-projected DEM)
        native_out_dir = resampled_images_dir / event
        dem_native, slp_native, asp_native = save_native_dem_products(dem_path, native_out_dir, target_crs)

        # If GT already exists — skip re-warp/rasterize
        gt_path = resampled_images_dir / event / f"{event}_GT.tif"
        if gt_path.exists() and not args.force:
            print(f"GT mask already exists for {event}, skipping resampling and rasterization.")
            resampled_paths = [resampled_images_dir / event / f"{path.name}" for path in [
                pre_vh_path, pre_vv_path, post_vh_path, post_vv_path, lia_path, dem_path, slp_path, asp_path
            ]]

            # Apply AOI (if any) to each resampled path and the GT, then proceed
            if aoi_gdf is not None:
                for rp in resampled_paths:
                    mask_raster_with_geom(rp, aoi_gdf)
                mask_raster_with_geom(gt_path, aoi_gdf)

            # bounds of the area we keep (after AOI masking, bounds match masked data)
            with rasterio.open(resampled_images_dir / event / f"{event}_GT.tif") as src_gt:
                sar_bounds = src_gt.bounds

            with rasterio.open(dem_native) as src_nat:
                nat_res_m = abs(src_nat.transform.a)
            margin_m = NATIVE_EXTRA_PX * nat_res_m

            crop_native_with_margin(dem_native, sar_bounds, margin_m)
            crop_native_with_margin(slp_native, sar_bounds, margin_m)
            crop_native_with_margin(asp_native, sar_bounds, margin_m)

        else:
            # Prepare reference (preVH) possibly reprojected / snapped
            reference_path = prepare_reference_raster(pre_vh_path, target_crs, event)

            print(f"Using CRS: {target_crs or 'source raster CRS'}")
            print("Resampling rasters...")
            others = [pre_vv_path, post_vh_path, post_vv_path, lia_path, dem_path]
            resampled_paths = [reference_path] + [
                resample_raster(p, reference_path, target_crs) for p in others
            ]
            dem_resampled = resampled_paths[-1]
            resampled_paths.append(dem_resampled.with_name(dem_resampled.stem.rsplit("_", 1)[0] + "_SLP.tif"))
            resampled_paths.append(dem_resampled.with_name(dem_resampled.stem.rsplit("_", 1)[0] + "_ASP.tif"))

            # Apply AOI before GT (so GT aligns with AOI bounds); this will crop each raster
            if aoi_gdf is not None:
                for rp in resampled_paths:
                    mask_raster_with_geom(rp, aoi_gdf)

            # prefer GPKG, else SHP, else None
            gpkg_path = event_dir / f"{event}.gpkg"
            shp_path = event_dir / f"{event}.shp"
            vector_path = gpkg_path if gpkg_path.exists() else (shp_path if shp_path.exists() else None)

            # Always materialize a GeoPackage copy under AvalCD/<event>/<event>.gpkg
            gpkg_out = resampled_images_dir / event / f"{event}.gpkg"
            if vector_path is not None:
                saved_gpkg = save_polygons_gpkg(vector_path, reference_path, gpkg_out, aoi_gdf=aoi_gdf)
                if saved_gpkg is not None:
                    print(f"[{event}] Saved polygons GeoPackage: {saved_gpkg}")
                    # Use this GPKG for GT to ensure CRS matches the rasters (and AOI clipping applied)
                    vector_path = saved_gpkg
            else:
                print(f"[{event}] No polygons found (.gpkg/.shp).")

            gt_path = resampled_images_dir / event / f"{event}_GT.tif"
            print("Building GT (rasterize if present, else 0-filled)...")
            gt_path = ensure_gt_mask(reference_path, vector_path, gt_path)

            # If AOI exists, mask GT as well (outside AOI -> 0 / nodata)
            if aoi_gdf is not None:
                mask_raster_with_geom(gt_path, aoi_gdf, prefer_nan_for_float=False)  # GT stays uint8

            # Crop by positives if any
            gt_path, pixel_window = crop_gt_to_positive_area(gt_path)

            if pixel_window is None:
                cropped_paths = resampled_paths
                assert_same_shape(*(cropped_paths + [gt_path]))
                reference_path = resampled_paths[0]
                with rasterio.open(reference_path) as ref:
                    sar_bounds = ref.bounds
            else:
                cropped_paths = [crop_by_window(p, pixel_window) for p in resampled_paths]
                assert_same_shape(*(cropped_paths + [gt_path]))
                reference_path = resampled_paths[0]
                with rasterio.open(reference_path) as ref:
                    sar_bounds = ref.bounds

            with rasterio.open(dem_native) as src_nat:
                nat_res_m = abs(src_nat.transform.a)
            margin_m = NATIVE_EXTRA_PX * nat_res_m

            crop_native_with_margin(dem_native, sar_bounds, margin_m)
            crop_native_with_margin(slp_native, sar_bounds, margin_m)
            crop_native_with_margin(asp_native, sar_bounds, margin_m)

        extract_and_save_patches(
            *resampled_paths,
            gt_path,
            PATCH_SIZE,
            STRIDE,
            dem_native,
            slp_native,
            asp_native
        )