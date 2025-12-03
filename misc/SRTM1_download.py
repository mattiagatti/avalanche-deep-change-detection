import elevation
import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
from pyproj import Transformer


os.environ["PROJ_LIB"] = "/home/jovyan/nfs/mgatti/python/avalanches/.venv/lib/python3.11/site-packages/pyproj/proj_dir/share/proj"

# Base directory
base_folder = Path("/home/jovyan/nfs/mgatti/datasets/Avalanches/images/")

# Loop through event folders
for event_dir in base_folder.iterdir():
    if not event_dir.is_dir() or event_dir.name.startswith('.'):
        continue

    event_name = event_dir.name
    resampled_dir = event_dir / "resampled"
    lia_path = resampled_dir / f"{event_name}_LIA.tif"
    dem_path = resampled_dir / f"{event_name}_DEM.tif"
    temp_reprojected_path = resampled_dir / f"{event_name}_DEM_temp.tif"

    if not lia_path.exists():
        print(f"Skipping {event_name}: LIA file not found")
        continue

    # Get bounds in EPSG:4326
    with rasterio.open(lia_path) as lia_src:
        bounds = lia_src.bounds
        crs_lia = lia_src.crs
        transformer = Transformer.from_crs(crs_lia, "EPSG:4326", always_xy=True)
        minx, miny = transformer.transform(bounds.left, bounds.bottom)
        maxx, maxy = transformer.transform(bounds.right, bounds.top)
        wgs84_bounds = (minx, miny, maxx, maxy)

    # Download DEM
    elevation.clean()
    elevation.clip(bounds=wgs84_bounds, output=str(dem_path))
    print(f"Downloaded DEM for {event_name}")

    # Reproject to LIA CRS, write to temp file
    with rasterio.open(dem_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, crs_lia, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': crs_lia,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(temp_reprojected_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=crs_lia,
                    resampling=Resampling.bilinear
                )

    # Overwrite original file
    temp_reprojected_path.replace(dem_path)
    print(f"Reprojected DEM saved (overwritten) for {event_name}")