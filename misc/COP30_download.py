import boto3
import math
import pandas as pd
import rasterio

from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path
from pyproj import Transformer
from rasterio.merge import merge

# AWS S3 Copernicus DEM public bucket
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket_name = 'copernicus-dem-30m'

# Base directory
base_folder = Path("/home/jovyan/nfs/mgatti/datasets/Avalanches/images_raw/")
MAPPING_PATH = base_folder / "crs_mapping.csv"

df = pd.read_csv(MAPPING_PATH, header=None, names=["event", "crs"])
target_crs_map = dict(zip(df.event, df.crs))


def get_tile_name(lat, lon):
    ns = 'N' if lat >= 0 else 'S'
    ew = 'E' if lon >= 0 else 'W'
    return f"Copernicus_DSM_COG_10_{ns}{abs(int(lat)):02d}_00_{ew}{abs(int(lon)):03d}_00_DEM/Copernicus_DSM_COG_10_{ns}{abs(int(lat)):02d}_00_{ew}{abs(int(lon)):03d}_00_DEM.tif"


def download_tile(tile_name, target_path):
    s3.download_file(bucket_name, tile_name, str(target_path))


for event_dir in base_folder.iterdir():
    if not event_dir.is_dir() or event_dir.name.startswith('.'):
        continue

    event_name = event_dir.name

    vh_path = event_dir / f"{event_name}_preVH.tif"
    dem_path = event_dir / f"{event_name}_DEM.tif"
    temp_reprojected_path = event_dir / f"{event_name}_DEM_temp.tif"

    if not vh_path.exists():
        print(f"Skipping {event_name}: VH file not found")
        continue

    # Get VH bounds and CRS
    with rasterio.open(vh_path) as vh_src:
        bounds = vh_src.bounds
        crs_vh = vh_src.crs
        transformer = Transformer.from_crs(crs_vh, "EPSG:4326", always_xy=True)
        minx, miny = transformer.transform(bounds.left, bounds.bottom)
        maxx, maxy = transformer.transform(bounds.right, bounds.top)
        wgs84_bounds = (minx, miny, maxx, maxy)

    # Determine Copernicus DEM tiles to download
    lons = range(math.floor(minx), math.ceil(maxx))
    lats = range(math.floor(miny), math.ceil(maxy))
    local_tiles = []

    for lat in lats:
        for lon in lons:
            tile_name = get_tile_name(lat, lon)
            local_tile_path = event_dir / Path(tile_name).name
            try:
                download_tile(tile_name, local_tile_path)
                local_tiles.append(local_tile_path)
                print(f"Downloaded {tile_name}")
            except Exception as e:
                print(f"Failed to download {tile_name}: {e}")

    if not local_tiles:
        print(f"No DEM tiles downloaded for {event_name}. Skipping.")
        continue

    # Merge tiles into one raster
    srcs = [rasterio.open(str(p)) for p in local_tiles]
    mosaic, out_trans = merge(srcs)

    out_meta = srcs[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })

    with rasterio.open(dem_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    print(f"Merged DEM saved for {event_name}")

    # Remove the original tile files
    for tile_path in local_tiles:
        try:
            tile_path.unlink()
            print(f"Removed {tile_path.name}")
        except Exception as e:
            print(f"Failed to remove {tile_path.name}: {e}")