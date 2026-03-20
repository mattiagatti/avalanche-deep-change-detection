import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from scipy.ndimage import binary_fill_holes


root = Path("/home/jovyan/nfs/mgatti/datasets/Avalanches/AvalCD")
rows = []

for d in root.rglob("*"):
    if not d.is_dir():
        continue
    vh = next(d.glob("*_preVH.tif"), None)
    if vh is None:
        continue

    with rasterio.open(vh) as src:
        data, tfm, crs = src.read(1), src.transform, src.crs
        nrows, ncols = data.shape

        if crs.to_epsg() == 4326:  # geographic CRS
            lat = (src.bounds.top + src.bounds.bottom) / 2
            mpd_lat = 111_320
            mpd_lon = mpd_lat * np.cos(np.radians(lat))
            pw = abs(tfm.a) * mpd_lon
            ph = abs(tfm.e) * mpd_lat
        else:  # projected CRS
            pw, ph = abs(tfm.a), abs(tfm.e)

        pa = pw * ph  # pixel area in m²
        region_px = binary_fill_holes(~np.isnan(data)).sum()
        bbox_px = nrows * ncols

        rows.append(
            dict(
                event=d.name,
                region_km2 = region_px * pa / 1e6,
                bbox_km2 = bbox_px * pa / 1e6
            )
        )

# ---- print CSV w/ no spaces ----
df = pd.DataFrame(rows).sort_values("event")
print(df.to_csv(index=False, header=False, float_format="%.2f"))