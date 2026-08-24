import numpy as np

from src.chapter3 import easting_northing_to_lat_lon

def derive_terrain_maps(model) -> None:
    """Derive terrain grids, convert UTM to Lat/Lon, and compute basin metadata."""
    # 1. Build NaN mask grid
    model.spatial.maskNaN = np.where(model.spatial.mask == 1, 1.0, np.nan)

    # 2. Convert angles to radians (masked to domain)
    model.spatial.slope_rad = np.radians(model.spatial.slope_deg)
    model.spatial.aspect_rad = np.radians(model.spatial.aspect_deg)

    # 3. Derive 1D Lat/Lon vectors from 1D Easting/Northing vectors
    if model.spatial.easting is not None and model.spatial.northing is not None:
        lat_1d, lon_1d = easting_northing_to_lat_lon(
            model.spatial.easting,
            model.spatial.northing,
            model.params.utmzone,
        )
        model.spatial.lat = lat_1d
        model.spatial.lon = lon_1d

        # 4. Compute mean lat/lon coordinates
        model.params.lat_mean = float(np.mean(lat_1d))
        model.params.lon_mean = float(np.mean(lon_1d))

