import numpy as np

from src.chapter7 import field_capacity, wilting_point
from src.chapter11 import topo_soil_index

def derive_soil_properties(model) -> None:
    """Derive spatial soil moisture limits, storage capacities, and apply domain masking."""

    # pull containers into separate variables
    control = model.control
    spatial = model.spatial
    params = model.params
    mask = spatial.maskNaN

    # 1. Directly create 2D spatial soil maps from scalar parameters
    spatial.THETAs = params.THETAs * mask
    spatial.PSIs = params.PSIs * mask
    spatial.b_BC = params.b_BC * mask
    spatial.T0 = params.T0 * mask
    spatial.K0 = params.K0 * mask

    # 2. Derive Field Capacity (THETAfc) and Permanent Wilting Point (THETApwp) via Brooks-Corey
    #   Note: we need to convert PSIs (m) to (cm)
    psis_cm = spatial.PSIs * 100.0
    spatial.THETAfc = field_capacity(psis_cm, spatial.b_BC, spatial.THETAs) * mask
    spatial.THETApwp = wilting_point(psis_cm, spatial.b_BC, spatial.THETAs) * mask
    spatial.albedo = params.albedo * mask
    spatial.emiss = params.emiss * mask

    # 3. Derive Maximum and Minimum Root Zone Storage (m)
    spatial.Srzmax = spatial.THETAfc * params.d_rz * mask
    spatial.Srzmin = spatial.THETApwp * params.d_rz * mask

    # 4. Compute soil-topographic index, basin area, and update transmissivity
    # Note: why do we not update T0 (surface transmissivity based on the output of the topo_soil_index function?)
    (
        spatial.lambda_map,
        params.lambda_mean,
        _,
        params.basin_area,
    ) = topo_soil_index(
        m=params.m,
        flowacc=spatial.flowacc,
        mask=mask,
        slope_deg=spatial.slope_deg,
        K0=spatial.K0,
        dx=control.dx,
        dy=control.dy,
    )

    # 5. Initialize spatially-masked snow emissivity grid
    spatial.snow_emiss = params.snow_emiss * mask