import numpy as np

def field_capacity(
    psi_s: np.ndarray, b: np.ndarray, porosity: np.ndarray
) -> np.ndarray:
    """Compute field capacity volumetric soil moisture. `psi_s` must be in cm."""
    return porosity * (340.0 / np.abs(psi_s)) ** (-1.0 / b)

def wilting_point(
    psi_s: np.ndarray, b: np.ndarray, porosity: np.ndarray
) -> np.ndarray:
    """Compute permanent wilting point volumetric soil moisture. `psi_s` must be in cm."""
    return porosity * (15000.0 / np.abs(psi_s)) ** (-1.0 / b)

def topo_soil_index(
    m: float,
    flowacc: np.ndarray,
    mask: np.ndarray,
    slope_deg: np.ndarray,
    K0: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Compute the soil-topographic index, mean topographic index, transmissivity, and basin area."""
    # Upstream drainage area per unit contour length
    ai = flowacc * dx

    # Convert slope to radians and handle 0-degree slopes to avoid division by zero
    slope_rad = np.radians(slope_deg)
    tan_slope = np.where(slope_rad == 0, np.nan, np.tan(slope_rad))

    # Transmissivity under saturated conditions
    T0 = K0 * m

    # Local soil-topographic index
    lambda_map = np.log(ai / (T0 * tan_slope)) * mask

    # Basin area and mean topographic index
    basin_area = float(np.nansum(mask * dx * dy))
    lambda_mean = float(np.nansum(lambda_map * dx * dy) / basin_area)

    return lambda_map, lambda_mean, T0, basin_area

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