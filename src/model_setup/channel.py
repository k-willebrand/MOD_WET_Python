import warnings
import numpy as np

from src.chapter11 import manning_roughness_mean_to_pixel, channel_width, flow_network

def derive_channel_properties(model) -> None:
    """Derive channel width, Manning's n, bed slope, and routing flow network."""
    control = model.control
    spatial = model.spatial
    network = model.network
    params = model.params
    mask = spatial.maskNaN

    # 1. Manning roughness map
    network.manning_n = (
        manning_roughness_mean_to_pixel(params.manning_n_mean, spatial.slope_deg)
        * mask
    )

    # 2. Channel width map
    network.width = (
        channel_width(
            flowacc=spatial.flowacc,
            alpha=params.channel_width_coeff_alpha,
            c=params.channel_width_exponent_c,
            dx=control.dx,
            dy=control.dy,
        )
        * mask
    )

    # Resolution warning check
    max_width = float(np.nanmax(network.width))
    if max_width > control.dx:
        warnings.warn(
            f"MOD-WET warning: Maximum computed stream width ({max_width:.2f} m) "
            f"exceeds DEM resolution dx ({control.dx:.2f} m). "
            "May want to consider coarsening DEM to make more consistent.",
            UserWarning,
        )

    # 3. Converted bed slope
    network.bed_slope = np.tan(np.radians(spatial.slope_deg)) * mask

    # 4. Flow network routing pairs
    network.Iupstream, network.Idownstream, network.Ioutlet = flow_network(
        network.flowdir, mask
    )