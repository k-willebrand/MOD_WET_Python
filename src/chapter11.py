import sys
import warnings
import numpy as np
import scipy.sparse as sp

# Note: assume topo_index.m superceded by topo_soil_index.m from original MATLAB script  

def channel_width(flowacc: np.ndarray, alpha: float, c: float, dx: float, dy: float) -> np.ndarray:
    """Compute distributed channel width (m) across the basin using scaling power law."""
    m2km = 1000.0
    # Catchment area in km^2
    ai = (flowacc * dx * dy) / (m2km**2)
    # Channel width in meters
    return alpha * (ai**c)

# New version that returns 2D row/col index tuples isntead of 1D Fortran order index vectors
def flow_network(flowdir: np.ndarray | sp.coo_matrix | sp.csr_matrix, mask: np.ndarray
                 ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[int, int]]:
    """
    Determine upstream/downstream pixel pairs and outlet index from flow direction matrix.
    Replicates MATLAB's equivalent function exactly (1D Fortran indices) up until the 
    last step, where everything is converted from 1D Fortran indices to 2D C index arrays.

    NOTE:
    Input rows and cols MUST BE Fortran order, will update to accept C order in the future.

    Returns 0-based 2D index tuples (row_indices, col_indices) for Iupstream, Idownstream,
    and a 2D coordinate tuple (row, col) for Ioutlet.
    """
    # Ensure COO format
    if not isinstance(flowdir, (sp.coo_matrix, sp.coo_array)):
        flowdir = flowdir.tocoo()

    # 1. Extract row (U) and column (D) indices where flowdir == 1 (in 1D/Fortran order)
    # Force CSC layout to sort by column, then convert to COO to extract row/col arrays
    # so U and D are ordered Column-by-Column (MATLAB style)
    if sp.issparse(flowdir):
        flowdir_mb_order = flowdir.tocsc().tocoo()
        ones_mask = flowdir_mb_order.data == 1
        U, D = flowdir_mb_order.row[ones_mask], flowdir_mb_order.col[ones_mask]
    else:
        U, D = np.where(flowdir == 1)

    # 2. Identify pixels in basin (using 1D/Fortran order)
    index_0 = False # work with 1-based indexing instead of 0-based for testing and validation with MATLAB
    if index_0:
        remove_val = -1
    else:
        remove_val = 0
    Imask = np.where(mask.ravel(order='F') == 1)[0]

    # Basin pixels with flowdir=1 using a replicate MATLAB ismember() function
    _, b = matlab_ismember(U, Imask, zero_based=index_0)
    _, bb = matlab_ismember(D, Imask, zero_based=index_0)
    
    # Remove any pixels that are not in the basin
    remove_mask = (b != remove_val) ^ (bb != remove_val)
    b[remove_mask] = remove_val
    bb[remove_mask] = remove_val

    # Extract upstream and downstream indices for basin pixels only
    UU = U[b > remove_val]
    DD = D[bb > remove_val]

    Iupstream = np.unravel_index(UU, mask.shape, order='F')
    Idownstream = np.unravel_index(DD, mask.shape, order='F')

    I = ~np.isin(DD, UU)
    Ioutlet_1d = DD[I][0] if np.any(I) else None # specifying DD[I][0] replicates 'first' in Ioutlet=DD(find(I,1,'first'));

    if Ioutlet_1d is None:
        sys.exit("Unable to find valid outlet point.")
    else:
        r_out, c_out = np.unravel_index(Ioutlet_1d, mask.shape, order="F")
        Ioutlet = (int(r_out), int(c_out))

    return Iupstream, Idownstream, Ioutlet

def flow_network_old(flowdir: np.ndarray | sp.coo_matrix | sp.csr_matrix, mask: np.ndarray
                 ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[int, int]]:
    """
    Determine upstream/downstream pixel pairs and outlet index from flow direction matrix.

    Returns 0-based 2D index tuples (row_indices, col_indices) for Iupstream, Idownstream,
    and a 2D coordinate tuple (row, col) for Ioutlet.
    """
    # Ensure COO format
    if not isinstance(flowdir, (sp.coo_matrix, sp.coo_array)):
        flowdir = flowdir.tocoo()

    # 1. Extract row (U) and column (D) indices where flowdir == 1
    if sp.issparse(flowdir):
        ones_mask = flowdir.data == 1
        U, D = flowdir.row[ones_mask], flowdir.col[ones_mask]
    else:
        U, D = np.where(flowdir == 1)

    # 2. Identify active basin pixels using C-contiguous memory layout
    mask_flat = mask.ravel(order="C")
    in_basin = mask_flat == 1.0

    # 3. Keep only pairs where both upstream and downstream pixels are inside the basin
    valid_pairs = in_basin[U] & in_basin[D]
    UU_1d = U[valid_pairs]
    DD_1d = D[valid_pairs]

    # 4. Outlet pixel: first downstream node (in DD order) that is not an upstream node
    outlet_candidates = DD_1d[~np.isin(DD_1d, UU_1d)]
    Ioutlet_1d = int(outlet_candidates[0]) if len(outlet_candidates) > 0 else -1

    # Convert 1D C-order indices to 2D row/col coordinate tuples
    Iupstream = np.unravel_index(UU_1d, mask.shape, order="C")
    Idownstream = np.unravel_index(DD_1d, mask.shape, order="C")

    if Ioutlet_1d >= 0:
        r_out, c_out = np.unravel_index(Ioutlet_1d, mask.shape, order="C")
        Ioutlet = (int(r_out), int(c_out))
    else:
        Ioutlet = (-1, -1)

    return Iupstream, Idownstream, Ioutlet

def manning_roughness_mean_to_pixel(n_mean: float, slope_deg: np.ndarray) -> np.ndarray:
    """Distribute mean Manning roughness over the basin based on local slope."""
    slope_rad = np.radians(slope_deg)
    slope_deg_mean = float(np.nanmean(slope_deg))
    slope_rad_mean = np.radians(slope_deg_mean)

    tan_slope = np.tan(slope_rad)
    tan_slope_mean = np.tan(slope_rad_mean)

    if tan_slope_mean == 0:
        return np.full_like(slope_deg, n_mean)

    return n_mean * (tan_slope / tan_slope_mean) ** (1.0 / 3.0)

def matlab_ismember(A, B, zero_based=True):
    """
    Replicates MATLAB [tf, loc] = ismember(A, B) for sorted array B (like Imask).
    """
    # Perform binary search
    idx = np.searchsorted(B, A)
    
    # Clamp indices to prevent out-of-bound array checks
    idx_clamped = np.clip(idx, 0, len(B) - 1)
    
    # Confirm exact value match
    mask_match = (B[idx_clamped] == A)
    
    if zero_based:
        # Python 0-based: matches -> 0 to N-1 | non-matches -> -1
        loc = np.where(mask_match, idx, -1)
    else:
        # MATLAB 1-based: matches -> 1 to N   | non-matches -> 0
        loc = np.where(mask_match, idx + 1, 0)
        
    return mask_match, loc

def muskingum_cunge(Qij: float, Qij1: float, Qi1j: float, flow: float, Ck: float, Bi: float, Si: float, dx: float, dt: float
                    ) -> float:
    """
    Description: This function applies the Muskingum-Cunge method between upstream and
    downstream pixels. Inputs can include a set of upstream and downstream
    pairs, i.e. across a basin.

    Inputs:
        Qij = (in)flow at spatial location i (upstream location) and time step j,
                [m^3/s]
        Qi1j = (out)flow at spatial location i+1 (downtream location) and time
                step j, [m^3/s]
        Qij1 = (in)flow at spatial location i (upstream location) and time step
                j+1, [m^3/s]
        flow = nominal flow in channel [m^3/s]
        Ck = celerity (m/s)
        Bi = width of channel [meters]
        Si = bed slope [-]
        dx = spatial discretization ([meters]
        dt = time step [in hours]

    Outputs:
        Qi1j1 = routed (out)flow at spatial location i+1 (downstream location)
                and time step j+1, [m^3/s]
    """
    # Convert time step to seconds
    dt_sec = dt * 3600.0

    # Compute Muskingum-Cunge parameters
    K = dx / Ck
    X = 0.5 * (1.0 - flow / (Bi * Ck * Si * dx))
    # Impose a check on X. Mays says that for numerical stability, 0<X<0.5.
    if X < 0.0:
        warnings.warn("X coefficient is < 0.0 ... setting equal to 0.0")
        X = 0.0
    if X > 0.5:
        warnings.warn("X coefficient is > 0.5 ... setting equal to 0.5")
        X = 0.5

    # Perform stability check. This uses the USACE stability quoted in Mays,
    # i.e. 1/(2(1-X)) <= K/dt <= 1/(2X). Rearranged in terms of celerity to
    # give the upper and lower limits shown below.
    # Reset Ck to upper bound if it is exceeding it.
    if Ck >= 2.0 * (1.0 - X) * dx / dt_sec:
        Ck = 2.0 * (1.0 - X) * dx / dt_sec
        # Recompute K parameter.
        K = dx / Ck
    # Reset Ck to lower bound if it is below it.
    elif Ck <= 2.0 * X * dx / dt_sec:
        Ck = 2.0 * X * dx / dt_sec
        # Recompute K parameter.
        K = dx / Ck

    # Compute M-C Coefficients
    denom = 2.0 * K * (1.0 - X) + dt_sec
    C1 = (dt_sec - 2.0 * X * K) / denom
    C2 = (dt_sec + 2.0 * X * K) / denom
    C3 = (2.0 * K * (1.0 - X) - dt_sec) / denom

    # Route flow
    Qi1j1 = C1 * Qij1 + C2 * Qij + C3 * Qi1j

    # Make sure flow is non-negative.
    if Qi1j1 < 0.0:
        Qi1j1 = 0.0

    return float(Qi1j1)

def routing_celerity_check(INFLOW_old: np.ndarray, INFLOW_new: np.ndarray, OUTFLOW_old: np.ndarray, n: np.ndarray, bed_slope: np.ndarray,  dx: float, dt: float, width: np.ndarray
                           ) -> int:
    """
    Description: This function checks the maximum flow velocity expected in the
    channel network to determine whether a reduced time step is needed for
    routing. Can be turned on/off via input flag.

    Inputs:
        INFLOW_old: inflow array (map) at previous time step, i  [m^3/s]
        INFLOW_new: inflow array (map) at current time step, i+1;
                    this is the model generated runoff at each pixel [m^3/s]
        OUTFLOW_old: outflow array (map) at previous time step, i  [m^3/s]
        n: Manning roughness coefficient array  (map)
        bed_slope: the slope of stream channels array [-]
        dx : spatial resolution of the grid [m]
        dt : length of timestep [hours]
        width: channel width array (map), i.e. channel width at each pixel.

    Outputs:
        dt_ratio: Ratio by which time step should be reduced to constrain the
                  flow to within one channel reach (dx).
    """
    # Determine nominal flow for use in Manning Eqn.
    # Use three-point average.
    nominal_flow = (INFLOW_new + INFLOW_old + OUTFLOW_old) / 3.0  # m^3/s

    # Takeuchi and others often invoke a "wide" rectangular channel assumption
    # (i.e. width >> depth) in which case the problem becomes non-iterative.
    # Variable celerity calculation
    # assume "wide" rectangular cross-section and use Manning Equation to
    # compute the celerity (dQ/dA).
    # compute wetted perimeter
    Pw = width  # meters; wide channel assumption (width>>depth)
    # Note: Don't actually need to compute depth, if so, could use equation
    # below:
    depth = (n * nominal_flow / width / np.sqrt(bed_slope)) ** (3.0 / 5.0)  # meters
    nominal_velocity = nominal_flow / (Pw * depth)

    # The coefficients below are based on the expression: Q=a*A^b, which is
    # given by Manning Equation.

    # Check velocity vs. space/time discretization:
    max_vel = float(np.nanmax(nominal_velocity))
    if max_vel > dx / (dt * 3600.0):  # m/s
        # Ratio of travel time step to model timestep
        dt_ratio = int(round(max_vel / (dx / (dt * 3600.0))))
    else:
        dt_ratio = 1

    return dt_ratio

# New version that is based on 2D row/col tuples
def routing_muskingum_cunge(INFLOW_old: np.ndarray, INFLOW_new: np.ndarray, OUTFLOW_old: np.ndarray, Iupstream: tuple[np.ndarray, np.ndarray], Idownstream: tuple[np.ndarray, np.ndarray],
                            mask: np.ndarray, n: np.ndarray, bed_slope: np.ndarray, dx: float, dt: float, width: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Description:
    The spatially-distributed routing employed in this model routes flow
    through a gridded network that is determined using raster DEM data. The
    routing method assumes that ALL pixels within a basin are connected to
    the basin's flow network. There are no assumptions made about the
    location of a stream/channel network which would only connect a subset of
    the basin pixels. Instead it is therefore assumed that each pixel
    contains a "stream" and is connected to the larger basin network linking
    all pixels. The approach is patterned after what is done in Takeuchi et
    al. and related references.

    The outlet pixel is only a downstream pixel. All other pixels in the
    basin can be either an upstream pixel or both an upstream and downstream
    pixel in the gridded flow network. A given pixel can serve as a
    downstream pixel for multiple upstream pixels. Upstream and downstream
    pixels are determined, stored, and used in pairs. Therefore, the order of
    the upstream pixels and their corresponding downstream counterpart must
    be preserved.

    This function uses the Muskingum Cunge scheme to route flow from an
    upstream pixel to the pixel immediately downstream during a
    given time step. This is based on the flow network defined for a basin.

    Inputs:
        INFLOW_old: inflow array (map) at previous time step, i  [m^3/s]
        INFLOW_new: inflow array (map) at current time step, i+1;
                    this is the model generated runoff at each pixel [m^3/s]
        OUTFLOW_old: outflow array (map) at previous time step, i  [m^3/s]

        UPDATE:
        The indices below are 2D (row, col) tuple indexes, not linear indices:

            Iupstream: upstream pixel indices array (paired with Idownstream)
            Idownstream: downstream pixel indices array (paired with Iupstream)
            mask: watershed mask array (map)
            n: Manning roughness coefficient array  (map)
            bed_slope: the slope of stream channels array [-]
            dx : spatial resolution of the grid [m]
            dt : length of timestep [hours]
            width: channel width array (map), i.e. channel width at each pixel.

    Outputs:
    NEW_INFLOWS: Updated inflows based on predicted upstream outflows and
                   the pixel-generated runoff.  [m^3/s]
    NEW_OUTFLOWS: The predicted outflow from each individual stream reach.
                   [m^3/s]
    """
    ny, nx = mask.shape

    # Determine nominal flow (3-point average)
    nominal_flow = (INFLOW_new + INFLOW_old + OUTFLOW_old) / 3.0  # m^3/s

    # Manning equation parameters
    Pw = width  # meters
    b = 3.0 / 5.0
    a = (n * Pw ** (2.0 / 3.0) / np.sqrt(bed_slope)) ** b
    dQdA = 1.0 / (a * b * nominal_flow ** (b - 1.0))
    celerity = dQdA  # (m/s)

    # Initialize flow matrix maps
    NEW_INFLOWS = np.zeros((ny, nx), dtype=float) * mask
    NEW_OUTFLOWS = np.zeros((ny, nx), dtype=float) * mask

    up_rows, up_cols = Iupstream
    down_rows, down_cols = Idownstream

    # Vectorized inflow evaluation at upstream nodes
    inflow_old_up = INFLOW_old[up_rows, up_cols]
    inflow_new_up = INFLOW_new[up_rows, up_cols]

    # Find active flow indices
    Iflow = np.where((inflow_old_up > 0) | (inflow_new_up > 0))[0]

    # Loop over active flow pairs
    for j in range(len(Iflow)):
        i = Iflow[j]

        r_up, c_up = up_rows[i], up_cols[i]
        r_down, c_down = down_rows[i], down_cols[i]

        OUTFLOW_new = muskingum_cunge(
            Qij=INFLOW_old[r_up, c_up],
            Qij1=INFLOW_new[r_up, c_up],
            Qi1j=OUTFLOW_old[r_up, c_up],
            flow=nominal_flow[r_up, c_up],
            Ck=celerity[r_up, c_up],
            Bi=width[r_up, c_up],
            Si=bed_slope[r_up, c_up],
            dx=dx,
            dt=dt,
        )

        NEW_OUTFLOWS[r_up, c_up] = OUTFLOW_new
        NEW_INFLOWS[r_down, c_down] += OUTFLOW_new

    return NEW_INFLOWS, NEW_OUTFLOWS

# Old MATLAB version that uses 1D indices
def routing_muskingum_cunge_old(INFLOW_old: np.ndarray, INFLOW_new: np.ndarray, OUTFLOW_old: np.ndarray, Iupstream: np.ndarray, Idownstream: np.ndarray, mask: np.ndarray, 
                            n: np.ndarray, bed_slope: np.ndarray, dx: float, dt: float, width: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Description:
    The spatially-distributed routing employed in this model routes flow
    through a gridded network that is determined using raster DEM data. The
    routing method assumes that ALL pixels within a basin are connected to
    the basin's flow network. There are no assumptions made about the
    location of a stream/channel network which would only connect a subset of
    the basin pixels. Instead it is therefore assumed that each pixel
    contains a "stream" and is connected to the larger basin network linking
    all pixels. The approach is patterned after what is done in Takeuchi et
    al. and related references.

    The outlet pixel is only a downstream pixel. All other pixels in the
    basin can be either an upstream pixel or both an upstream and downstream
    pixel in the gridded flow network. A given pixel can serve as a
    downstream pixel for multiple upstream pixels. Upstream and downstream
    pixels are determined, stored, and used in pairs. Therefore, the order of
    the upstream pixels and their corresponding downstream counterpart must
    be preserved.

    This function uses the Muskingum Cunge scheme to route flow from an
    upstream pixel to the pixel immediately downstream during a
    given time step. This is based on the flow network defined for a basin.

    Inputs:
        INFLOW_old: inflow array (map) at previous time step, i  [m^3/s]
        INFLOW_new: inflow array (map) at current time step, i+1;
                    this is the model generated runoff at each pixel [m^3/s]
        OUTFLOW_old: outflow array (map) at previous time step, i  [m^3/s]

        The indices below are all linear indices, not row/col. indices:

            Iupstream: upstream pixel indices vector (paired with Idownstream)
            Idownstream: downstream pixel indices vector (paired with Iupstream)
            mask: watershed mask array (map)
            n: Manning roughness coefficient array  (map)
            bed_slope: the slope of stream channels array [-]
            dx : spatial resolution of the grid [m]
            dt : length of timestep [hours]
            width: channel width array (map), i.e. channel width at each pixel.

    Outputs:
    NEW_INFLOWS: Updated inflows based on predicted upstream outflows and
                   the pixel-generated runoff.  [m^3/s]
    NEW_OUTFLOWS: The predicted outflow from each individual stream reach.
                   [m^3/s]
    """
    # Determine dimensions of domain
    ny, nx = mask.shape

    # Determine nominal flow for use in Manning Eqn.
    # Use three-point average.
    nominal_flow = (INFLOW_new + INFLOW_old + OUTFLOW_old) / 3.0  # m^3/s

    # Takeuchi and others often invoke a "wide" rectangular channel assumption
    # (i.e. width >> depth) in which case the problem becomes non-iterative.
    # Variable celerity calculation
    # assume "wide" rectangular cross-section and use Manning Equation to
    # compute the celerity (dQ/dA).
    # compute wetted perimeter
    Pw = width  # meters; wide channel assumption (width>>depth)
    # Note: Don't actually need to compute depth, if so, could use equation
    # below:
    # depth=(n.*nominal_flow./width./bed_slope.^(0.5)).^(3/5); % meters
    # nominal_velocity=nominal_flow./(Pw.*depth);

    # The coefficients below are based on the expression: Q=a*A^b, which is
    # given by Manning Equation.
    b = 3.0 / 5.0
    a = (n * Pw ** (2.0 / 3.0) / np.sqrt(bed_slope)) ** b
    dQdA = 1.0 / (a * b * nominal_flow ** (b - 1.0))
    celerity = dQdA  # (m/s)

    # Initialize flow matrix maps
    NEW_INFLOWS = np.zeros((ny, nx), dtype=float) * mask
    NEW_OUTFLOWS = np.zeros((ny, nx), dtype=float) * mask

    # Route flow if there is flow present in the basin
    # indices of pixels where inflow at t is positive
    Iflow1 = np.where(INFLOW_old.flat[Iupstream] > 0)[0]
    # indices of pixels where inflow at t+1 is positive
    Iflow2 = np.where(INFLOW_new.flat[Iupstream] > 0)[0]
    Iflow = np.union1d(Iflow1, Iflow2)

    # loop over upstream pixels contributing to a given downstream pixel.
    for j in range(len(Iflow)):
        # Current index
        i = Iflow[j]

        # Route flow from the upstream pixel Iupstream(i) to the downstream
        # pixel (Idownstream(i). This returns the predicted outflow at the
        # current timestep (only the size of the flow generating pixels).
        idx_up = Iupstream[i]
        idx_down = Idownstream[i]

        OUTFLOW_new = muskingum_cunge(
            Qij=INFLOW_old.flat[idx_up],
            Qij1=INFLOW_new.flat[idx_up],
            Qi1j=OUTFLOW_old.flat[idx_up],
            flow=nominal_flow.flat[idx_up],
            Ck=celerity.flat[idx_up],
            Bi=width.flat[idx_up],
            Si=bed_slope.flat[idx_up],
            dx=dx,
            dt=dt,
        )

        # Store the predicted outflows. These are the outflows for each single
        # reach, i.e. does not include outflow from multiple reaches
        # contributing to a downstream reach. Note: This variable cannot be
        # saved in downstream indices because Idownstream(i) can be a
        # downstream pixel for several upstream pixels. Instead, it is stored
        # in upstream indices for convenience.
        NEW_OUTFLOWS.flat[idx_up] = OUTFLOW_new

        # Save sum of water outflows from each upstream pixel. This would be
        # part of the inflow of the downstream pixel at the next time step.
        # For each of the outflows add it to the downstream inflows. Note: This
        # may be the result of outflow from several upstream pixels.
        NEW_INFLOWS.flat[idx_down] += OUTFLOW_new

    return NEW_INFLOWS, NEW_OUTFLOWS

def TOPMODEL(INFIL: np.ndarray, ET: np.ndarray, Srz0: np.ndarray, Srzmax: np.ndarray, Srzmin: np.ndarray, Suz0: np.ndarray, SD0: np.ndarray, T0: np.ndarray, 
             slope_deg: np.ndarray, dx: float, mask: np.ndarray, lambda_mean: float, lambda_val: np.ndarray, m: float, K0: np.ndarray, dt: float
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Description:
    This function uses the TOPMODEL soil moisture balance and saturation excess runoff
    and baseflow based on Takeuchi et al., HSJ 1999.

    Inputs:
        INFIL: Infiltration rate (m/hr)
        ET: Evapotranspiration rate (m/hr)
        Srz0: Rootzone storage (m)
        Srzmax: Maximum rootzone storage (m)
        Srzmin: Minimum rootzone storage (m)
        Suz0: Unsaturated zone storage (m)
        SD0: Storage deficit (m)
        T0: Transmissivity (m^2/hr)
        slope_deg: Slope (degrees)
        dx: Pixel size in x-direction (m)
        mask: Watershed mask
        lambda_mean: Mean soil-topographic index
        lambda_val: Soil-topographic index
        m: Exponential decay parameter for transmissivity (m)
        K0: Saturated hydraulic conductivity (m/hr)
        dt: Timestep duration (hr)

    Outputs:
        Srz_new: Rootzone storage at end of time step (m)
        Suz_new: Unsaturated zone storage at end of time step (m)
        SD_new: Storage deficit at end of time step (m)
        qv: Recharge flux (m)
        qb: Baseflow (m^2/hr)
        qse_new: Saturation excess runoff (m)
        Qv: Total basin-averaged (recharge) flux to groundwater (m)
        Qb: Total basin-averaged baseflow (m)
    """
    # Allocate variables
    Srz_new = np.full_like(SD0, np.nan, dtype=float)
    Suz_new = np.full_like(SD0, np.nan, dtype=float)
    qv = np.full_like(SD0, np.nan, dtype=float)
    qse_new = np.full_like(SD0, np.nan, dtype=float)
    qb = np.full_like(SD0, np.nan, dtype=float)

    # Only run on mask
    imask = mask == 1

    # Rootzone mass balance
    # Note this allows Srz_new to go above Srzmax which is done to
    # compute drainage flux below)
    Srz_new[imask] = np.maximum(
        (Srz0[imask] + INFIL[imask] * dt - ET[imask] * dt), Srzmin[imask]
    )  # meters

    # Unsaturated zone mass balance
    Suz_new[imask] = Suz0[imask] + np.maximum((Srz_new[imask] - Srzmax[imask]), 0.0)  # meters
    # Should then set Srz to Srzmax for those pixels that exceed it
    Srz_new[imask] = np.minimum(Srz_new[imask], Srzmax[imask])
    # drainage (recharge) flux
    qv[imask] = np.minimum(K0[imask] * np.exp(-SD0[imask] / m) * dt, Suz_new[imask])  # meters
    # Subtract recharge from unsat. zone
    Suz_new[imask] = Suz_new[imask] - qv[imask]  # meters

    # Overland flow
    # (note saturation excess generated if Suz exceeds SD)
    qse_new[imask] = np.maximum((Suz_new[imask] - SD0[imask]), 0.0)  # meters
    # Note this sets Suz equal to SD
    Suz_new[imask] = Suz_new[imask] - qse_new[imask]

    # Groundwater flow
    qb[imask] = T0[imask] * np.exp(-SD0[imask] / m) * np.tan(np.deg2rad(slope_deg[imask]))  # flow per unit contour length: m^2/hour

    # Basin-scale mass balance
    # Total flux to GW
    Qv = float(np.nanmean(qv))  # meters

    # Total baseflow leaving aquifer
    # Note: Technically L_i should be the length of stream channel receiving
    # baseflow. If applied pixelwise this should be on the order of the pixel
    # length scale.
    L_i = dx
    Qb = float(np.nanmean(qb / L_i) * dt)  # meters

    # Basin-wide saturation deficit
    # Note this takes the mean of SD0 (which had negative values removed)
    # rather than using the previously computed SD_mean
    SDmean_new = float(np.nanmean(SD0)) - Qv + Qb

    # Update pixel-wise saturation deficit
    SD_new = SDmean_new + m * (lambda_mean - lambda_val)

    # Set saturation excess runoff for cases with negative SD
    neg_SD = SD_new < 0
    qse_new[neg_SD] = qse_new[neg_SD] + np.abs(SD_new[neg_SD])  # meters
    # Reset SD to zero at those locations
    SD_new[neg_SD] = 0.0
    # Set Suz0 to zero where SD0 is zero
    Suz_new[SD_new == 0] = 0.0

    return Srz_new, Suz_new, SD_new, qv, qb, qse_new, Qv, Qb

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


