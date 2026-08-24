import sys
import warnings
from pathlib import Path
from typing import Union, Optional
import numpy as np
import pandas as pd
import xarray as xr

from scipy.interpolate import interp1d
from scipy.ndimage import binary_erosion
from scipy import sparse
from scipy.sparse.linalg import spsolve
from skimage.morphology import reconstruction

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# link core repo folder
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# import functions from other chapters
from src.chapter3 import compute_shade_lookup_table_and_SVF, generate_slope_and_aspect_from_DEM

def imfill_holes(elev: np.ndarray) -> np.ndarray:
    """
    Fills depressions/holes in a grayscale DEM matching MATLAB's `imfill(elev, 8, 'holes')`.
    """
    # 1. create copy of elevation profile 
    mask = elev.copy()
    
    # 1. Initialize seed (marker) array entirely to infinity
    seed = np.full_like(mask, np.inf)
    
    # 2. Copy ONLY the actual boundary values of the DEM to the seed borders
    seed[0, :] = mask[0, :]
    seed[-1, :] = mask[-1, :]
    seed[:, 0] = mask[:, 0]
    seed[:, -1] = mask[:, -1]
    
    # 3. Define 8-connectivity footprint (3x3 array of True values)
    footprint = np.ones((3, 3), dtype=bool)
    
    # 4. Perform morphological reconstruction by erosion
    filled_elev = reconstruction(seed, mask, method='erosion', footprint=footprint)
    
    return filled_elev

def ixneighbors(dem: np.ndarray, 
                ix: Optional[Union[np.ndarray, list]] = None, 
                nhood: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds neighbor pairs in a 2D grid using MATLAB column-major (F-style) ordering.
    Matches ixneighbors_original exactly when called with single input `ixneighbors(dem)`.

    Parameters
    ----------
    dem : np.ndarray
        2D array of elevations (shape: [rows, cols]).
    ix : np.ndarray, list, or None, optional
        - None: returns neighbors for all valid (non-NaN) cells in dem.
        - 2D boolean mask (same shape as dem): returns neighbors for True cells.
        - 1D array/list of linear indices: returns neighbors for specified cells.
    nhood : int, default 8
        Neighborhood connectivity: 4 or 8 directions.

    Returns
    -------
    ic : np.ndarray (1D)
        Linear indices of source cells.
    icd : np.ndarray (1D)
        Linear indices of neighbor cells.
    """
    rows, cols = dem.shape
    nrc = rows * cols

    # Define shifts based on nhood parameter
    if nhood == 8:
        shifts = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1)
        ]
    elif nhood == 4:
        shifts = [
                      (-1, 0),
            ( 0, -1),          ( 0, 1),
                      ( 1, 0)
        ]
    else:
        raise ValueError("nhood must be either 4 or 8.")

    # Convert `ix` argument into a 2D boolean mask if provided
    ix_mask = None
    if ix is not None:
        ix_arr = np.asarray(ix)
        if ix_arr.dtype == bool:
            if ix_arr.shape != dem.shape:
                raise ValueError("If ix is a boolean mask, it must match dem.shape.")
            ix_mask = ix_arr
        else:
            # Convert linear indices to 2D boolean mask
            ix_mask = np.zeros((rows, cols), dtype=bool)
            r_idx, c_idx = np.unravel_index(ix_arr, (rows, cols), order='F')
            ix_mask[r_idx, c_idx] = True

    # Grid of linear indices in Fortran ('F') column-major order
    G = np.arange(nrc, dtype=int).reshape((rows, cols), order='F')

    ic1_list = []
    icd1_list = []

    dem_flat = dem.flatten(order='F')

    for r_shift, c_shift in shifts:
        # Valid source coordinates (prevent boundary wrap-around)
        r_src = np.arange(max(0, -r_shift), min(rows, rows - r_shift))
        c_src = np.arange(max(0, -c_shift), min(cols, cols - c_shift))

        # Extract 2D subgrids for source and destination
        src_grid = G[np.ix_(r_src, c_src)]
        dst_grid = G[np.ix_(r_src + r_shift, c_src + c_shift)]

        src_flat = src_grid.flatten(order='F')
        dst_flat = dst_grid.flatten(order='F')

        # Base valid mask: NaN filtering
        valid_mask = ~np.isnan(dem_flat[src_flat]) & ~np.isnan(dem_flat[dst_flat])

        # If a subset mask `ix` was provided, apply it to source cells
        if ix_mask is not None:
            ix_mask_flat = ix_mask.flatten(order='F')
            valid_mask &= ix_mask_flat[src_flat]

        ic1_list.append(src_flat[valid_mask])
        icd1_list.append(dst_flat[valid_mask])

    if not ic1_list:
        return np.array([], dtype=int), np.array([], dtype=int)

    ic1 = np.concatenate(ic1_list)
    icd1 = np.concatenate(icd1_list)

    # Sort by (ic1, icd1) to match MATLAB and ixneighbors_original order exactly
    sort_idx = np.lexsort((icd1, ic1))
    return ic1[sort_idx], icd1[sort_idx]

def validate_wflowacc_inputs(X, Y, dem, type_flag, edges, exponent, routeflats, mode):
    """
    Validates wflowacc inputs against allowed shapes, types, and values.
    Note that the W0 input is handled in the compute_flow_acc function
    """
    # 1. Validate required 2D array inputs and shapes
    if not isinstance(dem, np.ndarray) or dem.ndim != 2:
        raise ValueError("`dem` must be a 2D numpy array.")
    
    if not isinstance(X, np.ndarray) or X.shape != dem.shape:
        raise ValueError("`X` must be a numpy array with the exact same shape as `dem`.")
        
    if not isinstance(Y, np.ndarray) or Y.shape != dem.shape:
        raise ValueError("`Y` must be a numpy array with the exact same shape as `dem`.")

    # 2. Validate type_flag
    allowed_types = ('multi', 'single')
    type_flag_clean = str(type_flag).lower().strip()
    if type_flag_clean not in allowed_types:
        raise ValueError(f"Invalid `type_flag` '{type_flag}'. Must be one of {allowed_types}.")

    # 3. Validate edges
    allowed_edges = ('open', 'closed')
    edges_clean = str(edges).lower().strip()
    if edges_clean not in allowed_edges:
        raise ValueError(f"Invalid `edges` option '{edges}'. Must be one of {allowed_edges}.")

    # # 4. Validate and initialize default W0
    # if W0 is None:
    #     W0 = np.ones_like(dem, dtype=float)
    # else:
    #     if not isinstance(W0, np.ndarray) or W0.shape != dem.shape:
    #         raise ValueError("`W0` must be a numpy array with the exact same shape as `dem`.")

    # 5. Validate exponent
    if exponent is not None:
        if not isinstance(exponent, (int, float, np.number)) or exponent <= 0:
            raise ValueError("`exponent` must be a positive number or None.")
        exponent = float(exponent)

    # 6. Validate routeflats
    allowed_routeflats = ('yes', 'no')
    routeflats_clean = str(routeflats).lower().strip()
    if routeflats_clean not in allowed_routeflats:
        raise ValueError(f"Invalid `routeflats` option '{routeflats}'. Must be 'yes' or 'no'.")

    # 7. Validate mode
    allowed_modes = ('default', 'random', 'randomized')
    mode_clean = str(mode).lower().strip()
    if mode_clean not in allowed_modes:
        raise ValueError(f"Invalid `mode` option '{mode}'. Must be one of {allowed_modes}.")

    return (
        X, Y, dem, 
        type_flag_clean, 
        edges_clean,
        exponent, 
        routeflats_clean, 
        mode_clean
    )

def route_flats(M_raw: sparse.csr_matrix, 
                dem: np.ndarray, 
                Ad: sparse.csr_matrix, 
                routeflats: str = 'yes') -> tuple[sparse.csr_matrix, int]:
    """
    Iteratively resolves routing through flats matching MATLAB TopoToolbox.
    
    Parameters
    ----------
    M_raw : sparse.csr_matrix
        The unnormalized positive slope matrix (shape: nrc x nrc).
    dem : np.ndarray
        2D digital elevation model array.
    Ad : sparse.csr_matrix
        Adjacency matrix (1 for neighbor connections).
    routeflats : str
        'yes' or 'no' flag.
        
    Returns
    -------
    M: sparse.csr_matrix
        Updated M_raw matrix with artificial gradients added across flats.
    run_count: integer
        number of times interior flat regions needed to be rerouted
    """
    if routeflats.lower() != 'yes':
        return M_raw

    # Copy M_raw to avoid modifying the input in-place
    M = M_raw.copy()
    
    # 2D DEM shape and total size
    dem_shape = dem.shape
    nrc = dem.size
    
    # Mask of NaNs (matching nans = isnan(dem))
    nans = np.isnan(dem).flatten(order='F')
    
    flagflats = True
    run_count = 0
    
    while flagflats:
        run_count += 1
        
        # 1. Find cells that "do not give" (sum of row in M == 0)
        row_sums = np.array(M.sum(axis=1)).flatten()
        ing = np.where(row_sums == 0)[0]
        
        # Remove NaN cells from ing
        ing = ing[~nans[ing]]
        
        if len(ing) == 0:
            break

        # 2. Identify interior flats (a == True if all 8 neighbors are valid/present)
        # full(sum(sparse(ing,ing,1,nrc,nrc)*Ad, 2) == 8)
        ing_diag = sparse.csr_matrix((np.ones(len(ing)), (ing, ing)), shape=(nrc, nrc))
        a = (np.array((ing_diag @ Ad).sum(axis=1)).flatten() == 8)

        # 3. Neighbors of flats
        # b = full(Ad * a)
        b = np.array(Ad @ a.astype(float)).flatten()

        # inb_flats = reshape(b < 8 & a, siz)
        inb_flats = ((b < 8) & a).reshape(dem_shape, order='F')
        
        # IX_outb_flats = find(b & ~a)
        IX_outb_flats = np.where((b > 0) & (~a))[0]

        if len(IX_outb_flats) == 0 or not np.any(inb_flats):
            break

        # 4. Query neighbors of inb_flats to find candidate connections where dem(ic) == dem(icd)
        ic, icd = ixneighbors(dem, ix=inb_flats)
        
        dem_flat = dem.flatten(order='F')
        same_elev_mask = (dem_flat[ic] == dem_flat[icd])
        ic = ic[same_elev_mask]
        icd = icd[same_elev_mask]

        if len(ic) == 0:
            break

        # 5. Check if icd targets are in IX_outb_flats (ismembc equivalent for sorted arrays)
        # Using np.isin to check if target cells are sill outlets
        i_mask = np.isin(icd, IX_outb_flats)

        if np.any(i_mask):
            ic = ic[i_mask]
            icd = icd[i_mask]

            # 6. Add artificial gradient (0.01) from flat cells (ic) to sills (icd)
            delta_M = sparse.csr_matrix((np.full(len(ic), 0.01), (ic, icd)), shape=(nrc, nrc))
            M = M + delta_M
            
            flagflats = True
        else:
            flagflats = False

    return M, run_count

def apply_flow_mode(M_raw: sparse.csr_matrix, 
                    mode: str = 'default', 
                    rc: float = 0.01) -> sparse.csr_matrix:
    """
    Applies deterministic, random, or randomized mode to the unnormalized flow matrix M_raw,
    matching MATLAB TopoToolbox sprandn behavior.

    Parameters
    ----------
    M_raw : sparse.csr_matrix
        The unnormalized positive slope matrix (shape: nrc x nrc).
    mode : str, default 'default'
        - 'default' / 'deterministic': keeps original slopes intact.
        - 'random': replaces slopes with absolute normal random values at non-zero locations.
        - 'randomized': adds scaled random noise to existing slopes.
    rc : float, default 0.01
        Randomization coefficient used in 'randomized' mode.

    Returns
    -------
    sparse.csr_matrix
        Updated M_raw matrix.
    """
    mode_clean = mode.lower()
    
    if mode_clean in ('default'):
        return M_raw

    # Copy matrix to ensure we don't modify the input in-place
    M = M_raw.copy()
    
    # Generate standard normal random values matching M's non-zero entries: abs(sprandn(M))
    num_nonzeros = M.nnz
    abs_rand_data = np.abs(np.random.standard_normal(num_nonzeros))

    if mode_clean == 'random':
        # M = abs(sprandn(M))
        M.data = abs_rand_data

    elif mode_clean == 'randomized':
        # M = M + (rc * abs(sprandn(M)))
        M.data = M.data + (rc * abs_rand_data)

    return M

def build_single_flow_M(M_raw: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Constructs a single-flow (D8) routing matrix matching TopoToolbox tie-breaking.
    """
    # Convert to CSR for fast row iteration
    csr = M_raw.tocsr()
    nrc = csr.shape[0]
    
    rows = []
    cols = []
    
    for r in range(nrc):
        start = csr.indptr[r]
        end = csr.indptr[r + 1]
        
        if start == end:
            continue
            
        row_data = csr.data[start:end]
        max_val = row_data.max()
        
        if max_val > 0:
            # Find indices of max values in this row
            max_mask = (row_data == max_val)
            
            # TIE-BREAKER: Pick the FIRST neighbor that achieves max slope
            first_max_local_idx = np.argmax(max_mask)
            
            # Destination column index in global matrix
            dst_col = csr.indices[start + first_max_local_idx]
            
            rows.append(r)
            cols.append(dst_col)
            
    # Assign 1.0 strictly to the winning single neighbor
    data = np.ones(len(rows), dtype=float)
    return sparse.csr_matrix((data, (rows, cols)), shape=(nrc, nrc))

def compute_flow_acc(M: sparse.csr_matrix, 
                     dem_shape: tuple,
                     edges: str = 'closed',
                     W0: np.ndarray = None,
                     edgecorrection: np.ndarray = None) -> np.ndarray:
    """
    Computes flow accumulation matching MATLAB TopoToolbox wflowacc.
    Returns a 2D numpy array reshaped to dem_shape.

    Note: To validate with MATLAB, Fortran ordering is enforced.

    """
    nrc = dem_shape[0] * dem_shape[1]  # numel(dem)
    
    # 1. Initialize W0 matching ones(size(dem))
    if W0 is None:
        W0_flat = np.ones(nrc, dtype=float)
    else:
        # Flatten input weight matrix in Fortran order
        W0_flat = W0.flatten(order='F')
        
    # 2. Build Identity Matrix
    I = sparse.identity(nrc, format='csr')
    
    # 3. Construct System Matrix based on edge choice
    if edges == 'closed':
        SystemMatrix = I - M.T.tocsr()
        
    elif edges == 'open':
        if edgecorrection is None:
            raise ValueError("edgecorrection vector is required when edges='open'.")
        
        # Build diagonal matrix spdiags(edgecorrection, 0, nrc, nrc)
        D_edge = sparse.diags(edgecorrection, offsets=0, shape=(nrc, nrc), format='csr')
        
        # System: (I - D_edge * M')
        SystemMatrix = I - (D_edge @ M.T).tocsr()
    else:
        raise ValueError("edges must be either 'closed' or 'open'")
        
    # 4. Solve linear system (SystemMatrix \ W0)
    flowacc_flat = spsolve(SystemMatrix, W0_flat)
    
    # 5. Reshape to original 2D grid shape (matching MATLAB reshape(flowacc, siz))
    flowacc_2d = flowacc_flat.reshape(dem_shape, order='F')
    
    return flowacc_2d

def wflowacc(X: np.ndarray, Y: np.ndarray, dem: np.ndarray, 
             type_flag: str = 'multi', edges: str = 'closed', W0: np.ndarray = None,
             exponent: float = None, routeflats='yes', mode = 'default') -> tuple[np.ndarray, sparse.csr_matrix, sparse.csr_matrix, int]:
    """
    Multiple flowdirection and flowaccumulation algorithm that routes 
    through flat terrain (not sinks). Remove sinks with the function imfill_holes.
    Matirx operations performed using column-major (F-style) ordering to validate with MATLAB.
    
    Inputs:
    -----------
    X,Y             coordinate matrices created by meshgrid

    dem             digital elevation model same size as X and Y

    type_flag       'multi' (default): multiple flowdirection (dinf)
                    'single': single flow direction (d8). Flow occurs only along the steepest descent.

    edges           decide on how to handle flow on grid edges. 
                    'closed' (default) forces all water to remain on the
                    grid, 'open' assumes that edge cells loose the ratio 
                    r = # of neighbor cells/8 of water.

    W0              W0 is an initiation grid same size as dem and refers 
                    to the water in each cell before routing through the 
                    catchment. By default W0 is a ones matrix with the same size as dem.

    exponent        exponent governing the relation between flow 
                    direction and slope. Default is 1 (not supplied), which means, there 
                    is a linear relation. You may want to increase the
                    exponent when flow direction should rather follow a
                    steepest descent (single) flow direction (e.g. 5). This
                    option is only effective for multiple flowdirection.

    routeflats      'yes' (default) or 'no', decides upon routing over
                    flats/plateaus.
    
    mode            'default': deterministic flow
                    'random': totally random flow to downward neighbors
                    'randomized': deterministic flow with noise
    
    Outputs:
    -----------
    flowacc         flow accumulation (upslope area) grid
    flowdir         flow direction (sparse matrix)
    slope           slope (sparse matrix)
    num_route_flats number of times interior flats needed to be routed (int)
    """

    # Validate inputs before doing anything
    # Note that the W0 input is handled in the compute_flow_acc function
    (
        X, Y, dem, 
        type_flag, 
        edges, 
        exponent, 
        routeflats, 
        mode
    ) = validate_wflowacc_inputs(
        X, Y, dem, type_flag, edges, exponent, routeflats, mode
    )

    nrc = dem.size

    # Extract neighbors using your existing ixneighbors function
    ic1, icd1 = ixneighbors(dem)
    
    # Calculate slopes between neighboring cells
    dist = np.hypot(X.flatten(order='F')[ic1] - X.flatten(order='F')[icd1], Y.flatten(order='F')[ic1] - Y.flatten(order='F')[icd1])
    e = (dem.flatten(order='F')[ic1] - dem.flatten(order='F')[icd1]) / dist
    
    # Slope sparse matrix
    S = sparse.csr_matrix((e, (ic1, icd1)), shape=(nrc, nrc))
    
    # Adjacency matrix
    Ad = sparse.csr_matrix((np.ones_like(ic1), (ic1, icd1)), shape=(nrc, nrc))
    
    # Handle negative slopes (upward neighbors)
    e_pos = np.maximum(e, 0.0)
    
    # Flow direction matrix representation (raw positive slopes)
    M_raw = sparse.csr_matrix((e_pos, (ic1, icd1)), shape=(nrc, nrc))

    # Routing through flats logic
    M_raw, num_route_flats  = route_flats(M_raw, dem, Ad, routeflats)

    # apply flow mode
    M_raw = apply_flow_mode(M_raw, mode)
    
    # Routing logic block
    if type_flag == 'single':
        # Single flow direction (D8): Flow occurs only along the steepest descent
        M = build_single_flow_M(M_raw)
        
    else: # Multi case
        # Apply exponent if supplied
        if exponent is not None and exponent !=1:
            M_raw.data = M_raw.data ** exponent

        # Multiple flow direction (MFD): normalize rows by sum of positive slopes
        row_sums = np.array(M_raw.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero for sinks/outlets

        # Scale each row by 1 / row_sum
        M = sparse.diags(1.0 / row_sums) @ M_raw
    
    # Solve equation based on edges designation
    if edges == 'open':
        # need to supply edge correction
        edge_corr = np.array(Ad.sum(axis=1)).flatten() / 8.0
    else:
        edge_corr = None
    
    flowacc = compute_flow_acc(M=M, dem_shape=dem.shape, edges=edges, W0=W0, edgecorrection=edge_corr)

    return flowacc, M, S, num_route_flats

def watershed_area_and_stream_delineation(easting: np.ndarray, northing: np.ndarray, elev_in: np.ndarray, 
                                         outlet_coordinate: np.ndarray, percent_basin_area: float = 0.01,
                                         save_plots: bool = False, display_plots: bool = False, plots_path: Optional[str | Path] = None,
                                         ) -> tuple[np.ndarray, np.ndarray, sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """
    Description:
    This code is written to take DEM data as an input and construct a mask
    array delineating the watershed. As part of the processing, the flow
    accumulation array is determined (using code by Wolfgang Schwanghart 
    shown below) which can also be used to delineate streams.

    Inputs:
    easting: x coordinate vector
    northing: y coordinate vector
    elev: elevation matrix (at each x,y point, i.e. the DEM)
    outlet_coordinate: 1x2 matrix with the x and y coordinate of the basin
      outlet;
    save_plots: boolean, if True, saves plots to plots_path
    display_plots: boolean, if True, displays plots to screen (save_plots must be set to True)
    plots_path: folder location of where to save plots
    percent_basin_area: The percent of basin area that must contribute to
    flow accumulation for a pixel to be deemed a "stream" pixel; expressed as 
    a fraction (i.e. use 0.01 for 1%).
    
    NOTES: easting and northing should be Nx1 and Mx1 vectors, and
    elev should be a MxN array; outlet coordinates should be in same units of
    easting and northing and should be on a stream channel
    
    Outputs:
    mask: a binary MxN array with 1 values inside delineated basin and 0
        values outside delineated basin
    flowacc: a MxN flow accumulation matrix indicating for a particular pixel
        how many upstream pixels drain into it (including itself)
    flowdir: flow direction matrix, a sparse matrix of 0s and 1s used for flow network computations
    slope: matrix of slope for each pixel
    stream_rows : x-coordinate index for the watershed outline
    stream_cols : y-coordinate index for the watershed outline

    """
    # Fill holes in DEM (requires image processing toolbox equivalent)
    elev = imfill_holes(elev_in)

    # Find matrix indices of outlet (col_coord, row_coord)
    col_coord = np.argmin(np.abs(easting - outlet_coordinate[0]))
    row_coord = np.argmin(np.abs(northing - outlet_coordinate[1]))

    # Create necessary X,Y matrices
    X, Y = np.meshgrid(easting, northing)

    # Call code to get flow accum., flowdir, slope, etc.
    flowacc, flowdir, slope, _ = wflowacc(X, Y, elev, type_flag='single', edges='open')

    # Convert sparse matrix slope to correct dimensions
    # Extract the max array configurations to fit the dimensions (column-wise maximums (1 x nrc) into a 1D array)
    slope_dense = np.array(slope.max(axis=0).toarray()).ravel()
    slope = slope_dense.reshape(elev.shape, order='F') # Reshape using Fortran order ('F') to preserve MATLAB spatial layout
    slope = np.arctan(slope)
    slope[slope == 0] = np.mean(slope)

    # Visual confirmations bypassed automatically for batch execution runs, plots are saved
    if save_plots or display_plots:
        # Figure 1
        fig1, ax = plt.subplots(num=1)
        im = ax.imshow(
            elev,
            extent=[easting[0], easting[-1], northing[-1], northing[0]],
            cmap='terrain'
        )
        # add color bar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig1.colorbar(im, ax=ax, cax=cax)
        cbar.set_label('Elevation (m)', fontsize=12)

        # formatting
        ax.set_xlabel("Easting (m)", fontsize=12)
        ax.set_ylabel("Northing (m)", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True)

        # Save Figure
        fig1.savefig(plots_path / "Fig1_elevation.png", dpi=300,  bbox_inches="tight", transparent=False)
        if not display_plots:
            plt.close(fig1)

        # Figure 2
        fig2, ax = plt.subplots(num=2)
        im = ax.imshow(
            np.log(flowacc), 
            extent=[easting[0], easting[-1], northing[-1], northing[0]], 
            cmap='viridis'
        )
        ax.plot(X[row_coord, col_coord], Y[row_coord, col_coord], 'ro', markersize=6)
        ax.scatter(X[row_coord, col_coord], Y[row_coord, col_coord], s=300, c="None", marker="o", edgecolors="red", linewidths=3, zorder=6)

        # add color bar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig2.colorbar(im, ax=ax, cax=cax)
        cbar.set_label('Log of Flow Accumulation', fontsize=12)

        # formatting
        ax.set_xlabel("Easting (m)", fontsize=12)
        ax.set_ylabel("Northing (m)", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True)

        # Save Figure
        fig2.savefig(plots_path / "Fig2_log_flow_accumulation.png", dpi=300,  bbox_inches="tight", transparent=False)
        if not display_plots:
            plt.close(fig2)

    # index of outlet point
    outlet_index = np.ravel_multi_index((row_coord, col_coord), X.shape, order='F')

    # Create a matrix where the first column is the pixel index being examined and
    # the seven following columns are the up to seven neighbor pixel indices flowing 
    # into it (assuming pits are filled there must be one outflow)
    basin_id_matrix = np.zeros((1, 8), dtype=int)
    basin_id_matrix[0, 0] = outlet_index

    # flag for while loop
    run_flag = 1
    # initialize row counter
    row_num = 0
    row_num2 = 1
    
    # Find pixels flowing into outlet
    # convert to CSC format to stay in Fortran ordering to validate with MATLAB
    flowdir_csc = flowdir.tocsc()
    I = flowdir_csc[:, outlet_index].indices
    
    # Fill id_matrix with these pixels (first row)
    if len(I) > 0:
        basin_id_matrix[0, 1:1+len(I)] = I

    # Determine all points in basin draining to outlet (loop works its way "upstream")
    while run_flag == 1:
        # Find the number pixels flowing into current pixel
        current_row_neighbors = basin_id_matrix[row_num, 1:]
        J = len(current_row_neighbors[current_row_neighbors != 0])
        
        # Write new pixels to first column (below current row number); only
        # identifies those pixels that aren't already part of set
        candidates = current_row_neighbors[current_row_neighbors != 0]
        new_pixels = np.setdiff1d(candidates, basin_id_matrix[:, 0])
        
        if len(new_pixels) > 0:
            J = len(new_pixels)
            # Expand container rows dynamically
            temp_block = np.zeros((J, 8), dtype=int)
            temp_block[:, 0] = new_pixels
            basin_id_matrix = np.vstack([basin_id_matrix, temp_block])
            
            # Loop through the up to seven pixels and grab their neighbors that flow into them
            for j in range(J):
                target_pixel = basin_id_matrix[row_num2 + j, 0]
                I_neighbors = flowdir_csc[:, target_pixel].indices
                if len(I_neighbors) > 0:
                    basin_id_matrix[row_num2 + j, 1:1+len(I_neighbors)] = I_neighbors
            
            # Augment row counters
            row_num += 1
            row_num2 += J
        else:
            if row_num == basin_id_matrix.shape[0] - 1: # row counter is at end of matrix
                run_flag = 0
            else:
                row_num += 1

    # Create basin mask
    mask = np.zeros(X.shape)
    # Create basin mask using 2D coordinates decoded from Fortran indices
    rows, cols = np.unravel_index(basin_id_matrix[:, 0], X.shape, order='F') 
    mask[rows, cols] = 1

    # Create watershed outline points
    # Watershed outline (1-pixel thin inner boundary using binary_erosion to match MATLAB edge())
    # Use a 3x3 square (8-connectivity) instead of default 4-connectivity cross
    mask_bool = mask.astype(bool)
    outline_image = mask_bool & ~binary_erosion(mask_bool)

    # Extract watershed outline coordinates in MATLAB (Fortran) order
    # # Old, 1D index based
    # I_outline = np.where(outline_image.flatten(order='F'))[0]
    # watershed_outline_x = X.ravel(order='F')[I_outline]
    # watershed_outline_y = Y.ravel(order='F')[I_outline]
    # New, 2D index tuples
    stream_rows, stream_cols = np.unravel_index(
        np.where(outline_image.ravel(order='F'))[0], 
        outline_image.shape, 
        order='F'
    )
    watershed_outline_x = X[stream_rows, stream_cols]
    watershed_outline_y = Y[stream_rows, stream_cols]

    # Create points on stream network using threshold
    threshold = percent_basin_area * np.sum(mask == 1)

    # Maintain Fortran ordering for linear indices
    I_stream = np.where((flowacc.ravel(order='F') > threshold) & (mask.ravel(order='F') == 1))[0]
    x_stream = X.ravel(order='F')[I_stream]
    y_stream = Y.ravel(order='F')[I_stream]

    if save_plots or display_plots:
        # Figure 3
        fig3, ax = plt.subplots(num=3)
        plt.clf()
        ax = plt.gca()

        # Set dark blue background for the plot area
        ax.set_facecolor('navy')

        # Reconstruct flowacc2 (zero everywhere except along the stream network)
        flowacc2 = np.zeros_like(flowacc)
        flowacc2.ravel(order='F')[I_stream] = flowacc.ravel(order='F')[I_stream]

        with np.errstate(divide='ignore'):
            log_flowacc2 = np.log(flowacc2)
            log_flowacc2[np.isinf(log_flowacc2) | (log_flowacc2 == 0)] = np.nan

        extent_bounds = [X.min(), X.max(), Y.min(), Y.max()]

        # Plot base layer (non-stream NaN cells will reveal the dark blue background)
        im =  ax.imshow(log_flowacc2, extent=extent_bounds, cmap='viridis', origin='upper')

        # Plot watershed outline and outlet
        ax.plot(watershed_outline_x, watershed_outline_y, 'w.', markersize=2, zorder=5)
        # ax.plot(X[row_coord, col_coord], Y[row_coord, col_coord], 'ro', markersize=6)
        # ax.scatter(X[row_coord, col_coord], Y[row_coord, col_coord], s=300, c="None", marker="o", edgecolors="red", linewidths=3, zorder=6)

        # add color bar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig3.colorbar(im, ax=ax, cax=cax)
        cbar.set_label('Log of Flow Accumulation', fontsize=12)

        # Overlay transparent mask (set non-basin 0s to NaN so background stays dark blue)
        mask_overlay = mask.copy().astype(float)
        mask_overlay[mask_overlay == 0] = np.nan
        ax.imshow(mask_overlay, extent=extent_bounds, cmap='gray', alpha=0.3, origin='upper')

        # Labels and formatting
        # ax.set_aspect('equal', adjustable='box')
        # ax.set_title('Watershed Mask with Stream Network', fontsize=20)
        ax.set_xlabel('Easting (m)', fontsize=12)
        ax.set_ylabel('Northing (m)', fontsize=12)
        ax.tick_params(labelsize=10)
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True)

        # Save Figure
        fig3.savefig(plots_path / "Fig3_watershed_mask_stream_network.png", dpi=300, bbox_inches="tight", transparent=False)
        if not display_plots:
            plt.close(fig3)

        # Figure 4 (3D Plot)

        # 1. Get outlet and grid center coordinates
        x_out, y_out = X[row_coord, col_coord], Y[row_coord, col_coord]
        x_center, y_center = (X.min() + X.max()) / 2.0, (Y.min() + Y.max()) / 2.0

        # 2. Compute azimuth angle pointing from outlet toward grid center
        dx = x_center - x_out
        dy = y_center - y_out
        azim_angle = np.degrees(np.arctan2(dy, dx)) - 180

        # Ensure interactive mode is disabled if running in a GUI backend
        plt.ioff()

        fig4, ax = plt.subplots(num=4, subplot_kw={'projection': '3d'}, figsize=(9, 9))

        # Relative elevation and surface plot
        elev_relative = (elev - elev[row_coord, col_coord]) * mask
        norm = plt.Normalize(elev.min(), elev.max())
        colors = plt.cm.viridis(norm(elev))

        ax.plot_surface(
            X, Y, elev_relative, 
            facecolors=colors, 
            rstride=1, cstride=1,
            edgecolor='white',
            linewidth=0.2, 
            antialiased=True, 
            shade=False
        )

        # Plot 3D stream network using Fortran order
        z_stream = elev.ravel(order='F')[I_stream] - elev[row_coord, col_coord]
        ax.plot(x_stream, y_stream, z_stream, 'w*', markersize=4)

        # Set static camera view (works properly on Axes3D object)
        x_out, y_out = X[row_coord, col_coord], Y[row_coord, col_coord]
        dx = ((X.min() + X.max()) / 2.0) - x_out
        dy = ((Y.min() + Y.max()) / 2.0) - y_out
        azim_angle = np.degrees(np.arctan2(dy, dx)) - 180

        ax.view_init(elev=25, azim=azim_angle)

        # Configure axes matching MATLAB properties
        ax.set_xlabel('Easting (m)', fontsize=12)
        ax.set_ylabel('Northing (m)', fontsize=12)
        ax.set_zlabel('Elevation Above Outlet (m)', fontsize=12)
        ax.grid(False)
        # ax.xaxis.grid(False)
        # ax.yaxis.grid(False)
        # ax.zaxis.grid(True)
        ax.tick_params(labelsize=10)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
        ax.set_xlim([X.min(), X.max()])
        ax.set_ylim([Y.min(), Y.max()])

        # Save Figure
        fig4.savefig(plots_path / "Fig4_3D_watershed_mask_stream_network.png", dpi=600, 
                     bbox_inches="tight", pad_inches=0.5, transparent=False)
        if not display_plots:
            plt.close(fig4)

    # display plots if requested
    if display_plots:
        plt.show()

    return mask, flowacc, flowdir, slope, stream_rows, stream_cols

def process_met_file(df: pd.DataFrame, gage_elev: float, dt: float = 0.25, file_path: str | Path = None,
                     ) -> None:
    """
    Description:
    This code is written to process a meteorological forcing CSV file into the format
    used by MOD-WET simulations.

    Inputs:
    df:
        Pandas DataFrame
        meteorological data, temporal resolution must be daily or finer
        columns MUST include: year, month, day, hour, minute, second, Ta, qa, Psfc, U, SW, PPT
        Data column details and units:
            Ta = air temperature (K)
            qa = specific humidity (kg/kg)
            Psfc = surface air pressure (Pa)
            U = windspeed (m/s)
            SW = incoming shortwave radiation (W/m^2)
            PPT = precipitation rate (mm/h)
    gage_elev:
        float
        elevation of meteorological station (in meters)
    dt:
        float
        time step to output (in hours, should match time step used for
        MOD-WET simulation (default is 0.25 hours)
    file_path
        str or Path
        file path and name of the preprocessed meteorological forcing data for MOD-WET model
    
    Notes: 
    1. Temporal resolution of the meteorological data and the water year are automatically inferred from the dataframe
    2. Temporal resolution of the meteorological data must range from 1 day to 0.25 hours
    3. We use 0-based days indexing instead of 1-based days so that Jan-01 at midnight is equal to 0 days, not 1 day. 
       In others, the unit of time is number of elapsed days, not day of the year.
    """
    method = 'linear' # interpolation method

    # add a datetime column
    df['datetime'] = pd.to_datetime(df[["year", "month", "day", "hour", "minute", "second"]])

    # Set datetime as the index and sort chronologically
    df = df.set_index("datetime").sort_index()

    # Calculate time difference between consecutive timestamps, in hours
    dt_hours = np.array(df.index.to_series().diff().dt.total_seconds() / 3600)[1:]
    dt_num_unique = len(np.unique(dt_hours))
    if dt_num_unique != 1:
        sys.exit("Meteorological data not evenly spaced in time, multiple time deltas are present.")
    dt_orig = dt_hours[0]

    # determine water year
    first_year = df.index[0].year
    first_month = df.index[0].month
    first_day = df.index[0].day
    first_hour = df.index[0].hour
    first_minute = df.index[0].minute
    first_second = df.index[0].second
    if (first_month != 10) or (first_day != 1):
        sys.exit("Meteorological data should be in water year format, beginning on October 1st")

    # determine water year (safely)
    if first_month >= 10:
        water_year = int(first_year + 1)
    else:
        water_year = int(first_year)

    # # determine number of days based on if water year is leap year
    # if (water_year % 4 == 0) and (water_year % 100 != 0 or water_year % 400 == 0):
    #     num_days = 366
    # else:
    #     num_days = 365
    # Update: Remove requirement that forcing data should be exactly 1 year
    #         model number of time steps will be based on length of preprocessed meteorological data record
    # required_length = int((num_days * 24) // dt_orig)
    # if len(df) != required_length:
    #     sys.exit(f"Meteorological data must span exactly one year. Given data {dt_orig} hours apart, in water year {water_year}, there should be {required_length} data points.")

    # Update: extract number of days on record and process data accordingly
    # Requiment: met data should include FULL days, not partial days (i.e., if data is sub-daily, each day's record should be complete)
    record_len = len(df)
    extra_records = int(((record_len * dt_orig) % 24) // dt_orig)
    if extra_records != 0:
        one_day_record = int(24 / dt_orig)
        missing_records = one_day_record - extra_records
        full_record_low = record_len - extra_records
        full_record_low_days = (full_record_low * dt_orig) // (24)
        full_record_high = record_len + missing_records
        full_record_high_days = (full_record_high * dt_orig) // (24)
        given_record = (record_len * dt_orig) / (24)
        sys.exit(f"""Meteorological data record contains {record_len} records with time resolution of {dt_orig}, resulting in {given_record} days which is in a non-integer number of days.
        Add {missing_records} records for a complete set of {full_record_high} records with {full_record_high_days} full days or, 
        Remove {extra_records} records for a complete set of {full_record_low} records with {full_record_low_days} full days.""")
    else:
        num_days = int((record_len * dt_orig) / 24) # placeholder

    # Build interpolation x-values
    t_new = np.arange(0, num_days, dt / 24)
    t_orig = np.arange(0, num_days, dt_orig / 24)

    # perform interpolartion to obtained desired temporal resolution
    Ta = interp1d(t_orig, df['Ta'].values, kind=method, fill_value="extrapolate")(t_new)
    qa = interp1d(t_orig, df['qa'].values, kind=method, fill_value="extrapolate")(t_new)
    Psfc = interp1d(t_orig, df['Psfc'].values, kind=method, fill_value="extrapolate")(t_new)
    U = interp1d(t_orig, df['U'].values, kind=method, fill_value="extrapolate")(t_new)
    SW = interp1d(t_orig, df['SW'].values, kind=method, fill_value="extrapolate")(t_new)
    PPT = interp1d(t_orig, df['PPT'].values, kind='nearest', fill_value="extrapolate")(t_new) # method switched to 'nearest' in MATLAB
    PPT[PPT < 0] = 0 # in case of interp. errors

    # build new datetime vector
    first_date_string = f"{first_year:04d}-{first_month:02d}-{first_day:02d} {first_hour:02d}:{first_minute:02d}:{first_second:02d}"
    start_date = pd.Timestamp(first_date_string)
    datetime_vector = start_date + pd.to_timedelta(t_new, unit="D").round("s")
    doy = datetime_vector.day_of_year + (datetime_vector.hour / 24 + datetime_vector.minute / 1440 + datetime_vector.second / 86400)

    # build to xarray Dataset
    ds = xr.Dataset(
        # The tuple format is: ("dimension_name", numpy_array, {attributes})
        data_vars={
            "Ta":               ("time", Ta,    {"units": "K",           "long_name": "air temperature"}),
            "qa":               ("time", qa,    {"units": "kg/kg",       "long_name": "specific humidity"}),
            "Psfc":             ("time", Psfc,  {"units": "Pa",          "long_name": "surface air pressure"}),
            "U":                ("time", U,     {"units": "m/s",         "long_name": "windspeed"}),
            "SW":               ("time", SW,    {"units": "W/m^2",       "long_name": "incoming shortwave radiation"}),
            "PPT":              ("time", PPT,   {"units": "mm/h",        "long_name": "precipitation rate"}),
            "elapsed_days":     ("time", t_new, {"units": "days",        "long_name": "elapased days since " + first_date_string}),
            "DOY":              ("time", doy,   {"units": "day-of-year", "long_name": f"day of year where Jan 1 00:00:00 = 1.0, Dec 31 00:00:00 = 365.0 if non-leap year, 366.0 if leap year"}),
        },
        # Define the shared 1D dimension coordinate
        coords={
            "time": ("time", datetime_vector),
        },
        # Store the single float parameter (x) as a global attribute
        attrs={
            "gage_elev": gage_elev,
            "start_date_time": first_date_string,
        }
    )

    # specify encoding for datetime
    encoding = {
        "time": {
            "units": "days since " + first_date_string,
            "dtype": "float64",
            "calendar": "gregorian",
        }
    }

    # Save to NetCDF
    ds.to_netcdf(file_path, encoding=encoding)

def MOD_WET_watershed_preprocessing(easting: np.ndarray, northing: np.ndarray, elev: np.ndarray, outlet_coordinate: np.ndarray,
                                    static_data_file: str | Path,
                                    terrain_flag: bool = True, delineation_flag: bool = True, slope_aspect_flag: bool = True, shade_calc_flag: bool = True,
                                    save_plots: Optional[bool] = False, display_plots: Optional[bool] = False, plots_path: Optional[str | Path] = None,
                                    met_data: Optional[pd.DataFrame] = None, gage_elev: Optional[float] = None, met_data_file: Optional[str | Path] = None,
                                    dt_interp: Optional[float] = 0.25
                                    ) -> None:
    """
    Description:
    This code is written to take DEM data as an input and construct the 
    various other terrain inputs and meteorological data needed to run the 
    MOD-WET model.

    Notes: the structure of this function (mainly its inputs) was revised to 
    enforce default settings and simply function calls.
    NOTES: easting and northing should be Nx1 and Mx1 vectors, and
    elev should be a MxN array; outlet coordinates should be in same 
    units of easting and northing and should be on a stream channel

    Inputs:

    DEM Data Inputs:
        easting: 
            1D x array of easting values
        northing: 
            1D y array of northing values
        elev: 
            elevation matrix (at each x,y point, i.e. the DEM)
            terain_flag: 
        outlet_coordinate: 
            1D array containing coordinate of the basin outlet (i.e., [outlet_easting, outlet_northing])
    
    Output Static Data File Options:
        static_data_file:
            file path, including name, of the processed watershed static data for the MOD-WET Model
    
    Data Processing Flags
        terrain_flag: 
            bool, default True    
            flag to identify whether at least one of the terrain processing steps will be performed
        delineation_flag:
            bool, default True
            flag to specify whether delineation is to be performed, requires terrian_flag set to True
        slope_aspect_flag:
            bool, default True
            flag to specify whether slope/aspect calcs. are to be performed, requires terrian_flag set to True
        shade_calc_flag: 
            bool, default True
            flag for specifying whether to run shade calculations, requires terrian_flag set to True

    Watershed Delineation Plotting Options:
        save_plots: 
            boolean, if True, develop and save plots to plot_path
        display_plots: 
            boolean, if True, displays plots to screen (save_plots must be set to True)
        plots_path: 
            location to save watershed delineation plots

    Meterological Data Processing
        met_data:
            Optional
            pandas DataFrame
            meteorological data, temporal resolution must be daily or finer
            columns MUST include: year, month, day, hour, minute, second, Ta, qa, Psfc, U, SW, PPT
            Data column details and units:
                Ta = air temperature (K)
                qa = specific humidity (kg/kg)
                Psfc = surface air pressure (Pa)
                U = windspeed (m/s)
                SW = incoming shortwave radiation (W/m^2)
                PPT = precipitation rate (mm/h)
        gage_elev:
            elevation of meteorological station (in meters)
        met_data_file: 
            file path, including name, of the processed meteorological data for the MOD-WET Model
        dt_interp: 
            float
            time step to interpolated to, in hours
            should match MOD-WET time resolution
            default is 0.25

    Outputs:
    
    2 NetCDF files are saved: one for static watershed data and one for meteorological data
    If met_data is not provided, only the static watershed data is saved.
        
    The static data file contains the following:
        variables:
            2D map data (easting, northing):
                aspect  - matrix of aspect direction for each pixel in degrees
                elev    - matrix of elevation for each pixel
                flowacc - matrix of flow accumulation for each pixel
                mask    - binary matrix indicating the delineated basin
                slope   - matrix of slope for each pixel in degrees
                SVF     - matrix of sky view factor for each pixel
            
            Flow direction sparse matrix (nnz X 1 vectors):
                flowdir_data - non-zero values of the flow direction sparse matrix
                flowdir_row  - row indices of the non-zero values
                flowdir_col  - column indices of the non-zero values
            
            Shade lookup table (northing, easting, zenith, azimuth):
                shade_lookup_table - 4D array indicating shade as a function of discrete zenith and azimuth angles
        
        coordinates:
            northing - 1D array of northing coordinates
            easting  - 1D array of easting coordinates
            zenith   - 1D array of discrete zenith angles for shade lookup table
            azimuth  - 1D array of discrete azimuth angles for shade lookup table
        
        attributes:
            outlet_coordinate - 1x2 array with the x and y coordinate of the basin outlet
            shade_calc_flag   - flag for specifying whether to run shade calculations
            dx                - DEM resolution in x direction (m)
            dy                - DEM resolution in y direction (m)
            stream_rows       - x-coordinate index for the watershed outline points
            stream_cols       - y-coordinate index for the watershed outline points
    
    The meteorological data file contains the following:
        variables:
            Ta    - 1D array of air temperature (K)
            qa    - 1D array of specific humidity (kg/kg)
            Psfc  - 1D array of surface air pressure (Pa)
            U     - 1D array of windspeed (m/s)
            SW    - 1D array of incoming shortwave radiation (W/m^2)
            PPT   - 1D array of precipitation rate (mm/h)
        coordinates:
            time  - 1D array of time in days
        attributes:
            gage_elev - elevation of the gage coordinate (m)
    """

    # input inspection
    inspect_DEM_data(easting, northing, elev, outlet_coordinate)
    inspect_plot_option(save_plots, plots_path, display_plots)
    inspect_paths(static_data_file, met_data_file)
    inspect_met(met_data, dt_interp, gage_elev, met_data_file)

    print('Starting MOD-WET model watershed input data pre-processing...')

    # Terrain processing
    if terrain_flag:
        # Define DEM resolution
        dx = abs(easting[1] - easting[0])
        dy = abs(northing[1] - northing[0])
        
        # Call watershed delineation function
        if delineation_flag:
            print("Performing stream delineation...")
            percent_basin_area = 0.01
            mask, flowacc, flowdir, slope_delin, stream_rows, stream_cols = watershed_area_and_stream_delineation(
                easting, northing, elev, outlet_coordinate, percent_basin_area, save_plots, display_plots, plots_path
            )
            # Convert flowdir to COO format
            flowdir = flowdir.tocoo()
            print("Done.")
        else: # note: this should not be called
            mask, flowacc, flowdir, slope_delin = None, None, None, None

        # Call slope/aspect function
        print("Building slope and aspect maps...")
        if slope_aspect_flag:
            slope, aspect = generate_slope_and_aspect_from_DEM(elev, easting, northing)
            print("Done.")
        else: # note: this should not be called
            slope, aspect = None, None

        # Call shade lookup table and SVF function
        if shade_calc_flag:
            print("Building shade lookup table...")
            print("Creating shade lookup table and calculating sky view factor (SVF) map ...")
            shade_lookup_table, SVF, discrete_zenith_values, discrete_azimuth_values = compute_shade_lookup_table_and_SVF(
                easting, northing, elev, slope, aspect
            )
            print('Done.')
        else: # note: this can be called
            shade_lookup_table, SVF, discrete_zenith_values, discrete_azimuth_values = None, None, None, None

        # Save data as NetCDF
        print("Saving static watershed data...")
        # Build the xarray Dataset
        if shade_calc_flag:
            static_data_vars={
                # 2D map data
                "aspect":   (("northing", "easting"), aspect,   {"units": "degrees",    "long_name": "Terrain aspect orientation grid"}),
                "elev":     (("northing", "easting"), elev,     {"units": "m",          "long_name": "Terrain elevation grid"}),
                "flowacc":  (("northing", "easting"), flowacc,  {"units": "m^2",        "long_name": "Accumulated upstream flow area grid"}),
                "mask":     (("northing", "easting"), mask,     {"units": "-",          "long_name": "Watershed binary mask array (0/1)"}),
                "slope":    (("northing", "easting"), slope,    {"units": "degrees",    "long_name": "Terrain slope grid"}),
                "SVF":      (("northing", "easting"), SVF,      {"units": "-",          "long_name": "Sky view factor grid"}),
                
                # flowdir sparse matrix
                "flowdir_data": (("nnz",), flowdir.data,        {"units": "-",          "long_name": "Sparse matrix indicating flow directions (0/1)"}),
                "flowdir_row":  (("nnz",), flowdir.row,         {"units": "-",          "long_name": "Sparse matrix non-zero row indices"}),
                "flowdir_col":  (("nnz",), flowdir.col,         {"units": "-",          "long_name": "Sparse matrix non-zero column indices"}),

                # Shade lookup table
                "shade_lookup_table": (("northing", "easting", "zenith", "azimuth"), shade_lookup_table),
            }
            static_coords={
                "northing": ("northing", northing, {"units": "m"}),
                "easting":  ("easting", easting, {"units": "m"}),
                "zenith":   ("zenith", discrete_zenith_values, {"units": "degrees"}),
                "azimuth":  ("azimuth", discrete_azimuth_values, {"units": "degrees"}),
            }
        else:
            static_data_vars={
                # 2D map data
                "aspect":   (("northing", "easting"), aspect,   {"units": "degrees",    "long_name": "Terrain aspect orientation grid"}),
                "elev":     (("northing", "easting"), elev,     {"units": "m",          "long_name": "Terrain elevation grid"}),
                "flowacc":  (("northing", "easting"), flowacc,  {"units": "m^2",        "long_name": "Accumulated upstream flow area grid"}),
                "mask":     (("northing", "easting"), mask,     {"units": "-",          "long_name": "Watershed binary mask array (0/1)"}),
                "slope":    (("northing", "easting"), slope,    {"units": "degrees",    "long_name": "Terrain slope grid"}),
                
                # flowdir sparse matrix
                "flowdir_data": (("nnz",), flowdir.data,        {"units": "-",          "long_name": "Sparse matrix indicating flow directions (0/1)"}),
                "flowdir_row":  (("nnz",), flowdir.row,         {"units": "-",          "long_name": "Sparse matrix non-zero row indices"}),
                "flowdir_col":  (("nnz",), flowdir.col,         {"units": "-",          "long_name": "Sparse matrix non-zero column indices"}),
            }
            static_coords={
                "northing": ("northing", northing, {"units": "m"}),
                "easting": ("easting", easting, {"units": "m"}),
            }

        static_ds = xr.Dataset(
            
            # Define the variables
            data_vars=static_data_vars,
            
            # Define shared coordinates
            coords=static_coords,
            
            # Scalar and Coordinate Attributes
            attrs={
                "outlet_coordinate": outlet_coordinate,
                "shade_calc_flag": int(1 if shade_calc_flag else 0),
                "dx": dx,
                "dy": dy,
                "flowdir_shape": 2*[len(northing) * len(easting)],
                "stream_rows": stream_rows,
                "stream_cols": stream_cols
            }
        )

        # Save directly to a NetCDF4 file
        static_ds.to_netcdf(static_data_file)
        print('Done.')
        print(f"Basin static data saved here: {static_data_file}")
    
    # Call meteorological processing function
    if met_data is not None:
        print("Processing meteorological data...")
        process_met_file(df=met_data, gage_elev=gage_elev, dt=dt_interp, file_path=met_data_file)
        print('Done.')
        print(f"Meteorological forcing data saved here: {met_data_file}")

def inspect_DEM_data(easting, northing, elev, oc):

    # check if elev matrix and northing and easting vectors align
    # Note: this doesn't work as intended if the number of northing and easting points are the same
    northing_dim, easting_dim = elev.shape
    if (northing_dim != northing.shape[0]) or (easting_dim != easting.shape[0]):
        sys.exit("Error: Northing and/or easting points are not aligned with elevation data.")
    
    # check if outlet coordinate lies inside the grid
    north_min = np.min(northing)
    north_max = np.max(northing)
    east_min = np.min(easting)
    east_max = np.max(easting)
    if (oc[0] < east_min) or (oc[0] > east_max) or (oc[1] < north_min) or (oc[1] > north_max):
        sys.exit("Error: Outlet coordinate not contained within DEM grid. Coordinate should be (easting, northing).")

def inspect_plot_option(save_plots, plots_path, display_plots):
    if save_plots:
        if plots_path is not None:
            if isinstance(plots_path, (str, Path)):
                plots_path = Path(plots_path)
            plots_path.mkdir(parents=True, exist_ok=True)
        else:
            sys.exit("Invalid or missing plots_path")
    # else:
    #     if display_plots:
    #         sys.exit("To display plots, save_plots must be set to True.")

def inspect_paths(static_data_file, met_data_file):
    # Static data output inspection
    if static_data_file is not None:
        if isinstance(static_data_file, (str, Path)):
            static_data_file_parent = Path(static_data_file).parent
            static_data_file_parent.mkdir(parents=True, exist_ok=True)
            if isinstance(static_data_file, (str)):
                if not static_data_file.endswith('.nc'):
                    sys.exit("static_data_file must contain file name that ends in '.nc'")
            elif isinstance(static_data_file, (Path)):
                if static_data_file.suffix != '.nc':
                    sys.exit("static_data_file must contain file name that ends in '.nc'")
        else:
            sys.exit("Invalid or missing static_data_file")

    if met_data_file is not None:
        if isinstance(met_data_file, (str, Path)):
            met_data_file_parent = Path(met_data_file).parent
            met_data_file_parent.mkdir(parents=True, exist_ok=True)
            if isinstance(met_data_file, (str)):
                if not met_data_file.endswith('.nc'):
                    sys.exit("met_data_file must contain file name that ends in '.nc'")
            elif isinstance(met_data_file, (Path)):
                if met_data_file.suffix != '.nc':
                    sys.exit("met_data_file must contain file name that ends in '.nc'")
        else:
            sys.exit("Invalid or missing met_data_file")

def inspect_met(met_data, dt_interp, gage_elev, met_data_file):
    if met_data is not None:
        if not isinstance(met_data, pd.DataFrame):
            sys.exit("met_data must be a Pandas Dataframe")
        else:
            required_columns = ['year', 'month', 'day', 'hour', 
                                'minute', 'second', 
                                'Ta', 'qa', 'Psfc', 'U', 'SW', 'PPT']
            if not set(required_columns).issubset(met_data.columns):
                sys.exit("met_data must contain: year, month, day, hour, minute, second, Ta, qa, Psfc, U, SW, PPT")
            else:
                if dt_interp is None:
                    warnings.warn("dt_interp not specified.")
                if any(v is None for v in (gage_elev, met_data_file)):
                    sys.exit("gage_elevation, and met_data_file must be specified if met_data is provided")

