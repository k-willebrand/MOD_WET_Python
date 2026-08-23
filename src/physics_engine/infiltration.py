import numpy as np

from src.chapter7 import TCA_infiltration

def step_infiltration(model, masksnow: np.ndarray, maskSEB: np.ndarray, PPT0: np.ndarray, step_idx: int, ts_idx: int):
    """
    Executes the surface soil infiltration model (TCA Philip), updates
    infiltration-excess runoff, computes ET/sublimation rates, and updates
    time-series tracking and cumulative flux accumulators.

    Inputs:
    model : Model
        Master container holding state, step_vars, accumulators, map_outputs, and control.
    masksnow : np.ndarray
        2D boolean mask of active snow-covered or snowing grid cells.
    maskSEB : np.ndarray
        2D boolean mask of active bare soil / snow-free grid cells within the watershed.
    PPT0: np.ndarray
        distributed precipitation field (actually constant)
    step_idx : int
        Current 0-indexed timestep iteration.
    ts_idx : int
        Current 0-indexed array index for TimeSeries (nt dimension).

    Outputs:
    f: np.ndarray
        infiltration rate
    ETsoil: np.ndarray
        soil ET rate
    """
    # Call infiltration model (i.e. TCA with Philip solution)
    # If snow exists, but no melt --> no infiltration
    # If snow exists with melt--> melt flux input to soil infiltration
    # model
    # If no snow, pass precip. input to infiltration mode
    PPT0_m = np.copy(PPT0) / 1000.0  # PPT mm/hr --> m/hr
    # Note: In MATLAB, the PPT0 gets over-written when units 
    # are converted from mm/hr to m/hr, here a copy is created 
    # instead to preserve units (not that it affects anything downstream)
    # Because of this decicion however, the mass balance calculation 
    # needs to be adjusted, and PPT0 must be converted from mm/hr to m/hr
    # in the mass balance check function as well

    # Pre-allocate infiltration rate array
    f = np.zeros((model.control.nx, model.control.ny), dtype=float)

    # For snowy pixels, set to melt rate
    if np.any(masksnow):
        f[masksnow] = model.step_vars.melt_out[masksnow]  # (m/hr)

    # For bare pixels, set to precipitation rate
    if np.any(maskSEB):
        f[maskSEB] = PPT0_m[maskSEB]  # (m/hr)

    # Determine infiltration excess runoff [m]
    # initialize
    model.step_vars.qie_new = np.zeros((model.control.nx, model.control.ny), dtype=float) * model.spatial.maskNaN

    # Compute rootzone soil moisture [-]
    theta_rz = model.state.Srz / model.params.d_rz

    Ipond = f >= model.spatial.K0  # pixels that could potentially experience ponding
    if np.any(Ipond):  # only call TCA if surface flux greater than Ksat (K0)
        method_flag = 1  # Philip solution
        # Note: This is called every time step with the storm "duration"
        # set to the time step duration. This is a simplification in order
        # to make the function compatible with inline calculation where the
        # real total storm duration is not known ahead of time. This may cause
        # issues at small time steps.
        F_out, qie_out = TCA_infiltration(
            P=f[Ipond],
            tr=model.control.dt,
            t=model.control.dt,
            theta_0=theta_rz[Ipond],
            porosity=model.spatial.THETAs[Ipond],
            Ks=model.spatial.K0[Ipond],
            psi_s=model.spatial.PSIs[Ipond],
            b=model.spatial.b_BC[Ipond],
            method_flag=method_flag,
        )
        # infiltration rate (m/hr)
        f[Ipond] = F_out / model.control.dt  # (m/hr)
        model.step_vars.qie_new[Ipond] = qie_out  # (m)

    # Initialize ET
    model.step_vars.ET_out = np.full((model.control.nx, model.control.ny), np.nan, dtype=np.float64)

    # Determine evaporative/sublimation flux at each pixel
    if np.any(masksnow):
        model.step_vars.ET_out[masksnow] = (
            model.step_vars.LE_out[masksnow] / model.constants.Ls / model.constants.rhow * 3600.0
        )  # (m/hr)
    if np.any(maskSEB):
        model.step_vars.ET_out[maskSEB] = (
            model.step_vars.LE_out[maskSEB] / model.constants.Lv / model.constants.rhow * 3600.0
        )  # (m/hr)
    
    # Only send ET for non-snowy pixels to TOPMODEL (i.e. set ET for snowy
    # pixels to zero in this case).
    ETsoil = np.copy(model.step_vars.ET_out)  # (m/hr)
    if np.any(masksnow):
        ETsoil[masksnow] = 0.0

    # Store melt, ET, infiltration flux results
    if (step_idx + 1) % model.control.timeseries_frq2store == 0:
        model.time_series.snowmelt[ts_idx] = float(np.nanmean(model.step_vars.melt_out))  # (m/hr)
        model.time_series.ET[ts_idx] = float(np.nanmean(model.step_vars.ET_out))  # (m/hr)
        model.time_series.infil[ts_idx] = float(np.nanmean(f))  # (m/hr)

        # Special pixels
        if model.params.special_pixels and len(model.params.special_pixels[0]) > 0:
            sp_rows, sp_cols = model.params.special_pixels
            model.time_series.pixel_snowmelt[ts_idx][sp_rows, sp_cols] = model.step_vars.melt_out[sp_rows, sp_cols]  # (m/hr)
            model.time_series.pixel_ET[ts_idx][sp_rows, sp_cols] = model.step_vars.ET_out[sp_rows, sp_cols]  # (m/hr)
            model.time_series.pixel_infil[ts_idx][sp_rows, sp_cols] = f[sp_rows, sp_cols]  # (m/hr)

    # Accumulate fluxes for time averaging
    # save as meters
    model.accumulators.cumulsnowmelt += model.step_vars.melt_out * model.control.dt  # (m)
    model.accumulators.cumulET += model.step_vars.ET_out * model.control.dt  # (m)
    model.accumulators.cumulinfil += f * model.control.dt  # (m)

    return f, ETsoil