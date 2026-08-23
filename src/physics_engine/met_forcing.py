import numpy as np

from src.chapter6 import distribute_met_forcing

def step_met_forcing_distribution(model, step_idx: int, ts_idx: int):
    """
    Processes temporal indices, calls forcing distribution, and updates accumulators.
    
    Parameters
    ----------
    model : Model
        Master container holding state, step_vars, accumulators, map_outputs, and control.
    step_idx : int
        Current 0-indexed timestep iteration.
    ts_idx : int
        Current 0-indexed array index for TimeSeries (nt dimension).

    Returns
    -------
    PPT0: array
        distributed precipitation field (actually constant)
    U0: np.ndarray
        distributed wind field (actually constant)
    Ta0: np.ndarray
        distributed temperature field (via elevation/lapse rate)
    Psfc0: np.ndarray
        distributed surface pressure
    qa0: np.ndarray
        distributed specific humidity
    SW0: np.ndarray
        distributed solar radiation (as a function of slope/aspect/elev.)
    LWdown0: np.ndarray
        distributed incoming longwave radiation
    """
    dt_hours = model.control.dt
    # forcing_idx = model.control.start_time + step_idx

    # Update time since last snowfall
    if step_idx > 0:
        model.state.NDayLastSnow += dt_hours / 24.0

    # Extract Day of Year (DOY) and UTC time
    current_time = model.forcing.DOY[step_idx]
    doy = int(np.floor(current_time))
    utc = float((current_time - doy) * 24.0)

    # Initial upward longwave estimate from ambient surface
    model.step_vars.Rlup_out = (
        model.spatial.emiss * model.constants.SB_const * (model.forcing.Ta[0] ** 4)
    )

    # Distribute meteorological forcings
    PPT0, U0, Ta0, Psfc0, qa0, SW0, LWdown0 = distribute_met_forcing(
        PPT=model.forcing.PPT[step_idx],
        SW=model.forcing.SW[step_idx],
        Ta=model.forcing.Ta[step_idx],
        qa=model.forcing.qa[step_idx],
        U=model.forcing.U[step_idx],
        Psfc=model.forcing.Psfc[step_idx],
        maskNaN=model.spatial.maskNaN,
        elev=model.spatial.elev,
        gage_elev=model.forcing.gage_elev,
        DOY=doy,
        UTC=utc,
        time_zone_shift=model.params.time_zone_shift,
        lat_mean=model.params.lat_mean,
        lon_mean=model.params.lon_mean,
        slope_rad=model.spatial.slope_rad,
        aspect_rad=model.spatial.aspect_rad,
        SVF=model.spatial.SVF,
        mask=model.spatial.mask,
        LapseRateTair=model.params.LapseRateTair,
        LapseRateTdew=model.params.LapseRateTdew,
        LapseRatePPT=model.params.LapseRatePPT,
        albedo=model.step_vars.albedo_out,
        shade_calc_flag=model.shade.shade_calc_flag,
        discrete_azimuth_values=model.shade.azimuth,
        discrete_zenith_values=model.shade.zenith,
        shade_lookup_table=model.shade.shade_table,
        clear_sky_atmos_emiss_model=model.params.clear_sky_atmos_emiss_model,
        cloudy_sky_atmos_emiss_model=model.params.cloudy_sky_atmos_emiss_model,
        solar_index=model.forcing.solar_index[step_idx],
        LW_up=model.step_vars.Rlup_out,
        g=model.constants.g,
        Rd=model.constants.Rd,
        T_0=model.constants.T_0,
        e_s0=model.constants.e_s0,
        Lv=model.constants.Lv,
        Rv=model.constants.Rv,
        epsilon=model.constants.epsilon,
        S0=model.constants.S0,
        SB_const=model.constants.SB_const
    )

    # Store 1D basin-wide averages
    if (step_idx + 1) % model.control.timeseries_frq2store == 0:
        model.time_series.Rs[ts_idx] = float(np.nanmean(SW0))
        model.time_series.Tair[ts_idx] = float(np.nanmean(Ta0))
        model.time_series.Rldown[ts_idx] = float(np.nanmean(LWdown0))
        model.time_series.qair[ts_idx] = float(np.nanmean(qa0))
        model.time_series.Psfc[ts_idx] = float(np.nanmean(Psfc0))
        model.time_series.PPT[ts_idx] = float(np.nanmean(PPT0) / 1000.0)  # mm/hr -> m/hr

    # Accumulate spatial maps for daily metrics
    model.accumulators.cumulRs += SW0
    model.accumulators.cumulTair += Ta0
    model.accumulators.cumulRldown += LWdown0
    model.accumulators.cumulqair += qa0
    model.accumulators.cumulPsfc += Psfc0
    model.accumulators.cumulPPT += PPT0 * dt_hours / 1000.0

    return PPT0, U0, Ta0, Psfc0, qa0, SW0, LWdown0