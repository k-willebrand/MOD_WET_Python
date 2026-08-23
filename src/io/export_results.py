from pathlib import Path
import numpy as np
import xarray as xr

def save_simulation_results_netcdf(model, output_path: str | Path) -> None:
    """
    Exports simulation results (spatial map outputs, basin time-series, and pixel time-series)
    to a NetCDF4 file using xarray, matching the precision reduction and metadata standard.
    """
    output_path = Path(output_path)

    # 1. Construct Time Coordinates (days)
    # end_day = model.control.start_day + model.control.n_days
    # time_states_val = np.linspace(model.control.start_day, end_day, model.control.nt + 1, dtype=np.float32)
    # time_fluxes_val = np.linspace(model.control.start_day, end_day, model.control.nt, dtype=np.float32)
    ts_dt_hours = model.control.dt * model.control.timeseries_frq2store
    time_states_val = (model.control.start_day + np.arange(model.control.nt + 1) * ts_dt_hours / 24.0).astype(np.float64)
    time_fluxes_val = (model.control.start_day + np.arange(model.control.nt) * ts_dt_hours / 24.0).astype(np.float64)
    # time_maps_val = np.linspace(model.control.start_day, end_day, model.map_outputs.n_map, dtype=np.float32)

    # Convert time coordinates to time stamps
    first_date_string = model.forcing.start_date_time
    first_date = np.datetime64(first_date_string)
    time_states = first_date + (time_states_val * 86400).astype("timedelta64[s]")
    time_fluxes = first_date + (time_fluxes_val * 86400).astype("timedelta64[s]")
    # time_maps = first_date + (time_maps_val * 86400).astype("timedelta64[s]") 
    time_maps = first_date + (np.arange(0,model.map_outputs.n_map)).astype("timedelta64[D]")

    # 2. Extract Spatial Coordinates
    northing = (
        model.spatial.northing.astype(np.float32)
        if model.spatial.northing is not None
        else np.arange(model.control.nx, dtype=np.float32)
    )
    easting = (
        model.spatial.easting.astype(np.float32)
        if model.spatial.easting is not None
        else np.arange(model.control.ny, dtype=np.float32)
    )
    latitude = (
        model.spatial.lat.astype(np.float32)
        if model.spatial.lat is not None
        else np.arange(model.control.ny, dtype=np.float32)
    )
    longitude = (
        model.spatial.lon.astype(np.float32)
        if model.spatial.lon is not None
        else np.arange(model.control.nx, dtype=np.float32)
    )

    n_special = model.time_series.n_special_pixels
    n_stream = model.time_series.n_stream_pixels

    data_vars = {}

    # 3. Process 3D Map Outputs (time_maps, northing, easting)
    map_meta = {
        "Srz": ("Rootzone soil moisture storage", "m"),
        "Suz": ("Unsaturated zone storage", "m"),
        "SD": ("Saturation deficit", "m"),
        "Tsurf": ("Surface temperature", "K"),
        "SWE": ("Snow water equivalent", "m"),
        "snowdepth": ("Snow depth", "m"),
        "snowdens": ("Snow density", "kg/m^3"),
        "snowfrac": ("Snow cover fraction", "-"),
        "Td": ("Deep soil temperature", "K"),
        "snowmelt": ("Period snowmelt depth", "m"),
        "Rn": ("Net radiation", "W/m^2"),
        "LE": ("Latent heat flux", "W/m^2"),
        "ET": ("Evapotranspiration rate", "m/h"),
        "H": ("Sensible heat flux", "W/m^2"),
        "qie": ("Infiltration excess runoff", "m"),
        "qse": ("Saturation excess runoff", "m"),
        "qb": ("Baseflow", "m"),
        "qv": ("Recharge to saturated zone", "m"),
        "infil": ("Infiltration depth", "m"),
        "Rlup": ("Upwelling longwave radiation", "W/m^2"),
        "Tair": ("Air temperature", "K"),
        "albedo": ("Surface albedo", "-"),
        "Rldown": ("Downwelling longwave radiation", "W/m^2"),
        "Rs": ("Solar radiation", "W/m^2"),
        "qair": ("Specific humidity", "kg/kg"),
        "Psfc": ("Surface atmospheric pressure", "Pa"),
        "PPT": ("Precipitation depth", "m"),
    }

    for var_name, (long_name, units) in map_meta.items():
        arr = getattr(model.map_outputs, var_name)
        if arr is not None:
            data_vars[f"map_{var_name}"] = (
                ("time_maps", "northing", "easting"),
                arr.astype(np.float32),
                {"long_name": long_name, "units": units},
            )

    # Save NDayLastSnow map if present
    if hasattr(model.map_outputs, "NDayLastSnow"):
        arr = getattr(model.map_outputs, "NDayLastSnow")
        if arr is not None:
            data_vars["map_NDayLastSnow"] = (
                ("time_maps", "northing", "easting"),
                arr.astype(np.float32),
                {"long_name": "Days since last major snowfall", "units": "days"},
            )

    # 4. Process 1D Basin-Averaged States (time_states)
    basin_state_meta = {
        "Srz": ("Basin-average rootzone soil moisture", "m"),
        "Suz": ("Basin-average unsaturated storage", "m"),
        "SD": ("Basin-average saturation deficit", "m"),
        "Tsurf": ("Basin-average surface temperature", "K"),
        "SWE": ("Basin-average snow water equivalent", "m"),
        "snowdepth": ("Basin-average snow depth", "m"),
        "snowdens": ("Basin-average snow density", "kg/m^3"),
        "snowfrac": ("Basin-average snow cover fraction", "-"),
        "Td": ("Basin-average deep soil temperature", "K"),
    }

    for var_name, (long_name, units) in basin_state_meta.items():
        arr = getattr(model.time_series, var_name)
        if arr is not None:
            data_vars[f"basin_{var_name}"] = (
                ("time_states",),
                arr.astype(np.float32),
                {"long_name": long_name, "units": units},
            )

    # 5. Process 1D Basin-Averaged Fluxes & Forcings (time_fluxes)
    basin_flux_meta = {
        "snowmelt": ("Basin-average snowmelt rate", "m/h"),
        "Rn": ("Basin-average net radiation", "W/m^2"),
        "LE": ("Basin-average latent heat flux", "W/m^2"),
        "ET": ("Basin-average evapotranspiration rate", "m/h"),
        "H": ("Basin-average sensible heat flux", "W/m^2"),
        "qie": ("Basin-average infiltration excess runoff rate", "m/h"),
        "qse": ("Basin-average saturation excess runoff rate", "m/h"),
        "qb": ("Basin-average baseflow rate", "m/h"),
        "qv": ("Basin-average recharge rate to saturated zone", "m/h"),
        "outlet_hydrograph": ("Basin outlet hydrograph", "m^3/s"),
        "Rlup": ("Basin-average upwelling longwave radiation", "W/m^2"),
        "infil": ("Basin-average infiltration rate", "m/h"),
        "Rs": ("Basin-average shortwave radiation", "W/m^2"),
        "Tair": ("Basin-average air temperature", "K"),
        "albedo": ("Basin-average surface albedo", "-"),
        "qair": ("Basin-average specific humidity", "kg/kg"),
        "Psfc": ("Basin-average surface pressure", "Pa"),
        "Rldown": ("Basin-average downwelling longwave radiation", "W/m^2"),
        "PPT": ("Basin-average precipitation rate", "m/h"),
    }

    for var_name, (long_name, units) in basin_flux_meta.items():
        arr = getattr(model.time_series, var_name)
        if arr is not None:
            data_vars[f"basin_{var_name}"] = (
                ("time_fluxes",),
                arr.astype(np.float32),
                {"long_name": long_name, "units": units},
            )

    # 6. Process Static Spatial Maps (2D: northing x easting)
    static_map_meta = {
        "aspect_deg": ("Terrain aspect orientation", "degrees"),
        "aspect_rad": ("Terrain aspect orientation", "radians"),
        "elev": ("Terrain elevation", "m"),
        "flowacc": ("Accumulated upstream flow area", "m^2"),
        "mask": ("Watershed binary mask array (0/1)", "-"),
        "slope_deg": ("Terrain slope", "degrees"),
        "slope_rad": ("Terrain slope", "radians"),
        "SVF": ("Sky view factor", "-"),
        "THETAs": ("Spatially distributed soil porosity", "-"),
        "PSIs": ("Spatially distributed saturated matric head", "m"),
        "b_BC": ("Spatially distributed Brooks-Corey exponent", "-"),
        "THETAfc": ("Volumetric soil moisture at field capacity", "-"),
        "THETApwp": ("Volumetric soil moisture at permanent wilting point", "-"),
        "albedo": ("Spatially distributed surface soil albedo", "-"),
        "emiss": ("Spatially distributed surface soil emissivity", "-"),
        "snow_emiss": ("Spatially distributed snow surface emissivity", "-"),
        "Srzmax": ("Maximum root zone storage", "m"),
        "Srzmin": ("Minimum root zone storage", "m"),
        "T0": ("Surface transmissivity", "m^2/h"),
        "K0": ("Saturated hydraulic conductivity", "m/h"),
        "lambda_map": ("Soil-topographic index", "ln(m^2/m^2/h)"),
        "manning_n": ("Channel Manning roughness coefficient", "-"),
        "width": ("Channel width", "m"),
        "bed_slope": ("Channel bed-slope", "-"),
    }

    # process static maps from model.spatial and add to data_vars
    for var_name, (long_name, units) in static_map_meta.items():
        if var_name not in ["manning_n", "width", "bed_slope"]:  # these are in model.network
            arr = getattr(model.spatial, var_name)
            if arr is not None:
                data_vars[f"static_map_{var_name}"] = (
                    ("northing", "easting"),
                    arr.astype(np.float32),
                    {"long_name": long_name, "units": units},
                )
    
    # process static maps from model.network and add to data_vars
    for var_name, (long_name, units) in static_map_meta.items():
        if var_name in ["manning_n", "width", "bed_slope"]:
            arr = getattr(model.network, var_name)
            if arr is not None:
                data_vars[f"static_map_{var_name}"] = (
                    ("northing", "easting"),
                    arr.astype(np.float32),
                    {"long_name": long_name, "units": units},
                )

    # 7. Process Network Data (flowdir, Iupstream, Idownstream, Ioutlet)
    # process flowdir sparse matrix and save as three separate arrays (data, row, col) along with shape
    flowdir = model.network.flowdir
    if flowdir is not None:
        flowdir_coo = flowdir.tocoo()
        data_vars["flowdir_data"] = (
            ("nnz",),
            flowdir_coo.data,
            {"units": "-", "long_name": "Sparse matrix indicating flow directions (0/1)"},
        )
        data_vars["flowdir_row"] = (
            ("nnz",),
            flowdir_coo.row,
            {"units": "-", "long_name": "Sparse matrix non-zero row indices in Fortran order (row-major)"},
        )
        data_vars["flowdir_col"] = (
            ("nnz",),
            flowdir_coo.col,
            {"units": "-", "long_name": "Sparse matrix non-zero column indices in Fortran order (row-major)"},
        )

    # process Iupstream, Idownstream, Ioutlet arrays and save as 1D arrays
    # process separately as each array may have different lengths
    data_vars["Iupstream_row"] = (
        ("n_upstream",),
        model.network.Iupstream[0].astype(np.int32),
        {"units": "-", "long_name": "Row indices of upstream pixels"},
    )
    data_vars["Iupstream_col"] = (
        ("n_upstream",),
        model.network.Iupstream[1].astype(np.int32),
        {"units": "-", "long_name": "Column indices of upstream pixels"},
    )
    data_vars["Idownstream_row"] = (
        ("n_downstream",),
        model.network.Idownstream[0].astype(np.int32),
        {"units": "-", "long_name": "Row indices of downstream pixels"},
    )
    data_vars["Idownstream_col"] = (
        ("n_downstream",),
        model.network.Idownstream[1].astype(np.int32),
        {"units": "-", "long_name": "Column indices of downstream pixels"},
    )

    # 6. Process Special Pixel Outputs (3D: time x northing x easting)
    if model.time_series.n_special_pixels > 0:
        pixel_state_meta = {
            "pixel_Srz": ("Special pixel rootzone moisture", "m"),
            "pixel_Suz": ("Special pixel unsaturated storage", "m"),
            "pixel_SD": ("Special pixel saturation deficit", "m"),
            "pixel_Tsurf": ("Special pixel surface temperature", "K"),
            "pixel_SWE": ("Special pixel snow water equivalent", "m"),
            "pixel_snowdepth": ("Special pixel snow depth", "m"),
            "pixel_snowdens": ("Special pixel snow density", "kg/m^3"),
            "pixel_snowfrac": ("Special pixel snow cover fraction", "-"),
            "pixel_Td": ("Special pixel deep soil temperature", "K"),
        }
        for var_name, (long_name, units) in pixel_state_meta.items():
            arr = getattr(model.time_series, var_name)
            if arr is not None:
                data_vars[var_name] = (
                    ("time_states", "northing", "easting"),
                    arr.astype(np.float32),
                    {"long_name": long_name, "units": units},
                )

        pixel_flux_meta = {
            "pixel_snowmelt": ("Special pixel snowmelt rate", "m/h"),
            "pixel_Rn": ("Special pixel net radiation", "W/m^2"),
            "pixel_LE": ("Special pixel latent heat flux", "W/m^2"),
            "pixel_ET": ("Special pixel evapotranspiration rate", "m/h"),
            "pixel_H": ("Special pixel sensible heat flux", "W/m^2"),
            "pixel_qie": (
                "Special pixel infiltration excess runoff rate",
                "m/h",
            ),
            "pixel_qse": ("Special pixel saturation excess runoff rate", "m/h"),
            "pixel_qb": ("Special pixel baseflow rate", "m/h"),
            "pixel_qv": ("Special pixel recharge rate", "m/h"),
            "pixel_Rlup": (
                "Special pixel upwelling longwave radiation",
                "W/m^2",
            ),
            "pixel_infil": ("Special pixel infiltration rate", "m/h"),
        }
        for var_name, (long_name, units) in pixel_flux_meta.items():
            arr = getattr(model.time_series, var_name)
            if arr is not None:
                data_vars[var_name] = (
                    ("time_fluxes", "northing", "easting"),
                    arr.astype(np.float32),
                    {"long_name": long_name, "units": units},
                )

    # Needs to be updated
    # 7. Process Stream Pixel Hydrographs (3D: time_fluxes x northing x easting)
    if model.time_series.n_stream_pixels > 0:
        arr = model.time_series.pixel_stream_hydrograph
        if arr is not None:
            data_vars["pixel_stream_hydrograph"] = (
                ("time_fluxes", "northing", "easting"),
                arr.astype(np.float32),
                {"long_name": "Stream pixel hydrograph", "units": "m^3/s"},
            )

    # 9. Define Coordinates
    coords = {
        "northing": ("northing", northing, {"units": "m"}),
        "easting": ("easting", easting, {"units": "m"}),
        "latitude": ("latitude", latitude, {"units": "degrees_north"}),
        "longitude": ("longitude", longitude, {"units": "degrees_east"}),
        "time_maps": ("time_maps", time_maps),
        "time_states": ("time_states", time_states),
        "time_fluxes": ("time_fluxes", time_fluxes),
    }
    # needs to be updated with actual pixel row and column indices
    if n_special > 0:
        coords["special_pixel"] = ("special_pixel", np.arange(n_special))
    if n_stream > 0:
        coords["stream_pixel"] = ("stream_pixel", np.arange(n_stream))

    # 10. Extract Global Attributes (global model parameters/constants)
    attrs = {
        # resolutions and basin area
        "dx": float(model.control.dx),
        "dy": float(model.control.dy),
        "dt": float(model.control.dt),
        "basin_area_m2": float(model.params.basin_area),
        # Outlet coordinate index
        "Ioutlet_row": int(model.network.Ioutlet[0]),
        "Ioutlet_col": int(model.network.Ioutlet[1]),
        # model parameters
        "z_m": float(model.params.z_m),
        "utmzone": model.params.utmzone,
        "time_zone_shift": float(model.params.time_zone_shift),
        "SDmean0": float(model.params.SDmean0),
        "m": float(model.params.m),
        "T0": float(model.params.T0),
        "K0": float(model.params.K0),
        "d_rz": float(model.params.d_rz),
        "THETAs": float(model.params.THETAs),
        "PSIs": float(model.params.PSIs),
        "b_BC": float(model.params.b_BC),
        "albedo": float(model.params.albedo),
        "emiss": float(model.params.emiss),
        "Csoil": float(model.params.Csoil),
        "dg": float(model.params.dg),
        "h_rough": float(model.params.h_rough),
        "h_snow": float(model.params.h_snow),
        "z_snow": float(model.params.z_snow),
        "snow_emiss": float(model.params.snow_emiss),
        "RestoreAlbedo": float(model.params.RestoreAlbedo),
        "LapseRateTair": float(model.params.LapseRateTair),
        "LapseRateTdew": float(model.params.LapseRateTdew),
        "LapseRatePPT": float(model.params.LapseRatePPT),
        "manning_n_mean": float(model.params.manning_n_mean),
        "channel_width_exponent_c": float(model.params.channel_width_exponent_c),
        "channel_width_coeff_alpha": float(model.params.channel_width_coeff_alpha),
        "mass_balance_tolerance": float(model.params.mass_balance_tolerance),
        "clear_sky_atmos_emiss_model": model.params.clear_sky_atmos_emiss_model,
        "cloudy_sky_atmos_emiss_model": model.params.cloudy_sky_atmos_emiss_model,
        "clear_sky_shortwave_model_name": model.params.clear_sky_shortwave_model_name,
        "precip_water_model_name": model.params.precip_water_model_name,
        "shade_calc_flag": int(1 if model.shade.shade_calc_flag else 0),
        "gamma_dust": float(model.params.gamma_dust) if model.params.gamma_dust is not None else np.nan,
        "lat_mean": float(model.params.lat_mean),
        "lon_mean": float(model.params.lon_mean),
        "lambda_mean": float(model.params.lambda_mean),
    }

    # 11. Assemble Dataset and Export NetCDF4 with zlib compression
    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {var: {"zlib": True, "complevel": 5} for var in ds.data_vars}
    ds.to_netcdf(output_path, encoding=encoding)

    print("MOD-WET simulation completed successfully.")
    print(f"Outputs stored in NetCDF format at: {output_path}")