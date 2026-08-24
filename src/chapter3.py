import numpy as np
import re
from tqdm import tqdm
from typing import Tuple, Union, Optional
from pyproj import Transformer

from src.chapter2 import Wp_from_near_surface_met_data

# Note: utm2deg.m from original MATLAB code overwritten by new parse_utm_epsg function and 
#       revised easting_northing_to_lat_lon function, which uses pyproj library

def clear_sky_emiss(e: np.ndarray, T: np.ndarray, model_name: str, T_0: float = 273.15, e_s0: float = 611.0, Lv: float = 2.5e6, Rv: float = 461.0,) -> np.ndarray:
    """Computes clear-sky atmospheric emissivity.
    Inputs:
        e: (near-surface) Vapor pressure in mb (1mb=100Pa)
        T: (near-surface) air temperature (K)
        clear_model_name : descriptor of which model to use:
                = 'brunt' : use Brunt (1932) model
                = 'brutsaert' : use Brutsaert (1975) model (default)
                = 'satterlund' : use Satterlund (1979) model
                = 'prata' : use Prata (1996) model
                = 'idso' : use Idso (1981?) model
    *** Note: default fall-back is brutsaert
    """
    model_name = str(model_name).lower()
    if model_name == 'brunt':
        return 0.605 + 0.048 * np.sqrt(e)
    elif model_name == 'brutsaert':
        return 1.24 * (e / T) ** (0.14)
    elif model_name == 'satterlund':
        return 1.08 * (1.0 - np.exp(- (e ** (T / 2016.0))))
    elif model_name == 'prata':
        # convert vapor pressure to Pa for precip. water function call
        e = e * 100 # Pa
        # compute precipitable water (in cm)
        Wp = Wp_from_near_surface_met_data(e, T, model_name, T_0=T_0, e_s0=e_s0, Lv=Lv, Rv=Rv)
        return 1.0 - (1.0 + Wp) * np.exp(- np.sqrt(1.2 + 3.0 * Wp))
    elif model_name == 'idso':
        return 0.74 + 0.0049 * e
    else:
        # Default fallback (Brutsaert)
        return 1.24 * (e / T) ** (0.14)

def clear_sky_shortwave_radiation(
    RsTOA: Union[float, np.ndarray],
    zenith_rad: Union[float, np.ndarray],
    Wp: Union[float, np.ndarray],
    gamma_dust: float,
    albedo: Union[float, np.ndarray],
    surface_pressure: Union[float, np.ndarray],
    model_name: str,
) -> Union[float, np.ndarray]:
    """Computes clear-sky downward shortwave radiation (W/m^2) at the surface.

    Args:
        RsTOA: Top-of-atmosphere shortwave solar flux (W/m^2)
        zenith_rad: Solar zenith angle (radians)
        Wp: Precipitable water (cm)
        gamma_dust: Nondimensional dust coefficient (used in Dingman model)
        albedo: Surface albedo (-)
        surface_pressure: Surface atmospheric pressure (Pa)
        model_name: Model selection string ('dingman' or 'crawford')

    Returns:
        Union[float, np.ndarray]: Downwelling clear-sky shortwave radiation (W/m^2)
    """
    model = model_name.lower()

    if model == "dingman":
        # Optical thickness / air mass
        Mopt = optical_depth(zenith_rad)

        # Direct transmissivity and diffuse scattering coefficient
        t_s = direct_sw_transmissivity(Wp, gamma_dust, Mopt)
        beta = diffuse_sw_scattering_coefficient(Wp, gamma_dust, Mopt)

        attenuation_factor = (
            t_s + beta + beta * albedo * t_s + (beta**2) * albedo
        )
        return RsTOA * attenuation_factor

    elif model == "crawford":
        cos_z = np.cos(zenith_rad)

        # Optical air mass
        m = 35.0 * cos_z * (1224.0 * (cos_z**2) + 1.0) ** (-0.5)

        # Convert surface pressure from Pa to kPa
        p_kPa = surface_pressure / 1000.0

        # Transmission coefficients for Rayleigh scattering, water vapor, and aerosols
        tau_R_tau_pg = 1.021 - 0.084 * np.sqrt(m * (0.00949 * p_kPa + 0.051))
        tau_w = 1.0 - 0.077 * np.power(Wp * m, 0.3)
        tau_a = np.power(0.935, m)

        return RsTOA * tau_R_tau_pg * tau_w * tau_a

    else:
        raise ValueError(
            f"Unsupported clear-sky shortwave model: '{model_name}'. Expected 'dingman' or 'crawford'."
        )

def cloudy_sky_emiss(e: np.ndarray, T: np.ndarray, clear_model_name: str, C: float, S: float, cloudy_model_name: str,
                     T_0: float, e_s0: float, Lv: float, Rv: float) -> np.ndarray:
    """Computes effective atmospheric emissivity under clear or cloudy sky conditions.
        Inputs:
        e: (near-surface) Vapor pressure in mb (1mb=100Pa)
        T: (near-surface) air temperature (K)
        clear_model_name : descriptor of which model to use:
                = 'brunt' : use Brunt (1932) model
                = 'brutsaert' : use Brutsaert (1975) model
                = 'satterlund' : use Satterlund (1979) model
                = 'prata' : use Prata (1996) model
                = 'idso' : use Idso (1981?) model
        C: cloud-cover fraction (used in Kustas model)
        S: solar index (used in Prata model)
        cloudy_model_name : descriptor of which model to use:
                = 'kustas' : use Kustas (1994) model
                = 'crawford' : use Crawford and Duchon (1999) model
    """
    clear_sky_atmos_emissivity = clear_sky_emiss(e, T, clear_model_name, T_0=T_0, e_s0=e_s0, Lv=Lv, Rv=Rv)

    if cloudy_model_name == 'kustas':
        return (1.0 + 0.22 * (C ** 2)) * clear_sky_atmos_emissivity
    elif cloudy_model_name == 'crawford':
        return (1.0 - S) + S * clear_sky_atmos_emissivity
    else:
        return clear_sky_atmos_emissivity

def compute_shade_lookup_table_and_SVF(easting: np.ndarray, northing: np.ndarray, hterrain: np.ndarray, slope: np.ndarray, aspect: np.ndarray
                                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a topographic shade lookup table and Sky View Factor (SVF) map.

    Rather than explicitly computing shade for every hour of year
    (function of zenith and azimuth) this code instead generates a shade 
    lookup table for a prescribed set of zenith and azimuth angles.  This 
    lookup table can then be interpolated to determine the dynamic shade for 
    a given hour/pixel.  The sky view factor (SVF) is a static parameter and 
    can also be determined from the lookup table.

    Parameters
    ----------
    easting : np.ndarray
        1D array of x-coordinates (meters).
    northing : np.ndarray
        1D array of y-coordinates (meters).
    hterrain : np.ndarray
        2D DEM grid.
    slope : np.ndarray
        2D array of slope angles in degrees.
    aspect : np.ndarray
        2D array of aspect angles in degrees.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - shade_lookup_table : 4D uint8 array (nrows, ncols, n_zenith, n_azimuth)
        - SVF : 2D float array of Sky View Factor
        - discrete_zenith_values : 1D array of zenith angles (degrees)
        - discrete_azimuth_values : 1D array of azimuth angles (degrees)
    """
    dzenith = 5  # degrees
    discrete_zenith_values = np.arange(0, 91, dzenith)

    dazimuth = 15  # degrees
    discrete_azimuth_values = np.arange(0, 360, dazimuth)

    nrows, ncols = hterrain.shape
    n_zen = len(discrete_zenith_values)
    n_az = len(discrete_azimuth_values)

    shade_lookup_table = np.zeros(
        (nrows, ncols, n_zen, n_az), dtype=np.uint8
    )
    horizon_angle = np.zeros((nrows * ncols, n_az))

    # Azimuth Loop
    for iaz in tqdm(range(n_az), desc="Shade Lookup Progress"):
        az_deg = discrete_azimuth_values[iaz]
        sazimuth = np.radians(az_deg)

        # Zenith = 0 degrees (unobstructed)
        shade = np.ones((nrows, ncols), dtype=np.uint8)
        shade_lookup_table[:, :, 0, iaz] = shade

        # Remaining Zenith Angles
        for izen in range(1, n_zen):
            zen_deg = discrete_zenith_values[izen]
            # print(f"Running shade calc. for: Azimuth: {az_deg}; Zenith: {zen_deg}")

            saltitude = np.radians(90.0 - zen_deg)
            shade = topo_shade_calc(
                saltitude, sazimuth, easting, northing, hterrain
            ).astype(np.uint8)

            # Local zenith angle calculation
            # NOTE: Preserving MATLAB sind(cosd(zenith)) formula for exact matching
            coszen_local = (
                np.cos(np.radians(zen_deg)) * np.cos(np.radians(slope))
                + np.sin(np.radians(np.cos(np.radians(zen_deg))))
                * np.sin(np.radians(slope))
                * np.cos(np.radians(az_deg - aspect))
            )

            # Mask slopes that are locally shaded
            locally_shaded = coszen_local < 0
            shade[locally_shaded] = 0

            # Find pixels shaded for the first time along this azimuth
            coszen_flat = coszen_local.reshape(-1)
            shade_flat = shade.reshape(-1)

            I = ((shade_flat == 0) | (coszen_flat < 0)) & (
                horizon_angle[:, iaz] == 0
            )

            # Horizon angle defined as an elevation/altitude angle
            horizon_angle[I, iaz] = saltitude
            shade_lookup_table[:, :, izen, iaz] = shade

    # Reshape horizon angle into map format
    horizon_angle_map = horizon_angle.reshape((nrows, ncols, n_az))

    # SVF calculation (Dozier and Marks, 1987)
    mean_horizon_angle = np.mean(horizon_angle_map, axis=2)
    SVF = np.cos(mean_horizon_angle) ** 2

    return shade_lookup_table, SVF, discrete_zenith_values, discrete_azimuth_values

def diffuse_sw_scattering_coefficient(
    Wp: Union[float, np.ndarray],
    gamma_dust: float,
    Mopt: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Computes diffuse beam shortwave scattering coefficient (-) via Dingman model."""
    a_s = -0.0363 - 0.0084 * Wp
    b_s = -0.0572 - 0.0173 * Wp
    tau_s = np.exp(a_s + b_s * Mopt)
    return 0.5 * (1.0 - tau_s + gamma_dust)

def direct_sw_transmissivity(
    Wp: Union[float, np.ndarray],
    gamma_dust: float,
    Mopt: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Computes direct beam shortwave transmissivity (-) via Dingman model."""
    a_sa = -0.124 - 0.0207 * Wp
    b_sa = -0.0682 - 0.0248 * Wp
    tau_sa = np.exp(a_sa + b_sa * Mopt)
    return tau_sa - gamma_dust

def disaggregate_SW(
    SWin: np.ndarray,
    press: np.ndarray,
    slope_rad: np.ndarray,
    aspect_rad: np.ndarray,
    zenith_rad: float,
    azimuth_rad: float,
    hrangle: float,
    shade: np.ndarray,
    SVF: np.ndarray,
    sunrise: float,
    sunset: float,
    RsTOA: float,
    mask: np.ndarray,
    albedo: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description:
    This function disaggregates the incoming shortwave radiation flux at the
    suface by accounting for topographic (slope and aspect) effects.

    Based on disaggregation methods in Allen et al. (2006) and Muller and 
    Scherer (2005) and modified for application in CEE 150.

    Inputs:
    SWin: Incoming shortwave radiation data on a horizontal plane array (W/m^2)
    press: Disaggregated pressure data array (Pa)
    slope_rad: Slope (radians)
    aspect_rad: Aspect (radians)
    zenith_rad: Solar zenith angle (radians)
    azimuth_rad: Solar azimuth angle (radians)
    hrangle: Hour angle (radians)
    shade: Shade array
    SVF: Sky view factor array
    sunrise: Local hour of sunrise
    sunset: Local hour of sunset
    RsTOA: TOA solar flux at central lat/lon (W/m^2)
    mask: Basin mask array
    albedo: albedo map array

    Outputs:
    Rs: Total incident radiation at the surface (W/m^2)
    RsDir: Direct beam solar flux: RsDir (W/m^2)
    RsDif: Diffuse solar flux (W/m^2)

    Note: Can run for a single time over a spatial domain
    """
    nx, ny = mask.shape

    # Assumed parameters
    Gamma_SW = 1.0  # Solar flux correction factor
    press_mean_areal = np.nanmean(press)
    AlbedoSurrounding = np.full((nx, ny), np.nan)
    AlbedoSurrounding[:, :] = albedo  # Surrounding albedo associated with mean areal data

    # Allocate
    tau_SW_hor = np.full((nx, ny), np.nan)
    K_B_hor = np.full((nx, ny), np.nan)
    RsDir = np.full((nx, ny), np.nan)
    fcor = np.full((nx, ny), np.nan)
    RsDif = np.full((nx, ny), np.nan)

    # Perform over mask
    imask = (mask == 1)

    # Cosine of solar zenith angle
    cos_sza_hor = np.cos(zenith_rad)

    # Ensure SWin is a broadcastable array matching spatial grid shape
    SWin = np.array(SWin, dtype=float)
    if SWin.ndim == 0:
        SWin = np.full((nx, ny), SWin)

    # Adjust mean areal SW forcings
    SWin = SWin * Gamma_SW

    ## Disaggregate incoming shortwave radiation at the surface from mean areal data:
    # Partition direct from diffuse flux

    # For clear sky: broadband atmos. transmissivity= Solar insolation horizontal / TOA solar radiation
    if RsTOA == 0.0:
        tau_SW_hor[imask] = 0.0
    else:
        tau_SW_hor[imask] = SWin[imask] / RsTOA

    # Direct beam transmissivity (based on Allen et al. 2006 paper)
    ind = (tau_SW_hor <= 0.175) & (mask == 1)
    K_B_hor[ind] = 0.016 * tau_SW_hor[ind]

    ind = (tau_SW_hor > 0.175) & (tau_SW_hor < 0.42) & (mask == 1)
    K_B_hor[ind] = (0.022 - 0.280 * tau_SW_hor[ind] +
                    0.828 * tau_SW_hor[ind]**2.0 + 0.765 * tau_SW_hor[ind]**3.0)

    ind = (tau_SW_hor >= 0.42) & (mask == 1)
    K_B_hor[ind] = 1.56 * tau_SW_hor[ind] - 0.55

    ind = (K_B_hor > tau_SW_hor) & (mask == 1)
    K_B_hor[ind] = tau_SW_hor[ind]

    # Diffuse beam transmissivity
    K_D_hor = tau_SW_hor - K_B_hor

    K_B_hor[K_B_hor < 0.0] = 0.0

    # Check tau values at night vs. day
    SWin[tau_SW_hor == 0.0] = 0.0
    SWin[SWin < 0.0] = 0.0

    # Elevation angle
    ThetaS = np.pi / 2.0 - zenith_rad  # radians

    # Sunrise/sunset at local time
    sunrise_rad = (sunrise - 12.0) * np.pi / 12.0
    sunset_rad = (sunset - 12.0) * np.pi / 12.0

    ## Disaggregate direct component for Elevation/Topography:
    with np.errstate(divide='ignore'): # suppress divide by zero encountered in log warning
        t_MeanAreal = -np.log(np.nanmean(K_B_hor))  # Optical depth associated with mean areal data
    tElev = t_MeanAreal * (press / press_mean_areal)  # At the pixel elevation optical depth
    KB_elev_Sch = np.exp(-tElev)  # Trasmissivity of the direct component at the pixel elevation
    RsDir_Elev = KB_elev_Sch * RsTOA  # Direct flux at the pixel elevation

    # Check if slope is obstructed from sun and apply thresholds to avoid 
    # numerical issues at times close to sunset/sunrise.
    with np.errstate(divide='ignore', invalid='ignore'): # suppress divide by zero encountered in divide warning
        ConstCheck = (np.tan(slope_rad) / np.tan(ThetaS) * np.cos(azimuth_rad - aspect_rad) + 1.0)
    ind = ((ConstCheck > 0.0) & (cos_sza_hor > 0.001) &
           (hrangle > (sunrise_rad + 0.001)) & (hrangle < (sunset_rad - 0.001)) &
           (ThetaS > (np.pi / 180.0)) & (mask == 1))

    # Slice spatial arrays where applicable
    slope_i = slope_rad[ind] if isinstance(slope_rad, np.ndarray) and slope_rad.shape == mask.shape else slope_rad
    aspect_i = aspect_rad[ind] if isinstance(aspect_rad, np.ndarray) and aspect_rad.shape == mask.shape else aspect_rad
    shade_i = shade[ind] if isinstance(shade, np.ndarray) and shade.shape == mask.shape else shade

    fcor[ind] = shade_i * (1.0 + np.tan(slope_i) / np.tan(ThetaS) * np.cos(azimuth_rad - aspect_i))
    RsDir[ind] = RsDir_Elev[ind] * fcor[ind]

    # Account for remaining pixels in mask
    ind2 = imask & ~ind
    RsDir[ind2] = 0.0

    ## Disaggregate diffuse component for elevation/SVF/reflection term...
    # Define the mean areal data diffuse component. Apply thresholds to avoid 
    # numerical issues at times close to sunset/sunrise.
    ind = ((tau_SW_hor > 0.001) & (hrangle > (sunrise_rad + 0.001)) &
           (hrangle < (sunset_rad - 0.001)) & (cos_sza_hor > 0.001) &
           (K_D_hor > 0.001) & (ThetaS > (np.pi / 180.0)) & (mask == 1))
    E_Sdiff = SWin[ind] * K_D_hor[ind] / tau_SW_hor[ind]

    # Disaggregate diffuse component for elevation
    P0 = 101325.0  # Nominal surface pressure, Pa
    Mz = (1.0 - 0.027 * np.exp(2.0 * press[ind] / P0)) * (1.075 - 0.105 * np.log(1.0 / cos_sza_hor))
    M_MeanAreal = (1.0 - 0.027 * np.exp(2.0 * press_mean_areal / P0)) * (1.075 - 0.105 * np.log(1.0 / cos_sza_hor))
    with np.errstate(over='ignore'): # suppress overflow encountered in exp warning
        RsDif_Elev = E_Sdiff * ((Mz - np.exp(-tElev[ind] / cos_sza_hor)) / (M_MeanAreal - np.exp(-t_MeanAreal / cos_sza_hor)))

    # Reflection term
    SVF_i = SVF[ind] if isinstance(SVF, np.ndarray) and SVF.shape == mask.shape else SVF
    ToTSW_Elev = RsDif_Elev * SVF_i + RsDir[ind]
    RsDif[ind] = RsDif_Elev * SVF_i + (AlbedoSurrounding[ind] * ToTSW_Elev) * (1.0 - SVF_i)

    # Account for remaining pixels in mask
    ind2 = imask & ~ind
    RsDif[ind2] = 0.0

    ## Compute total SW flux
    Rs = RsDif + RsDir  # W/m^2

    return Rs, RsDir, RsDif

def easting_northing_to_lat_lon(easting_1d: np.ndarray, northing_1d: np.ndarray, utm_zone_str: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert 1D Easting and Northing UTM vectors into 1D lat/lon vectors"""

    # Parse UTM zone string 
    epsg = parse_utm_epsg(utm_zone_str)

    transformer = Transformer.from_crs(
        f"EPSG:{epsg}", "EPSG:4326", always_xy=True
    )

    # Create 2D meshgrid arrays from 1D coordinate vectors
    E, N = np.meshgrid(easting_1d, northing_1d)
    lon_2d, lat_2d = transformer.transform(E, N)

    # Collapse to 1D vectors
    lat_1d = np.mean(lat_2d, axis=1)  # Mean across columns
    lon_1d = np.mean(lon_2d, axis=0)  # Mean across rows

    return lat_1d, lon_1d

def generate_slope_and_aspect_from_DEM(elev: np.ndarray, easting: np.ndarray, northing: np.ndarray
                                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    Description:
    Description:
    Function to create slope and aspect maps from DEM (using ArcGIS algorithm)
    
    Inputs:
    elev: elevation matrix (meters)
    easting: x coordinate (meters)
    northing: y coordinate (meters)
    
    Outputs:
    slope: slope in degrees
    aspect: aspect direction in degrees; Note: Aspect uses the ArcGIS
            convention which has for aspect directions: 
            0 deg. == due NORTH
            90 deg. == due EAST
            180 deg. == due SOUTH
            270 deg. == due WEST
    """
    # 1. Coordinate Trend Orienting & Flipping Logic
    flip_flag_y = False
    flip_flag_x = False

    # Check to make sure DEM is in correct orientation for the calculations;
    # where correct orientation has the NW corner in element (0,0) in Python.
    if northing[-1] > northing[0]:  # Northing coordinate is increasing downward
        elev = np.flipud(elev)      # Needs to be flipped
        flip_flag_y = True

    if easting[-1] < easting[0]:    # Easting is decreasing to the east
        elev = np.fliplr(elev)      # Needs to be flipped
        flip_flag_x = True

    nrows, ncols = elev.shape
    x_cell_size = abs(easting[0] - easting[1])
    y_cell_size = abs(northing[0] - northing[1])

    # Initialize slope/aspect arrays with NaN (matching MATLAB's no_data_value = NaN)
    slope = np.full((nrows, ncols), np.nan)
    aspect = np.full((nrows, ncols), np.nan)

    # Loop through interior pixels (0-based: 1 to nrows-2)
    for k in range(1, nrows - 1):
        for j in range(1, ncols - 1):
            
            # Check for NaN values in 3x3 neighborhood (ignore calculations if data is missing)
            neighbors = elev[k-1:k+2, j-1:j+2]
            if np.any(np.isnan(neighbors)):
                continue

            # Elevations at each pixel matching ArcGIS 3x3 window grid:
            # a  b  c
            # d  e  f
            # g  h  i
            # where "e" is the pixel of interest

            a, b, c = elev[k-1, j-1], elev[k-1, j], elev[k-1, j+1]
            d, e, f = elev[k, j-1],   elev[k, j],   elev[k, j+1]
            g, h, i = elev[k+1, j-1], elev[k+1, j], elev[k+1, j+1]

            # Taken directly from ArcGIS documentation formulas
            dz_dx = ((c + 2*f + i) - (a + 2*d + g)) / (8 * x_cell_size)
            dz_dy = ((g + 2*h + i) - (a + 2*b + c)) / (8 * y_cell_size)

            slope_degrees = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)) * 180.0 / np.pi
            slope[k, j] = slope_degrees
 
            # Aspect calculation (ignoring cell-size multipliers for pure direction)
            dz_dx_nonelem = ((c + 2*f + i) - (a + 2*d + g)) / 8.0
            dz_dy_nonelem = ((g + 2*h + i) - (a + 2*b + c)) / 8.0

            aspect_degrees = 180.0 / np.pi * np.arctan2(dz_dy_nonelem, -dz_dx_nonelem)
            
            # Apply ArcGIS aspect compass conventions (0° = North, 90° = East, etc.)
            if aspect_degrees < 0:
                aspect[k, j] = 90.0 - aspect_degrees
            elif aspect_degrees > 90.0:
                aspect[k, j] = 360.0 - aspect_degrees + 90.0
            else:
                aspect[k, j] = 90.0 - aspect_degrees

    # Restore the original array orientation if they were flipped
    if flip_flag_y:
        slope = np.flipud(slope)
        aspect = np.flipud(aspect)

    if flip_flag_x:
        slope = np.fliplr(slope)
        aspect = np.fliplr(aspect)

    return slope, aspect

def longwave_flux(e: float | np.ndarray, Ta: float | np.ndarray, Ts: float | np.ndarray, surf_emissivity: float | np.ndarray, SB_const: float = 5.67e-8,
                  ) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """
    Compute the net, incoming, and outgoing longwave fluxes.

    Parameters
    ----------
    e : float or numpy.ndarray
        Vapor pressure in mb.
    Ta : float or numpy.ndarray
        Air temperature in K.
    Ts : float or numpy.ndarray
        Surface temperature in K.
    surf_emissivity : float or numpy.ndarray
        Surface emissivity.
    SB_const : float, default=5.67e-8
        Stefan-Boltzmann constant in W/m^2/K^4.

    Returns
    -------
    Rl_net : float or numpy.ndarray
        Net longwave flux in W/m^2.
    Rl_in : float or numpy.ndarray
        Incoming longwave flux in W/m^2.
    Rl_out : float or numpy.ndarray
        Outgoing longwave flux in W/m^2.

    Notes
    -----
    This function can run for an array of inputs.
    Requires clear_sky_emiss function.
    """
    # Compute longwave using Idso model for atmospheric emissivity
    Rl_in = clear_sky_emiss(e, Ta, "idso") * SB_const * Ta**4
    Rl_out = surf_emissivity * SB_const * Ts**4

    # Compute net longwave
    Rl_net = Rl_in - Rl_out

    return Rl_net, Rl_in, Rl_out

def optical_depth(
    zenith_rad: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Computes atmospheric optical depth mass from solar zenith angle (radians)."""
    return 1.0 / np.cos(zenith_rad)

def parse_utm_epsg(utmzone_str: str) -> int:
    """Parse any UTM zone string into its corresponding EPSG code."""
    # Matches zone number and optional band/hemisphere indicator
    match = re.match(r"^(\d{1,2})\s*([A-Za-z]+)?$", utmzone_str.strip())
    if not match:
        raise ValueError(f"Invalid UTM zone format: '{utmzone_str}'. Standard format is [<Zone Number> <Latitude Band Letter>] (e.g., 10 S, 32 T, 18 M)")
    zone = int(match.group(1))
    if not (1 <= zone <= 60):
        raise ValueError(f"UTM zone must be between 1 and 60, got {zone}")
    indicator = (match.group(2) or "N").upper()
    # 1. Handle explicit hemisphere words
    if indicator in ("SOUTH", "SOUTHERN"):
        is_south = True
    elif indicator in ("NORTH", "NORTHERN"):
        is_south = False
    # 2. Handle MGRS Latitude Bands (C through M = South; N through X = North)
    else:
        is_south = indicator[0] <= "M"
    return (32700 if is_south else 32600) + zone

def solar_geometry(DOY: float | np.ndarray, UTC: float | np.ndarray, time_zone_shift: float, lat_deg: float, lon_deg: float
                   ) -> tuple[float | np.ndarray,
                              float | np.ndarray,
                              float | np.ndarray,
                              float | np.ndarray,
                              float | np.ndarray,
                              float | np.ndarray]:
    """Computes solar zenith/azimuth angles, sunrise/sunset hours, declination, and hour angle.

    Returns:
        Tuple: (zenith_angle_deg, azimuth_angle_deg, sunrise, sunset, solar_decl, hour_angle)
    """
    latrad = np.radians(lat_deg)

    # Local time and Day-of-Year adjustment for time zone
    time_local = UTC + time_zone_shift
    doy_adjusted = np.where(time_local < 0.0, DOY - 1.0, DOY)
    time_local = np.where(time_local < 0.0, 24.0 + time_local, time_local)

    # Day angle in radians
    day_angle = 2.0 * np.pi * (doy_adjusted - 1.0) / 365.0

    # Solar declination angle (radians)
    solar_decl = (
        0.006918
        - 0.399912 * np.cos(day_angle)
        + 0.070257 * np.sin(day_angle)
        - 0.006758 * np.cos(2.0 * day_angle)
        + 0.000907 * np.sin(2.0 * day_angle)
        - 0.002697 * np.cos(3.0 * day_angle)
        + 0.001480 * np.sin(3.0 * day_angle)
    )

    # Local standard time meridian (degrees) & Equation of Time (minutes)
    LSTM = 15.0 * time_zone_shift
    B = 360.0 / 365.0 * (doy_adjusted - 81.0)
    EofT_min2 = (
        9.87 * np.sin(np.radians(2.0 * B))
        - 7.53 * np.cos(np.radians(B))
        - 1.5 * np.sin(np.radians(B))
    )

    # Time correction (minutes) and local solar time (hours)
    TC = 4.0 * (lon_deg - LSTM) + EofT_min2
    LST = time_local + TC / 60.0

    # Hour angle (radians)
    hour_angle = 15.0 * (LST - 12.0) * np.pi / 180.0

    # Solar zenith angle (radians)
    cos_zenith = np.sin(latrad) * np.sin(solar_decl) + np.cos(latrad) * np.cos(
        solar_decl
    ) * np.cos(hour_angle)
    zenith_angle = np.arccos(np.clip(cos_zenith, -1.0, 1.0))

    # Solar azimuth angle (radians)
    cos_azimuth = (
        np.sin(solar_decl) * np.cos(latrad)
        - np.cos(solar_decl) * np.sin(latrad) * np.cos(hour_angle)
    ) / np.sin(zenith_angle)
    azimuth_angle = np.arccos(np.clip(cos_azimuth, -1.0, 1.0))
    azimuth_angle = np.where(
        LST > 12.0, 2.0 * np.pi - azimuth_angle, azimuth_angle
    )

    zenith_angle_deg = np.degrees(zenith_angle)
    azimuth_angle_deg = np.degrees(azimuth_angle)

    # Sunrise / sunset hours in local time
    cos_sun_angle = -np.sin(latrad) * np.sin(solar_decl) / (
        np.cos(latrad) * np.cos(solar_decl)
    )
    sun_hour_term = (180.0 / (15.0 * np.pi)) * np.arccos(
        np.clip(cos_sun_angle, -1.0, 1.0)
    )

    sunrise = 12.0 - sun_hour_term - TC / 60.0
    sunset = 12.0 + sun_hour_term - TC / 60.0

    return (
        zenith_angle_deg,
        azimuth_angle_deg,
        sunrise,
        sunset,
        solar_decl,
        hour_angle,
    )

def sw_cloud_attenuation_factor(C: float | np.ndarray,) -> float | np.ndarray:
    """
    Compute the empirical shortwave cloud attenuation factor based on Bras (1990).

    Parameters
    ----------
    C : float or numpy.ndarray
        Areal cloud cover fraction (-).

    Returns
    -------
    f_sc : float or numpy.ndarray
        Shortwave cloud attenuation factor.
    """
    # Shortwave cloud attenuation factor
    f_sc = 1.0 - 0.65 * C**2
    return f_sc

def TOA_incoming_solar(DOY: float | np.ndarray, UTC: float | np.ndarray, time_zone_shift: float, lat_deg: float, lon_deg: float, S0: float
                       ) -> tuple[float | np.ndarray,
                                  float | np.ndarray,
                                  float | np.ndarray,
                                  float | np.ndarray,
                                  float | np.ndarray,
                                  float | np.ndarray]:
    """Computes Top of Atmosphere (TOA) incident solar flux and solar geometry parameters.

    Returns:
        Tuple: (RsTOA, zenith_angle_deg, azimuth_angle_deg, sunrise, sunset, solar_decl, hour_angle)
    """
    # Compute solar geometry parameters via helper function
    (
        zenith_angle_deg,
        azimuth_angle_deg,
        sunrise,
        sunset,
        solar_decl,
        hour_angle,
    ) = solar_geometry(DOY, UTC, time_zone_shift, lat_deg, lon_deg)

    # Ratio of actual to mean Earth-Sun distance (-)
    r = 1.0 + 0.017 * np.cos(2.0 * np.pi / 365.0 * (186.0 - DOY))

    # Cap zenith angle at 90 degrees (below horizon set to horizon)
    zenith_angle_deg = np.minimum(zenith_angle_deg, 90.0)

    # Convert zenith angle to radians
    theta = np.radians(zenith_angle_deg)

    # Calculate TOA Solar Radiation (W/m^2)
    RsTOA = S0 * np.cos(theta) / (r**2)

    return (
        RsTOA,
        zenith_angle_deg,
        azimuth_angle_deg,
        sunrise,
        sunset,
        solar_decl,
        hour_angle,
    )

def topo_shade_calc(saltitude: float, sazimuth: float, easting: np.ndarray, northing: np.ndarray, elev: np.ndarray
                    ) -> np.ndarray:
    """
    Calculates topographic shading for a given solar altitude and azimuth.

    Adapted from ARPS model subroutine.

    The code works as follows:
        The solar geometry defines the quadrant that the sun is inhabiting;
        Based on the sun's quadrant, the code starts with the pixels furthest
        from the sun (i.e. extreme of opposite quadrant);
        From the starting point, a line is constructed between the pixel and
        the sun, which defines a projected line on the 2D domain;
        Along the line all pixels are checked to see if they are higher than
        the original pixel; if any are then the original pixel is in shade.
        The checking breaks as soon as one pixel of higher elevation is found;
        The code then moves on to the next pixel and does the same check;
        After this is done for each pixel the shade matrix is fully filled-in.

    Parameters
    ----------
    saltitude : float
        Solar altitude angle in radians (angle above horizontal).
    sazimuth : float
        Solar azimuth angle in radians (0 = North, pi/2 = East, pi = South,
        3pi/2 = West).
    easting : np.ndarray
        1D array of x-coordinates (meters), increasing left-to-right.
    northing : np.ndarray
        1D array of y-coordinates (meters), decreasing top-to-bottom.
    elev : np.ndarray
        2D array of DEM topography (map orientation).

    Returns
    -------
    np.ndarray
        2D binary shade matrix (1 = illuminated, 0 = shaded).
    """
    # Rotate elevation matrix into x-y orientation from row-column orientation
    hterrain = np.rot90(elev, k=-1)
    nxlg, nylg = hterrain.shape

    dx = float(abs(easting[1] - easting[0]))
    dy = float(abs(northing[0] - northing[1]))

    xlg = np.arange(1, nxlg + 1) * dx
    ylg = np.arange(1, nylg + 1) * dy

    shlg = np.ones((nxlg, nylg), dtype=np.uint8)

    pi2 = np.pi / 2.0
    pi32 = 3.0 * np.pi / 2.0
    p2i = 2.0 * np.pi

    # Check azimuth bounds
    if sazimuth > p2i:
        sazimuth -= p2i
    if sazimuth < 0:
        sazimuth += p2i

    # Translate azimuth for trigonometric circle inside subroutine
    if 0 <= sazimuth <= pi2:
        sazimuth = pi2 - sazimuth
    else:
        sazimuth = p2i + pi2 - sazimuth

    # Determine quadrant parameters
    if 0 <= sazimuth < pi2:  # NE
        a, b, C, D, E, F, H = 1, 1, 1, 1, 0, 0, 0
    elif pi2 <= sazimuth < np.pi:  # NW
        a, b, C, D, E, F, H = -1, 1, 2, 0, 1, 0, 0
    elif np.pi <= sazimuth < pi32:  # SW
        a, b, C, D, E, F, H = -1, -1, 3, 0, 0, 1, 0
    else:  # SE
        a, b, C, D, E, F, H = 1, -1, 4, 0, 0, 0, 1

    # Determine loop ranges based on propagation direction
    i_range = range(0, nxlg, 1) if a > 0 else range(nxlg - 1, -1, -1)
    j_range = range(0, nylg, 1) if b > 0 else range(nylg - 1, -1, -1)

    # Check for direct N, S, E, W rays
    is_cardinal = (
        abs(sazimuth - 0.0) < 0.0001
        or abs(sazimuth - pi2) < 0.0001
        or abs(sazimuth - np.pi) < 0.0001
        or abs(sazimuth - pi32) < 0.0001
    )

    if is_cardinal:
        for i in i_range:
            for j in j_range:
                xg = xlg[i]
                yg = ylg[j]

                Sx = xlg[i] + a * dx * D + a * dx * F
                Sy = ylg[j] + b * dy * E + b * dy * H

                for _ in range(2 * nxlg + 2 * nylg):
                    if (
                        xg >= xlg[-1]
                        or yg >= ylg[-1]
                        or xg <= xlg[0]
                        or yg <= ylg[0]
                    ):
                        break

                    ztest = hterrain[i, j] + np.sqrt(
                        (Sx - xlg[i]) ** 2 + (Sy - ylg[j]) ** 2
                    ) * np.tan(saltitude)

                    xh = int(np.round((xg + (D + F) * a * dx) / dx)) - 1
                    yh = int(np.round((yg + (E + H) * b * dy) / dy)) - 1

                    if xh < 0 or xh >= nxlg or yh < 0 or yh >= nylg:
                        break

                    htest = hterrain[xh, yh]
                    if ztest < htest:
                        shlg[i, j] = 0
                        break

                    xg += a * dx * (D + F)
                    yg += b * dy * (E + H)
                    Sx += a * dx * (D + F)
                    Sy += b * dy * (E + H)
    else:
        # General case (NE, NW, SW, SE)
        for i in i_range:
            for j in j_range:
                xg = xlg[i]
                yg = ylg[j]

                Sx = xlg[i] + a * (b * (yg + b * dy) - b * ylg[j]) * (
                    (D + F) * np.tan(C * pi2 - sazimuth)
                    + (E + H) * np.tan(sazimuth - (C - 1.0) * pi2)
                )
                Sy = ylg[j] + b * (a * (xg + a * dx) - a * xlg[i]) * (
                    (E + H) * np.tan(C * pi2 - sazimuth)
                    + (D + F) * np.tan(sazimuth - (C - 1.0) * pi2)
                )

                for _ in range(2 * nxlg + 2 * nylg):
                    if (
                        xg >= xlg[-1]
                        or yg >= ylg[-1]
                        or xg <= xlg[0]
                        or yg <= ylg[0]
                    ):
                        break

                    if (
                        abs(Sy - (yg + b * dy)) < 0.01
                        and abs(Sx - (xg + a * dx)) < 0.01
                    ):
                        ztest = hterrain[i, j] + np.sqrt(
                            (Sx - xlg[i]) ** 2 + (Sy - ylg[j]) ** 2
                        ) * np.tan(saltitude)

                        xh = int(np.round((xg + a * dx) / dx)) - 1
                        yh = int(np.round((yg + b * dy) / dy)) - 1

                        if xh < 0 or xh >= nxlg or yh < 0 or yh >= nylg:
                            break

                        htest = hterrain[xh, yh]
                        if ztest < htest:
                            shlg[i, j] = 0
                            break

                        xg += a * dx
                        yg += b * dy

                        Sx = xlg[i] + a * (b * (yg + b * dy) - b * ylg[j]) * (
                            (D + F) * np.tan(C * pi2 - sazimuth)
                            + (E + H) * np.tan(sazimuth - (C - 1.0) * pi2)
                        )
                        Sy = ylg[j] + b * (a * (xg + a * dx) - a * xlg[i]) * (
                            (E + H) * np.tan(C * pi2 - sazimuth)
                            + (D + F) * np.tan(sazimuth - (C - 1.0) * pi2)
                        )

                    elif abs(Sy - yg) > dy:
                        ztest = hterrain[i, j] + np.sqrt(
                            (Sx - xlg[i]) ** 2
                            + ((yg + b * dy) - ylg[j]) ** 2
                        ) * np.tan(saltitude)

                        xh = int(np.round((xg + a * dx) / dx)) - 1
                        yh = int(np.round((yg + b * dy) / dy)) - 1
                        if xh < 0 or xh >= nxlg or yh < 0 or yh >= nylg:
                            break

                        xhh = int(np.round(xg / dx)) - 1
                        yhh = int(np.round((yg + b * dy) / dy)) - 1
                        if xhh < 0 or xhh >= nxlg or yhh < 0 or yhh >= nylg:
                            break

                        htest = (
                            a * (Sx - xg) * hterrain[xh, yh]
                            + a * ((xg + a * dx) - Sx) * hterrain[xhh, yhh]
                        ) / dx

                        if ztest < htest:
                            shlg[i, j] = 0
                            break

                        yg += b * dy

                        Sx = xlg[i] + a * (b * (yg + b * dy) - b * ylg[j]) * (
                            (D + F) * np.tan(C * pi2 - sazimuth)
                            + (E + H) * np.tan(sazimuth - (C - 1.0) * pi2)
                        )
                        Sy = ylg[j] + b * (a * (xg + a * dx) - a * xlg[i]) * (
                            (E + H) * np.tan(C * pi2 - sazimuth)
                            + (D + F) * np.tan(sazimuth - (C - 1.0) * pi2)
                        )

                    elif abs(Sy - yg) < dy:
                        ztest = hterrain[i, j] + np.sqrt(
                            ((xg + a * dx) - xlg[i]) ** 2 + (Sy - ylg[j]) ** 2
                        ) * np.tan(saltitude)

                        xh = int(np.round((xg + a * dx) / dx)) - 1
                        yh = int(np.round((yg + b * dy) / dy)) - 1
                        if xh < 0 or xh >= nxlg or yh < 0 or yh >= nylg:
                            break

                        xhh = int(np.round((xg + a * dx) / dx)) - 1
                        yhh = int(np.round(yg / dy)) - 1
                        if xhh < 0 or xhh >= nxlg or yhh < 0 or yhh >= nylg:
                            break

                        htest = (
                            b * (Sy - yg) * hterrain[xh, yh]
                            + b * ((yg + b * dy) - Sy) * hterrain[xhh, yhh]
                        ) / dy

                        if ztest < htest:
                            shlg[i, j] = 0
                            break

                        xg += a * dx

                        Sx = xlg[i] + a * (b * (yg + b * dy) - b * ylg[j]) * (
                            (D + F) * np.tan(C * pi2 - sazimuth)
                            + (E + H) * np.tan(sazimuth - (C - 1.0) * pi2)
                        )
                        Sy = ylg[j] + b * (a * (xg + a * dx) - a * xlg[i]) * (
                            (E + H) * np.tan(C * pi2 - sazimuth)
                            + (D + F) * np.tan(sazimuth - (C - 1.0) * pi2)
                        )

    # Extrapolation in the direction of the sun
    if C == 1:
        for i in range(1, nxlg - 3):
            shlg[i, nylg - 3] = shlg[i, nylg - 4]
        for j in range(1, nylg - 3):
            shlg[nxlg - 3, j] = shlg[nxlg - 4, j]
        shlg[nxlg - 3, nylg - 3] = shlg[nxlg - 4, nylg - 4]

    if C == 2:
        for i in range(2, nxlg - 2):
            shlg[i, nylg - 3] = shlg[i, nylg - 4]
        for j in range(1, nylg - 3):
            shlg[1, j] = shlg[2, j]
        shlg[1, nylg - 3] = shlg[2, nylg - 4]

    if C == 3:
        for i in range(2, nxlg - 2):
            shlg[i, 1] = shlg[i, 2]
        for j in range(2, nylg - 2):
            shlg[1, j] = shlg[2, j]
        shlg[1, 1] = shlg[2, 2]

    if C == 4:
        for i in range(1, nxlg - 3):
            shlg[i, 1] = shlg[i, 2]
        for j in range(2, nylg - 2):
            shlg[nxlg - 3, j] = shlg[nxlg - 4, j]
        shlg[nxlg - 3, 1] = shlg[nxlg - 4, 2]

    # Extrapolation at outer boundaries
    for i in range(1, nxlg - 2):
        shlg[i, 0] = shlg[i, 1]
        shlg[i, nylg - 2] = shlg[i, nylg - 3]

    for j in range(1, nylg - 2):
        shlg[0, j] = shlg[1, j]
        shlg[nxlg - 2, j] = shlg[nxlg - 3, j]

    shlg[0, 0] = shlg[1, 1]
    shlg[0, nylg - 2] = shlg[1, nylg - 3]
    shlg[nxlg - 2, 0] = shlg[nxlg - 3, 1]
    shlg[nxlg - 2, nylg - 2] = shlg[nxlg - 3, nylg - 3]

    for i in range(0, nxlg - 1):
        shlg[i, nylg - 1] = shlg[i, nylg - 2]
    for j in range(0, nylg - 1):
        shlg[nxlg - 1, j] = shlg[nxlg - 2, j]

    shlg[nxlg - 1, nylg - 1] = shlg[nxlg - 2, nylg - 2]

    # Rotate shade matrix back to match input DEM orientation
    shade = np.rot90(shlg, k=1)
    return shade

