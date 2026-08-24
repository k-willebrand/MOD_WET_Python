import numpy as np

from src.chapter2 import air_density, sat_vapor_pressure, specific_humidity_to_vp, vp_to_specific_humidity, vapor_pressure_deficit
from src.chapter7 import field_capacity, wilting_point

def aero_resistance(z: float | np.ndarray, h: float | np.ndarray, v: np.ndarray, kappa: float = 0.4
                    ) -> np.ndarray:
    """
    Calculate aerodynamic resistance for evapotranspiration models.

    Parameters
    ----------
    z : float or numpy.ndarray
        Velocity reference height in meters.
    h : float or numpy.ndarray
        Characteristic roughness height in meters.
    v : numpy.ndarray
        Horizontal wind velocity at reference height (z) in m/s.
    kappa : float, default=0.4
        Von Karman constant (-).

    Returns
    -------
    numpy.ndarray
        Aerodynamic resistance in s/m.
    """
    d = 0.7 * h     # Zero-plane displacement height approximation
    z_0 = 0.1 * h   # Momentum roughness height approximation

    # Equation
    r_a = np.log((z - d) / z_0) ** 2 / ((kappa ** 2) * v)

    return r_a

def canopy_resistance(
    r_smin: float | np.ndarray,
    LAI: float | np.ndarray,
    incoming_solar: float | np.ndarray,
    vapor_pressure_deficit: float | np.ndarray,
    T_air: float | np.ndarray,
    theta: float | np.ndarray,
    porosity: float | np.ndarray,
    psi_s: float | np.ndarray,
    b: float | np.ndarray,
) -> float | np.ndarray:
    """
    Compute canopy resistance used in evapotranspiration models.

    Parameters
    ----------
    r_smin : float or numpy.ndarray
        Minimum stomatal resistance in s/m.
    LAI : float or numpy.ndarray
        Leaf area index (-).
    incoming_solar : float or numpy.ndarray
        Incoming solar radiation in W/m^2.
    vapor_pressure_deficit : float or numpy.ndarray
        Vapor pressure deficit in Pa.
    T_air : float or numpy.ndarray
        Air temperature in K.
    theta : float or numpy.ndarray
        Volumetric soil moisture content (-).
    porosity : float or numpy.ndarray
        Soil porosity (-).
    psi_s : float or numpy.ndarray
        Absolute value of saturated matric head in cm.
    b : float or numpy.ndarray
        Soil-type static parameter b (-).

    Returns
    -------
    r_c : float or numpy.ndarray
        Canopy resistance in s/m.
    """
    incoming_solar = np.asarray(incoming_solar, dtype=float)
    vapor_pressure_deficit = np.asarray(vapor_pressure_deficit, dtype=float)
    T_air = np.asarray(T_air, dtype=float)
    theta = np.asarray(theta, dtype=float)
    porosity = np.asarray(porosity, dtype=float)
    psi_s = np.asarray(psi_s, dtype=float)
    b = np.asarray(b, dtype=float)

    # Compute incoming solar radiation stress factor
    f_Rs = np.full_like(incoming_solar, np.nan)
    mask_solar = (incoming_solar >= 0) & (incoming_solar <= 1100)
    f_Rs[mask_solar] = 1.105 * incoming_solar[mask_solar] / (
        1.007 * incoming_solar[mask_solar] + 104.4
    )

    # Compute vapor pressure deficit stress factor
    f_de = np.full_like(vapor_pressure_deficit, np.nan)
    mask_vpd = (vapor_pressure_deficit >= 0) & (vapor_pressure_deficit <= 4200)
    f_de[mask_vpd] = 1.0 - 0.000238 * vapor_pressure_deficit[mask_vpd]

    # Convert temperature from K to deg C
    T_air_C = T_air - 273.15
    f_Ta = np.full_like(T_air_C, np.nan)
    mask_Ta = (T_air_C >= 0) & (T_air_C <= 40)
    f_Ta[mask_Ta] = (T_air_C[mask_Ta] * (40.0 - T_air_C[mask_Ta]) ** 1.18) / 690.0
    f_Ta[T_air_C < 0] = 0.0

    # Compute soil moisture stress factor
    theta_fc = field_capacity(psi_s, b, porosity)
    theta_pwp = wilting_point(psi_s, b, porosity)

    f_theta = np.full_like(theta, np.nan)
    mask_theta_mid = (theta >= theta_pwp) & (theta <= theta_fc)
    f_theta[mask_theta_mid] = (theta[mask_theta_mid] - theta_pwp[mask_theta_mid]) / (
        theta_fc[mask_theta_mid] - theta_pwp[mask_theta_mid]
    )

    mask_theta_high = (theta >= theta_fc) & (theta <= porosity)
    f_theta[mask_theta_high] = 1.0

    mask_theta_low = (theta >= 0) & (theta <= theta_pwp)
    f_theta[mask_theta_low] = 0.0

    # Compute stomatal and canopy resistance
    r_s = r_smin / (f_Rs * f_theta * f_de * f_Ta)
    r_c = r_s / LAI

    return r_c

def clausius_clapeyron_slope(T: float | np.ndarray, T_0: float = 273.15, e_s0: float = 611.0, Lv: float = 2.5e6, Rv: float = 461.0,) -> float | np.ndarray:
    """
    Compute the slope of the Clausius-Clapeyron equation at temperature T.

    Parameters
    ----------
    T : float or numpy.ndarray
        Air temperature in K.
    T_0 : float, default=273.15
        Reference temperature in Clausius-Clapeyron equation in K.
    e_s0 : float, default=611.0
        Reference saturated vapor pressure in Clausius-Clapeyron equation in Pa.
    Lv : float, default=2.5e6
        Latent heat of vaporization in J/kg.
    Rv : float, default=461.0
        Ideal gas constant of water vapor in J/kg/K.

    Returns
    -------
    delta : float or numpy.ndarray
        Slope of Clausius-Clapeyron equation in Pa/K.
    """
    e_s = sat_vapor_pressure(T, Rv=Rv, e_s0=e_s0, T_0=T_0, Lv=Lv)
    delta = (Lv / Rv) * (e_s / (T**2))
    return delta

def EBBR(R: float | np.ndarray, G: float | np.ndarray, T1: float | np.ndarray, T2: float | np.ndarray, q1: float | np.ndarray, q2: float | np.ndarray,
         cp: float = 1004.0, Lv: float = 2.5e6,) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Compute latent and sensible heat flux via Energy Balance Bowen Ratio (EBBR).

    Parameters
    ----------
    R : float or numpy.ndarray
        Net incoming radiation in W/m^2.
    G : float or numpy.ndarray
        Ground heat flux in W/m^2.
    T1 : float or numpy.ndarray
        Temperature at reference point 1 in K.
    T2 : float or numpy.ndarray
        Temperature at reference point 2 in K.
    q1 : float or numpy.ndarray
        Specific humidity at reference point 1 in kg H2O/kg air.
    q2 : float or numpy.ndarray
        Specific humidity at reference point 2 in kg H2O/kg air.
    cp : float, default=1004.0
        Specific heat capacity of air in J/kg/K.
    Lv : float, default=2.5e6
        Latent heat of vaporization in J/kg.

    Returns
    -------
    latent_heat_flux : float or numpy.ndarray
        Latent heat flux in W/m^2.
    sensible_heat_flux : float or numpy.ndarray
        Sensible heat flux in W/m^2.
    """
    B = (cp * (T2 - T1)) / (Lv * (q2 - q1))
    latent_heat_flux = (R - G) / (1.0 + B)
    sensible_heat_flux = B * latent_heat_flux

    return latent_heat_flux, sensible_heat_flux

def mass_transfer(P: np.ndarray, T_surf: np.ndarray, T_air: np.ndarray, q_air: np.ndarray, r_a: np.ndarray, r_c: float | np.ndarray,  beta: float | np.ndarray, method: int,
                  cp: float, Lv: float, Rd: float, epsilon: float, e_s0: float, Rv: float, T_0: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Computes latent and sensible heat fluxes using Mass Transfer / Diffusion Analogy."""
    e_air = specific_humidity_to_vp(q_air, P, epsilon=epsilon)
    density = air_density(T_air, e_air, P, Rd=Rd, epsilon=epsilon)
    
    esat_surf = sat_vapor_pressure(T_surf, T_0=T_0, e_s0=e_s0, Lv=Lv, Rv=Rv)
    q_sat = vp_to_specific_humidity(esat_surf, P, epsilon=epsilon)

    if method == 0:
        latent_heat_flux = density * Lv * (beta * q_sat - q_air) / (r_a + r_c)
    elif method == 1:
        latent_heat_flux = density * Lv * beta * (q_sat - q_air) / (r_a + r_c)
    else:
        latent_heat_flux = np.zeros_like(P)

    sensible_heat_flux = density * cp * (T_surf - T_air) / r_a
    return latent_heat_flux, sensible_heat_flux

def penman(Rn: float | np.ndarray, G: float | np.ndarray, T: float | np.ndarray, e: float | np.ndarray, P: float | np.ndarray, r_a: float | np.ndarray,
           cp: float = 1004.0, epsilon: float = 0.622,T_0: float = 273.15, e_s0: float = 611.0, Lv: float = 2.5e6, Rv: float = 461.0,) -> float | np.ndarray:
    """Compute potential evaporation latent heat flux using the Penman model.

    Parameters
    ----------
    Rn : float or numpy.ndarray
        Net incoming radiation in W/m^2.
    G : float or numpy.ndarray
        Ground heat flux in W/m^2.
    T : float or numpy.ndarray
        Air temperature in K.
    e : float or numpy.ndarray
        Vapor pressure in Pa.
    P : float or numpy.ndarray
        Surface pressure in Pa.
    r_a : float or numpy.ndarray
        Aerodynamic resistance in s/m.
    cp : float, default=1004.0
        Specific heat capacity of air in J/kg/K.
    epsilon : float, default=0.622
        Ratio of dry air to water vapor gas constants (-).
    T_0 : float, default=273.15
        Reference temperature in Clausius-Clapeyron equation in K.
    e_s0 : float, default=611.0
        Reference saturated vapor pressure in Clausius-Clapeyron equation in Pa.
    Lv : float, default=2.5e6
        Latent heat of vaporization in J/kg.
    Rv : float, default=461.0
        Ideal gas constant of water vapor in J/kg/K.

    Returns
    -------
    LEp : float or numpy.ndarray
        Latent heat flux under potential evaporative conditions in W/m^2.
    """
    delta = clausius_clapeyron_slope(T, T_0=T_0, e_s0=e_s0, Lv=Lv, Rv=Rv)
    gamma = psychrometric_constant(P, cp=cp, epsilon=epsilon, Lv=Lv)
    density = air_density(T, e, P)
    dq = vp_to_specific_humidity(vapor_pressure_deficit(e, T), P)

    LEp = ((delta / gamma) * (Rn - G) + (density * Lv / r_a) * dq) / (
        1.0 + delta / gamma
    )
    return LEp

def penman_monteith(Rn: float | np.ndarray, G: float | np.ndarray, T: float | np.ndarray, e: float | np.ndarray, P: float | np.ndarray, r_a: float | np.ndarray, r_c: float | np.ndarray,
                    cp: float = 1004.0, epsilon: float = 0.622,T_0: float = 273.15, e_s0: float = 611.0, Lv: float = 2.5e6, Rv: float = 461.0,) -> float | np.ndarray:
    """Compute latent heat flux using the Penman-Monteith model.

    Parameters
    ----------
    Rn : float or numpy.ndarray
        Net incoming radiation in W/m^2.
    G : float or numpy.ndarray
        Ground heat flux in W/m^2.
    T : float or numpy.ndarray
        Air temperature in K.
    e : float or numpy.ndarray
        Vapor pressure in Pa.
    P : float or numpy.ndarray
        Surface pressure in Pa.
    r_a : float or numpy.ndarray
        Aerodynamic resistance in s/m.
    r_c : float or numpy.ndarray
        Canopy resistance factor in s/m.
    cp : float, default=1004.0
        Specific heat capacity of air in J/kg/K.
    epsilon : float, default=0.622
        Ratio of dry air to water vapor gas constants (-).
    T_0 : float, default=273.15
        Reference temperature in Clausius-Clapeyron equation in K.
    e_s0 : float, default=611.0
        Reference saturated vapor pressure in Clausius-Clapeyron equation in Pa.
    Lv : float, default=2.5e6
        Latent heat of vaporization in J/kg.
    Rv : float, default=461.0
        Ideal gas constant of water vapor in J/kg/K.

    Returns
    -------
    latent_heat_flux : float or numpy.ndarray
        Latent heat flux in W/m^2.
    """
    delta = clausius_clapeyron_slope(T, T_0=T_0, e_s0=e_s0, Lv=Lv, Rv=Rv)
    gamma = psychrometric_constant(P, cp=cp, epsilon=epsilon, Lv=Lv)
    density = air_density(T, e, P)
    dq = vp_to_specific_humidity(vapor_pressure_deficit(e, T), P)

    latent_heat_flux = ((delta / gamma) * (Rn - G) + (density * Lv / r_a) * dq) / (
        1.0 + delta / gamma + r_c / r_a
    )
    return latent_heat_flux

def psychrometric_constant(P: float | np.ndarray, cp: float = 1004.0, epsilon: float = 0.622, Lv: float = 2.5e6,) -> float | np.ndarray:
    """
    Compute the psychrometric constant gamma.

    Parameters
    ----------
    P : float or numpy.ndarray
        Surface pressure in Pa.
    cp : float, default=1004.0
        Specific heat capacity of air in J/kg/K.
    epsilon : float, default=0.622
        Ratio of dry air to water vapor gas constants (-).
    Lv : float, default=2.5e6
        Latent heat of vaporization in J/kg.

    Returns
    -------
    gamma : float or numpy.ndarray
        Psychrometric constant in Pa/K.
    """
    gamma = (P * cp) / (epsilon * Lv)
    return gamma

def richardson_number(z: float | np.ndarray, Tair: np.ndarray, U: np.ndarray, Tsurf: np.ndarray, g: float
                      ) -> np.ndarray:
    """Determines the bulk Richardson number (dimensionless)."""
    return g * z * (Tair - Tsurf) / (Tsurf * (U ** 2))

def soil_SEB_solver_prognostic(
    SW: np.ndarray,
    Ta: np.ndarray,
    qa: np.ndarray,
    U: np.ndarray,
    Psfc: np.ndarray,
    LWdown: np.ndarray,
    theta_rz: np.ndarray,
    Ts0: np.ndarray,
    theta_wp: np.ndarray,
    theta_fc: np.ndarray,
    emiss: np.ndarray,
    albedo: np.ndarray,
    Td0: np.ndarray,
    z_m: float,
    h_rough: float,
    Csoil: float,
    dg: float,
    dt: float,
    SB_const: float,
    kappa: float,
    g: float,
    cp: float,
    Lv: float,
    Rd: float,
    epsilon: float,
    e_s0: float,
    Rv: float,
    T_0: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Description:
    This function solves the surface energy balance prognostically
    as a function of surface temperature.

    Inputs:
    SW: Incoming shortwave radiation (W/m^2)
    Ta: Reference-level air temperature (K)
    qa: Reference-level specific humidity (kg/kg)
    U: Reference-level wind speed (m/s)
    Psfc: Reference-level air pressure (Pa)
    LWdown: Downwelling longwave radiation (W/m^2)
    theta_rz: rootzone soil moisture (-)
    Ts0: initial guess for surface temperature (from previous time step) (K)
    theta_wp: wilting point (-)
    theta_fc: field capacity (-)
    emiss: Surface emissivity (-)
    albedo: Surface albedo (-)
    Td0: initial guess for deep soil temperature (from previous time step) (K)
    z_m: Meteorological reference measurement height (m)
    h_rough: Characteristic soil roughness height (m)
    Csoil: Soil heat capacity (J/m^3/K)
    dg: Surface soil depth (m)
    dt: Timestep (hours)
    Constants:
        SB_const: Stefan-Boltzmann constant (W/m^2/K^4)
        kappa: Von Karman constant (-)
        g: Gravitational acceleration (m/s^2)
        cp: Specific heat capacity of air (J/kg/K)
        Lv: Latent heat of vaporization (J/kg)
        Rd: Gas constant for dry air (J/kg/K)
        epsilon: Ratio of molecular weights (H2O/Dry Air)
        e_s0: Saturation vapor pressure at reference temperature (Pa)
        Rv: Gas constant for water vapor (J/kg/K)
        T_0: Reference temperature (K)

    Outputs:
    Tsurf: Surface temperature (K)
    LE: Latent heat flux (W/m^2)
    H: Sensible heat flux (W/m^2)
    G: Ground heat flux (W/m^2)
    Rn: Net radiation (W/m^2)
    Td: Deep soil temperature [K]
    LWup: Upwelling longwave radiation (W/m^2)
    """
    omega = 1.0 / 86400.0  # diurnal frequency (1/s)
    
    # Compute net shortwave radiation (W/m^2)
    SW_net = SW * (1.0 - albedo)

    # Compute net radiation
    LWup = emiss * SB_const * (Ts0 ** 4)
    Rn = SW_net + LWdown - LWup

    # Compute actual evaporation
    beta = np.full_like(theta_rz, np.nan)
    beta[theta_rz >= theta_fc] = 1.0
    beta[theta_rz <= theta_wp] = 0.0
    ind = np.isnan(beta)
    beta[ind] = (theta_rz[ind] - theta_wp[ind]) / (theta_fc[ind] - theta_wp[ind])

    # Compute aerodynamic resistance
    # check for near-zero (less than 0.5 m/s) windspeed and set to low, but
    # positive value
    U_calc = np.copy(U)
    U_calc[U_calc < 0.5] = 0.5  # m/s
    r_a = aero_resistance(z_m, h_rough, U_calc, kappa=kappa)

    # use stability corrections
    RiB = richardson_number(z=z_m, Tair=Ta, U=U_calc, Tsurf=Ts0, g=g)
    phi_m, phi_h = stab_corr_factors(RiB)
    r_a = r_a * phi_m * phi_h

    ET_method_flag = 1
    LE, H = mass_transfer(Psfc, Ts0, Ta, qa, r_a, 0, beta, ET_method_flag, cp, Lv, Rd, epsilon, e_s0, Rv, T_0)
    CT = 1.0 / (Csoil * dg)
    G = Rn - LE - H

    # Mimics the force restore equation (with a constant soil heat capacity)
    Tsurf = Ts0 + dt * (CT * G - 2.0 * np.pi * omega * (Ts0 - Td0)) * 3600.0
    Td = Td0 + dt * (omega * (Ts0 - Td0)) * 3600.0

    # # Safety Guard: Prevent explicit Euler overshoots relative to air temp
    # Tsurf = np.clip(Tsurf, Ta - 25.0, Ta + 25.0)

    return Tsurf, LE, H, G, Rn, Td, LWup

def stab_corr_factors(RiB: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Determines atmospheric stability correction factors from the bulk Richardson number."""
    # Takes care of RiB values above the allowable limit
    # RiB_capped = np.minimum(RiB, 0.19) # CAUSES massive thresholding swings when wind speed is close to 0.19, DO NOT USE
    RiB_capped = np.where(RiB >= 0.2, 0.19, RiB) # replicates original MATLAB behavior, more stable

    phi_m = np.full_like(RiB_capped, np.nan, dtype=np.float64)
    phi_h = np.full_like(RiB_capped, np.nan, dtype=np.float64)

    # Unstable case (RiB <= 0)
    unstable = RiB_capped <= 0.0
    phi_h[unstable] = (1.0 - 15.0 * RiB_capped[unstable]) ** -0.5
    phi_m[unstable] = phi_h[unstable] ** 0.5

    # Stable case (0 < RiB < 0.2)
    stable = (RiB_capped > 0.0) & (RiB_capped < 0.2)
    phi_h[stable] = (1.0 - 5.0 * RiB_capped[stable]) ** -1.0
    phi_m[stable] = phi_h[stable]

    return phi_m, phi_h

