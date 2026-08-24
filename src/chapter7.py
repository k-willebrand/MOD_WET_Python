import numpy as np

def brooks_corey_PSI_and_K(theta: float | np.ndarray, porosity: float | np.ndarray, Ks: float | np.ndarray, psi_s: float | np.ndarray, b: float | np.ndarray,
                           ) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Calculate unsaturated zone matric head and hydraulic conductivity based on Brooks and Corey (1964).

    Parameters
    ----------
    theta : float or numpy.ndarray
        Volumetric soil moisture (-).
    porosity : float or numpy.ndarray
        Porosity (-).
    Ks : float or numpy.ndarray
        Saturated conductivity [L/T].
    psi_s : float or numpy.ndarray
        Saturated matric head [L].
    b : float or numpy.ndarray
        Brooks-Corey soil parameter (-).

    Returns
    -------
    psi : float or numpy.ndarray
        Matric head [L].
    K : float or numpy.ndarray
        Unsaturated hydraulic conductivity [L/T].

    Notes
    -----
    The matric head and hydraulic conductivity are output with the same units as the
    saturated matric head and saturated hydraulic conductivity, respectively. This
    function can run for an array of inputs.
    """
    # Matric head equation
    psi = psi_s * (theta / porosity) ** (-b)

    # Hydraulic conductivity equation
    K = Ks * (theta / porosity) ** (2.0 * b + 3.0)

    return psi, K

def field_capacity(psi_s: float | np.ndarray, b: float | np.ndarray, porosity: float | np.ndarray,) -> float | np.ndarray:
    """
    Compute the soil field capacity volumetric soil moisture.

    Parameters
    ----------
    psi_s : float or numpy.ndarray
        Saturated matric head [cm].
    b : float or numpy.ndarray
        Brooks-Corey soil parameter (-).
    porosity : float or numpy.ndarray
        Porosity (-).

    Returns
    -------
    theta_fc : float or numpy.ndarray
        Field capacity volumetric soil moisture (-).
    """
    # Equation
    theta_fc = porosity * (340.0 / np.abs(psi_s)) ** (-1.0 / b)
    return theta_fc

def green_ampt(Fc: np.ndarray, theta_0: float, porosity: float, Ks: float, psi_s: float, b: float
               ) -> tuple[np.ndarray, np.ndarray]:
    """
    Description:
    This function computes the infiltration capacity implicitly for a given
    range of cumulative infiltration values using the Green-Ampt model

    Inputs:
        Fc: Discretized range of cumulative infiltration (in cm)
        theta_0: Initial soil moisture
        porosity: Soil porosity
        Ks: Saturated hydraulic conductivity (in cm/hr)
        psi_s: Saturated matric head (in cm)
        b: Brooks-Corey "b" parameter

    Outputs:
        fc: Infiltration capacity vector with the i-th element corresponding to
            the time at the i-th element of the t vector(in cm/hr)
        t:  Storm time vector with the i-th element corresponding to
            the infiltration capacity at the i-th element of fc (in hr)
    """
    # Effective matric head at wetting front
    psi_f = ((2.0 * b + 3.0) / (b + 3.0)) * np.abs(psi_s)

    # Discretized Time
    # Handle Fc=0 safely to avoid division by zero log warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (1.0 / Ks) * (
            Fc
            + psi_f
            * (porosity - theta_0)
            * np.log((psi_f * (porosity - theta_0)) / (Fc + psi_f * (porosity - theta_0)))
        )
        t[Fc == 0] = 0.0

        # Discretized Infiltration Capacity
        fc = Ks * (1.0 + (psi_f * (porosity - theta_0)) / Fc)
        fc[Fc == 0] = np.inf

    return fc, t

def philip(t: np.ndarray | float, theta_0: np.ndarray | float,  porosity: np.ndarray | float,  Ks: np.ndarray | float,  psi_s: np.ndarray | float,  b: np.ndarray | float
           ) -> tuple[np.ndarray | float, np.ndarray | float]:
    """
    Description:
    This function computes the infiltration capacity and cumulative
    infiltration for a given time using the Philip Equation

    Inputs:
        t: Elapsed time of storm, must be between 0 and total storm duration (in hr)
        theta_0: Initial soil moisture
        porosity: Soil porosity
        Ks: Saturated hydraulic conductivity (in cm/hr)
        psi_s: Saturated matric head (in cm)
        b: Brooks-Corey "b" parameter

    Outputs:
        fc: Infiltration capacity (in cm/hr)
        Fc: Cumulative infiltration capacity (in cm)
    """
    # Sorptivity Equation Sp in cm/hr^.5
    Sp = sorptivity(theta_0, porosity, Ks, psi_s, b)

    # Philip Equation fc [cm/hr]
    fc = (Sp / 2.0) * (t ** (-0.5)) + Ks

    Fc = Sp * (t**0.5) + Ks * t  # cm

    return fc, Fc

def sorptivity(theta_0: np.ndarray | float, porosity: np.ndarray | float, Ks: np.ndarray | float, psi_s: np.ndarray | float, b: np.ndarray | float
               ) -> np.ndarray | float:
    """
    Description:
    This function computes the soil sorptivity

    Inputs:
        theta_0: Initial soil moisture
        porosity: Soil porosity
        Ks: Saturated hydraulic conductivity (in cm/hr)
        psi_s: saturated matric head (in cm) [e.g. -21.8 cm]
        b: Brooks-Corey "b" parameter

    Outputs:
        Sp: Soil sorptivity (in cm/hr^.5)
    """
    # Sorptivity Equation
    Sp = np.sqrt((porosity - theta_0) * Ks * np.abs(psi_s) * ((2.0 * b + 3.0) / (b + 3.0)))

    return Sp

def TCA_infiltration(P: np.ndarray | float, tr: float, t: float, theta_0: np.ndarray, porosity: np.ndarray, Ks: np.ndarray, psi_s: np.ndarray, b: np.ndarray, method_flag: int
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Description:
    This function computes the actual infiltration and cumulative
    infiltration using the time compression approximation method with either
    the Philip solution or the Green-Ampt model

    Inputs:
        P: Precipitation rate of storm event over duration tr (in cm/hr)
        tr: Duration of storm event of precipitation rate P (in hr)
        t: Elapsed time of storm, must be between 0 and total storm duration (in hr)
        theta_0: Initial soil moisture
        porosity: Soil porosity
        Ks: Saturated hydraulic conductivity (in cm/hr)
        psi_s: Saturated matric head (in cm)
        b: Brooks-Corey "b" parameter
        method_flag: A flag denoting the method used to compute infiltration.
            method_flag=1 corresponds to the Philip solution
            method_flag=2 corresponds to the Green-Ampt model

    Outputs:
        F: Actual cumulative infiltration (in cm)
        qie: Infiltration excess runoff (in cm)
    """
    # If input time is past the duration of the storm or prior to storm, return 0
    if t > tr or t <= 0:
        return np.zeros_like(theta_0, dtype=float), np.zeros_like(theta_0, dtype=float)

    # Determine size of input map and initialize output map
    F = np.zeros_like(theta_0, dtype=float)
    qie = np.zeros_like(theta_0, dtype=float)

    # Define the maximum cumulative infiltration to be the
    # precipitation rate * time of interest
    Fcmax = P * t

    if method_flag == 1:  # Philip
        # Sorptivity Equation cm/hr^.5
        Sp = sorptivity(theta_0, porosity, Ks, psi_s, b)

        # Compute ponding time [hr]
        tp = (Sp**2 / (2.0 * P * (P - Ks))) * (1.0 + Ks / (2.0 * (P - Ks)))

        # Determine time compression [hr]
        tc = tp - (Sp / (2.0 * (P - Ks))) ** 2

        # Apply Philip model to pixels where ponding has occurred
        I = np.where((t > tp) & (tp >= 0))
        if len(I[0]) > 0:
            _, F[I] = philip(t - tc[I], theta_0[I], porosity[I], Ks[I], psi_s[I], b[I])

            # Compute cumulative infiltration
            Ftp = np.zeros_like(theta_0, dtype=float)
            _, Ftp[I] = philip(tp[I] - tc[I], theta_0[I], porosity[I], Ks[I], psi_s[I], b[I])
            # F[I] = F[I] - Ftp[I] + P[I] * tp
            # Changed by SM:
            F[I] = F[I] - Ftp[I] + P[I] * tp[I]
            # Compute infiltration excess runoff for ponded pixels
            qie[I] = P[I] * t - F[I]

        # Find pixels where ponding has not occurred
        I_unponded = np.where(t <= tp)

        # Set infiltration at unponded pixels to precipitation rate and
        # compute cumulative infiltration and runoff
        F[I_unponded] = P[I_unponded] * t
        qie[I_unponded] = 0.0

    elif method_flag == 2:  # Green Ampt
        # Compute effective matric head at wetting front
        psi_f = ((2.0 * b + 3.0) / (b + 3.0)) * np.abs(psi_s)

        # Compute ponding time
        tp = (Ks / (P * (P - Ks))) * psi_f * (porosity - theta_0)

        # Determine time compression
        tc = tp - (1.0 / Ks) * (
            P * tp
            + psi_f
            * (porosity - theta_0)
            * np.log((psi_f * (porosity - theta_0)) / (P * tp + psi_f * (porosity - theta_0)))
        )

        # For saturated pixels
        I_sat = np.where(porosity == theta_0)
        F[I_sat] = Ks[I_sat] * t  # Compute cumulative infiltration
        qie[I_sat] = P[I_sat] * t - F[I_sat]  # Compute infiltration excess runoff

        # For unponded pixels
        I_unponded = np.where(t <= tp)
        F[I_unponded] = P[I_unponded] * t  # Compute cumulative infiltration
        qie[I_unponded] = 0.0  # No runoff is generated from unponded pixels

        # Loop through map to compute infiltration at ponded pixels.
        for idx in np.ndindex(theta_0.shape):
            # If ponding has occurred and pixel is not saturated
            if t > tp[idx] and porosity[idx] != theta_0[idx]:
                # Discretize cumulative infiltration using 51 elements
                Fc_val = Fcmax[idx] if np.ndim(Fcmax) > 0 else Fcmax
                Fc = np.linspace(0.0, Fc_val, 51)

                # Compute infiltration using Green-Ampt
                f0, t0 = green_ampt(Fc, theta_0[idx], porosity[idx], Ks[idx], psi_s[idx], b[idx])

                # Discard irrelevant portions of Fc, f0, and t0 and
                # recompute to improve interpolation accuracy
                I_over = np.where(t0 > t)[0]
                if len(I_over) > 0:
                    Fc = np.linspace(0.0, Fc[I_over[0]], 51)
                    f0, t0 = green_ampt(Fc, theta_0[idx], porosity[idx], Ks[idx], psi_s[idx], b[idx])

                # Interpolate to compute cumulative infiltration at time t
                F[idx] = np.interp(t - tc[idx], t0, Fc)
                Ftp = np.interp(tp[idx] - tc[idx], t0, Fc)
                F[idx] = F[idx] - Ftp + P[idx] * tp[idx]

                # Compute infiltration excess runoff
                qie[idx] = P[idx] * t - F[idx]

    else:
        raise ValueError("Invalid method_flag input")

    # If conductivity > precipitation
    I_ks = np.where(Ks >= P)
    F[I_ks] = P[I_ks] * t  # all water will infiltrate
    qie[I_ks] = 0.0  # no runoff generated

    return F, qie

def wilting_point(psi_s: float | np.ndarray, b: float | np.ndarray, porosity: float | np.ndarray,) -> float | np.ndarray:
    """Compute the soil permanent wilting point volumetric soil moisture.

    Parameters
    ----------
    psi_s : float or numpy.ndarray
        Saturated matric head [cm].
    b : float or numpy.ndarray
        Brooks-Corey soil parameter (-).
    porosity : float or numpy.ndarray
        Porosity (-).

    Returns
    -------
    theta_wp : float or numpy.ndarray
        Permanent wilting point volumetric soil moisture (-).
    """
    # Equation
    theta_wp = porosity * (15000.0 / np.abs(psi_s)) ** (-1.0 / b)
    return theta_wp

