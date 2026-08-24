import numpy as np
import matplotlib.pyplot as plt

from src.chapter2 import sat_vapor_pressure

def compute_mean_precip_from_thiessen(rain_matrix: np.ndarray, rel_gage_area: np.ndarray,) -> np.ndarray | None:
    """
    Compute mean areal precipitation using the Thiessen polygon method.

    Parameters
    ----------
    rain_matrix : numpy.ndarray
        A Ntime x Ngage matrix corresponding to the rain measurements
        at each of the Ngage gages for Ntime timesteps.
    rel_gage_area : numpy.ndarray
        A vector containing the Thiessen areal weights of each gage.

    Returns
    -------
    meanP : numpy.ndarray or None
        A Ntime x 1 vector corresponding to the mean areal precip. for Ntime
        timesteps, or None if gage counts do not match.

    Notes
    -----
    The function thiessen must be called prior to running this function.
    """
    rel_gage_area = np.asarray(rel_gage_area).flatten()
    rain_matrix = np.asarray(rain_matrix)

    # Determine the number of gages
    Ngages = len(rel_gage_area)

    # Confirm the correct # of gages and determine the number of timesteps in the gage record
    Ntime, Ngages_test = rain_matrix.shape
    if Ngages_test != Ngages:
        print(
            "The number of gages in the precip. matrix does not match those in the gage coordinates matrix"
        )
        return None

    # Compute mean areal precip by multiplying elementwise and summing across gages
    meanP = np.sum(rain_matrix * rel_gage_area, axis=1, keepdims=True)

    return meanP

def sat_adiabatic_lapse_rate(T: float | np.ndarray, p: float | np.ndarray, 
                             gamma_d: float = 9.800, cp: float = 1004.0, epsilon: float = 0.622, 
                             T_0: float = 273.15, e_s0: float = 611.0, Lv: float = 2.5e6, Rv: float = 461.0) -> float | np.ndarray:
    """
    Compute the saturated adiabatic lapse rate.

    Parameters
    ----------
    T : float or numpy.ndarray
        Air temperature in K.
    p : float or numpy.ndarray
        Air pressure in Pa.
    gamma_d : float, default=9.800
        Dry adiabatic lapse rate in K/km.
    Lv : float, default=2.5e6
        Latent heat of vaporization in J/kg.
    cp : float, default=1004.0
        Specific heat capacity of air in J/kg/K.
    epsilon : float, default=0.622
        Ratio of dry air to water vapor gas constants (Rd/Rv) (-).
    Rv : float, default=461.0
        Ideal gas constant of water vapor in J/kg/K.
    e_s0 : float, default=611.0
        Reference saturated vapor pressure in Clausius-Clapeyron equation in Pa.
    T_0 : float, default=273.15
        Reference temperature in Clausius-Clapeyron equation in K.

    Returns
    -------
    gamma_s : float or numpy.ndarray
        Saturated adiabatic lapse rate in K/km.

    Notes
    -----
    This function can run for an array of inputs.

    Examples
    --------
    >>> gamma_s = sat_adiabatic_lapse_rate(270.0, 90000.0)
    >>> print(f"{gamma_s:.4f}")
    6.0477

    >>> T = np.array([270.0, 290.0, 300.0])
    >>> P = np.array([90000.0, 70000.0, 100000.0])
    >>> gamma_s = sat_adiabatic_lapse_rate(T, P)
    """
    # Equation
    e_s = sat_vapor_pressure(T, Rv=Rv, e_s0=e_s0, T_0=T_0, Lv=Lv)
    gamma_s = gamma_d / (1.0 + (Lv / cp) * ((epsilon / p) * (Lv / Rv) * (e_s / (T**2))))

    return gamma_s

def thiessen(mask: np.ndarray, easting: np.ndarray, northing: np.ndarray, gage_coordinates: np.ndarray,  plot_results: bool = True,) -> tuple[np.ndarray, np.ndarray]:
    """
    Implement the Thiessen polygon method to compute mean areal precipitation weights.

    Parameters
    ----------
    mask : numpy.ndarray
        A J x I matrix filled with 1.0 and np.nan. Covers the rectangular domain
        including the basin of interest. All points inside the basin should have
        values of 1.0 and outside should have values of np.nan.
    easting : numpy.ndarray
        X-coordinate vector of dimension I.
    northing : numpy.ndarray
        Y-coordinate vector of dimension J.
    gage_coordinates : numpy.ndarray
        An Ngage x 2 matrix with the first column corresponding to the easting
        coordinate of each gage and the second column corresponding to the northing.
    plot_results : bool, default=True
        Optionally render the Thiessen polygon visualization map.

    Returns
    -------
    gage_assignment_matrix : numpy.ndarray
        A J x I matrix filled with np.nan outside the basin and integers 1 to Ngage
        inside the basin corresponding to the nearest gage.
    rel_gage_area : numpy.ndarray
        A 1 x Ngage vector containing the relative area (area weight) for each gage.
    """
    mask = np.asarray(mask, dtype=float)
    easting = np.asarray(easting)
    northing = np.asarray(northing)
    gage_coordinates = np.asarray(gage_coordinates)

    # Specify the # of gages to be used in the analysis
    Ngages = gage_coordinates.shape[0]

    # Find grid matrix coordinate (column, row) of each gage
    coordinates = np.zeros((Ngages, 2), dtype=int)
    for ii in range(Ngages):
        coordinates[ii, 0] = np.argmin(np.abs(gage_coordinates[ii, 0] - easting))
        coordinates[ii, 1] = np.argmin(np.abs(gage_coordinates[ii, 1] - northing))

    # Determine Row/Column dimensions of rectangular domain encompassing basin from mask
    J, I = mask.shape

    # Create a matrix in which each cell will be filled by index of closest gage (1-indexed)
    gage_assignment_matrix = np.zeros((J, I), dtype=float)

    # Loop through each pixel in domain assigning closest gage
    for i in range(I):
        for j in range(J):
            D = np.zeros(Ngages)
            # Check to see which gage is closest to current point (i,j)
            for k in range(Ngages):
                D[k] = np.sqrt((i - coordinates[k, 0]) ** 2 + (j - coordinates[k, 1]) ** 2)

            # Determine minimum distance
            Dmin = np.min(D)

            # Find minimum distance gage and check for equidistant gages
            indices_of_nearest_gages = np.where(D == Dmin)[0]

            if len(indices_of_nearest_gages) == 1:
                # Store 1-based gage index to match MATLAB conventions
                gage_assignment_matrix[j, i] = indices_of_nearest_gages[0] + 1
            else:
                # Randomly assign one of the equidistant nearest gages
                chosen_gage = np.random.choice(indices_of_nearest_gages)
                gage_assignment_matrix[j, i] = chosen_gage + 1

    # Apply basin mask (multiplying sets outside cells to NaN)
    gage_assignment_matrix = gage_assignment_matrix * mask

    # Compute gage weights
    basin_squares = ~np.isnan(gage_assignment_matrix) & (gage_assignment_matrix > 0)
    num_total_squares = np.sum(basin_squares)

    rel_gage_area = np.zeros(Ngages)
    for k in range(Ngages):
        gage_k_squares = np.sum(gage_assignment_matrix == (k + 1))
        rel_gage_area[k] = gage_k_squares / num_total_squares

    # Plot Thiessen polygons by plotting gage assignment matrix
    if plot_results:
        plt.figure()
        plt.imshow(
            gage_assignment_matrix,
            extent=[easting.min(), easting.max(), northing.min(), northing.max()],
            origin="lower",
            aspect="auto",
        )
        plt.plot(
            gage_coordinates[:, 0],
            gage_coordinates[:, 1],
            "w+",
            markersize=10,
            markeredgewidth=2,
        )
        plt.title("Thiessen Polygons Map")
        plt.xlabel("Easting (m)")
        plt.ylabel("Northing (m)")
        plt.colorbar(label="Gage Assignment Index")
        plt.show()

    return gage_assignment_matrix, rel_gage_area

