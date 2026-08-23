from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Sequence, Tuple, Optional
import csv
import numpy as np
import scipy.sparse as sp

@dataclass
class ControlParameters:
    """Simulation control and time-stepping parameters."""

    # Display / Print Flags
    print_screen_flag: bool = False

    # Simulation Timing Parameters
    dt: float = 0.25                                        # Timestep (hours)
    start_day: float = 0.0                                  # Starting day in forcing file
    n_days: int = 365                                       # Number of simulation days
    timeseries_frq2store: int = 4                           # Frequency to store time series
    map_frq2store: int = 4 * 24                             # Frequency to store maps (e.g., 96 = daily)
    frq2screen: int = 24 * 4 * 10                           # Frequency of screen updates

    # Routing Settings
    routing_time_step_flag: bool = True                     # Dynamic time-stepping flag

    # Spatial Discretization (populated when DEM is loaded)
    dx: float = 0.0                                         # Easting resolution (m)
    dy: float = 0.0                                         # Northing resolution (m)
    nx: int = 0                                             # Number of grid rows
    ny: int = 0                                             # Number of grid columns

    # Derived Time Parameters (automatically calculated in __post_init__)
    ntime: int = field(init=False)
    nt: int = field(init=False)
    start_time: int = field(init=False)

    def __post_init__(self) -> None:
        self.compute_derived_time_params()

    def compute_derived_time_params(self) -> None:
        """Compute total timesteps, storage length, and start index."""
        # Total number of simulation timesteps
        self.ntime = int(round((1.0 / self.dt) * 24.0 * self.n_days))

        # Length of stored time series vector
        self.nt = int(self.ntime // self.timeseries_frq2store)

        # Length of stored time series vector
        self.n_map = int(self.ntime // self.map_frq2store)

        # 0-based starting time index (MATLAB: 1 + start_day * 24 / dt)
        self.start_time = int(round(self.start_day * 24.0 / self.dt))

    def update_spatial_grid(
        self, dx: float, dy: float, elev_shape: tuple[int, int]
    ) -> None:
        """Populate grid resolution and dimensions when loading DEM data."""
        self.dx = dx
        self.dy = dy
        self.nx, self.ny = elev_shape

@dataclass
class PhysicalConstants:
    """Universal physical constants."""
    cp: float = 1004.0                                      # Specific heat capacity of air (J/kg/K)
    rhow: float = 1000.0                                    # Density of water (kg/m^3)
    rhoi: float = 917.0                                     # Density of ice (kg/m^3)
    ci: float = 2102.0                                      # Specific heat capacity of ice (J/kg/K)
    cw: float = 4216.0                                      # Specific heat capacity of water (J/kg/K)
    Rd: float = 287                                         # Ideal gas constant of dry air (J/kg/K)
    Rv: float = 461                                         # Ideal gas constant of water vapor (J/kg/K)
    epsilon: float = 0.622                                  # Rd/Rv (-)
    e_s0: float = 611                                       # Reference staurated vapor pressure in Clausius-Clapeyron Equatioin (Pa)
    T_0: float = 273.15                                     # Reference temperature in Clausius-Clapeyron Equation (K)
    Lv: float = 2.5e6                                       # Latent heat of vaporzation (J/kg)
    Lf: float = 3.34e5                                      # Latent heat of fusion (J/kg)
    Ls: float = 2.83e6                                      # Latent heat of sublimation (J/kg)
    g: float = 9.81                                         # Acceleration of gravity (m/s^2)
    S0: float = 1367                                        # Solar constant in (W/m^2)
    SB_const: float = 5.67e-8                               # Stefan-Boltzman constant (W/m^2/K^4)
    kappa: float = 0.4                                      # Von Karman constant (-)
    N0: float = 0.08                                        # Marshall-Palmer parameter (cm^-4)
    gamma: float = 64.6                                     # Psychrometric constant (Pa/K)
    T_f: float = 273.15                                     # Water freezing temp.(K)
    gamma_d: float = 9.800                                  # dry adiabatic lapse rate (K/km)

    @classmethod
    def load(cls, csv_path: Optional[str | Path] = None) -> "PhysicalConstants":
        """
        Instantiates PhysicalConstants using default values unless a valid CSV path 
        is provided to override specified parameters.
        """
        if csv_path is None:
            return cls()

        path = Path(csv_path)
        if path.exists():
            print(f"File '{csv_path}' will be used to override default physical constants.")

            # Retrieve valid attribute names from the dataclass
            valid_fields = {f.name for f in fields(cls)}
            overrides = {}

            with open(path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = row["key"].strip()
                    val_str = row["value"].strip()
                    if key in valid_fields and val_str:
                        overrides[key] = float(val_str)

            # Unpack CSV overrides over default dataclass values
            return cls(**overrides)
        else:
            print(f"Warning: File '{csv_path}' not found. Using default physical constants.")
            return cls()

@dataclass
class ModelParameters:
    """Scalar model parameters, configuration settings, and basin-wide properties."""

    # Measurement Reference Heights & Time
    z_m: float = 2.0                                        # Meteorological reference-level measurement height (m)
    utmzone: str = "10 S"                                   # UTM zone identifier string
    time_zone_shift: float = -8.0                           # Time zone shift between UTC and local time (hours)

    # TOPMODEL & Groundwater Initial State
    SDmean0: float = 0.1                                    # Initial average saturation deficit (m)
    m: float = 0.02                                         # Decay factor of lateral transmissivity w.r.t. saturation deficit (m)
    T0: float = 0.06                                        # Surface transmissivity (m^2/hour)
    # K0, the surface saturated hydraulic conductivity (m/hour), is defined below

    # Soil Properties (Homogeneous across basin)
    d_rz: float = 0.5                                       # Depth of rootzone (m)
    THETAs: float = 0.435                                   # Soil porosity (-)
    PSIs: float = 0.2180                                    # Absolute value of saturated matric head (m)
    b_BC: float = 4.9                                       # Brooks-Corey exponent parameter (-)
    albedo: float = 0.25                                    # Surface soil albedo (-)
    emiss: float = 0.9                                      # Surface soil emissivity (-)
    Csoil: float = 1.3e6                                    # Soil heat capacity (J/m^3/K)
    dg: float = 0.05                                        # Surface soil depth (m)
    h_rough: float = 0.05                                   # Characteristic soil roughness height (m)

    # Snow Model Parameters
    h_snow: float = 0.03                                    # Height of snow roughness elements (m)
    z_snow: float = 2.0                                     # Meteorological ref.-level height over snow (m)
    snow_emiss: float = 0.99                                # Snow surface emissivity (-)
    RestoreAlbedo: float = 1 * 100 / 24                     # Snowfall rate required to reset albedo to "new snow" (mm/h)

    # Meteorological Disaggregation Lapse Rates
    LapseRateTair: float = -0.0065                          # Constant lapse rate in air temperature (K/m)
    LapseRateTdew: float = -0.0046                          # Constant lapse rate in air dewpoint temperature (K/m)
    LapseRatePPT: float = 0.35                              # Lapse rate parameter for precipitation (1/km)

    # Channel & Routing Parameters
    manning_n_mean: float = 0.045                           # Mean channel Manning roughness over basin (-)
    channel_width_exponent_c: float = 0.5                   # Scaling parameter in channel width power law (-)
    channel_width_coeff_alpha: float = 10.0                 # Multiplicative parameter in channel width power law (-)
    mass_balance_tolerance: float = 1e-4                    # Mass balance tolerance for convergence check (-)

    # Atmospheric & Radiation Model Options
    clear_sky_atmos_emiss_model: str = "prata"              # Clear-sky atmospheric emissivity model
    cloudy_sky_atmos_emiss_model: str | None = "crawford"   # Cloudy-sky atmospheric emissivity model
    clear_sky_shortwave_model_name: str = "crawford"        # Clear-sky shortwave attenuation model
    precip_water_model_name: str = "prata"                  # Precipitable water model
    gamma_dust: float | None = None                         # Dust coefficient for Dingman model

    # # Output Control (old)
    # special_pixels: list[int] = field(default_factory=list) # Pixel indices for detailed hourly state/flux outputs
    # stream_pixels: list[int] = field(default_factory=list)  # Pixel indices for detailed hourly hydrograph outputs

    # Output Control (new, coordinate based, not linear index based) 
    special_pixels: Tuple[Sequence[int], Sequence[int]] = ((), ())
    stream_pixels: Tuple[Sequence[int], Sequence[int]] = ((), ())

    # Aggregated Basin Metadata (Populated during static map derivation)
    lat_mean: float | None = None                           # Mean latitude over basin (degrees)
    lon_mean: float | None = None                           # Mean longitude over basin (degrees)
    lambda_mean: float | None = None                        # Basin-averaged soil-topographic index (ln(m^2/m^2/h))
    basin_area: float | None = None                         # Total watershed surface area (m^2)

    @property
    def K0(self) -> float:
        """Surface saturated hydraulic conductivity (m/hour)."""
        return self.T0 / self.m  #

    @classmethod
    def load(cls, csv_path: Optional[str | Path] = None) -> "ModelParameters":
        """
        Instantiates ModelParameters using default values unless a CSV path is provided.
        Only variables present in the CSV will override the class defaults.
        """
        if csv_path is None:
            return cls()

        path = Path(csv_path)
        if path.exists():
            print(f"File '{csv_path}' will be used to override default model parameters.")

            field_map = {f.name: f for f in fields(cls)}
            overrides = {}

            with open(path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = row["key"].strip()
                    val_str = row["value"].strip()

                    if key not in field_map:
                        continue

                    # Parse empty/None entries
                    if val_str == "" or val_str.lower() == "none":
                        overrides[key] = None
                        continue

                    # Convert values according to field types
                    field_type = field_map[key].type
                    if field_type is str or "str" in str(field_type):
                        overrides[key] = val_str
                    else:
                        try:
                            overrides[key] = float(val_str)
                        except ValueError:
                            overrides[key] = val_str

            return cls(**overrides)
        else:
            print(f"Warning: File '{csv_path}' not found. Using default model parameters.")
            return cls()

@dataclass
class SpatialMaps:
    """Spatial 1D/2D grid maps, coordinate vectors, and derived terrain/soil properties."""

    # Coordinate-Related Data
    northing: np.ndarray | None = None                      # 1D vector of M northing values (m)
    easting: np.ndarray | None = None                       # 1D vector of N easting values (m)
    lat: np.ndarray | None = None                           # 2D latitude grid (degrees)
    lon: np.ndarray | None = None                           # 2D longitude grid (degrees)
    outlet_coordinate: np.ndarray | None = None             # Basin outlet coordinate [easting, northing] (m)

    # 2D Basin Spatial Data (all have shape: M x N)
    aspect_deg: np.ndarray | None = None                    # Terrain aspect orientation grid (degrees)
    aspect_rad: np.ndarray | None = None                    # Terrain aspect orientation grid (radians)
    elev: np.ndarray | None = None                          # Terrain elevation grid (m)
    flowacc: np.ndarray | None = None                       # Accumulated upstream flow area grid (m^2)
    mask: np.ndarray | None = None                          # Watershed binary mask array (0/1) (-)
    maskNaN: np.ndarray | None = None                       # Watershed mask array with NaNs outside domain (NaN/1) (-)
    slope_deg: np.ndarray | None = None                     # Terrain slope grid (degrees)
    slope_rad: np.ndarray | None = None                     # Terrain slope grid (radians)
    SVF: np.ndarray | None = None                           # Sky view factor grid (-)
    THETAs: np.ndarray | None = None                        # Spatially distributed soil porosity map (-)
    PSIs: np.ndarray | None = None                          # Spatially distributed saturated matric head map (m)
    b_BC: np.ndarray | None = None                          # Spatially distributed Brooks-Corey exponent map (-)
    THETAfc: np.ndarray | None = None                       # Volumetric soil moisture at field capacity grid (-)
    THETApwp: np.ndarray | None = None                      # Volumetric soil moisture at permanent wilting point grid (-)
    albedo: np.ndarray | None = None                        # Spatially distributed surface soil albedo map (-)
    emiss: np.ndarray | None = None                         # Spatially distributed surface soil emissivity map (-)
    snow_emiss: np.ndarray | None = None                    # Spatially distributed snow surface emissivity map (-)
    Srzmax: np.ndarray | None = None                        # Maximum root zone storage (m)
    Srzmin: np.ndarray | None = None                        # Minimum root zone storage (m)
    T0: np.ndarray | None = None                            # Surface transmissivity map (m^2/h)
    K0: np.ndarray | None = None                            # Saturated hydraulic conductivity map (m/h)
    lambda_map: np.ndarray | None = None                    # Soil-topographic index map (ln(m^2/m^2/h))
    outline_x: np.ndarray | None = None                     # Watershed outline easting coordinates (m)
    outline_y: np.ndarray | None = None                     # Watershed outline northing coordinates (m)

@dataclass
class NetworkTopology:
    """Sparse flow matrices, channel properties, and routing network connectivity."""

    flowdir: sp.coo_matrix | sp.csr_matrix | None = None    # Sparse matrix indicating flow directions (-)
    manning_n: np.ndarray | None = None                     # Channel Manning roughness coefficient grid (-)
    width: np.ndarray | None = None                         # Channel width grid (m)
    bed_slope: np.ndarray | None = None                     # Channel bed-slope grid (-)

    # # Old (MATLAB-based)
    # Iupstream: np.ndarray | None = None                     # Array of upstream pixel indices in DEM (-)
    # Idownstream: np.ndarray | None = None                   # Array of downstream pixel indices in DEM (-)
    # Ioutlet: int | np.ndarray | None = None                 # Index of basin outlet pixel in DEM (-)

    # New (Python-based)
    Iupstream: tuple[np.ndarray, np.ndarray] | None = None      # Tuple of (row, col) vectors of upstream pixels in DEM (-)
    Idownstream: tuple[np.ndarray, np.ndarray] | None = None    # Tuple of (row, col) vectors of downstream pixels in DEM (-)
    Ioutlet: tuple[int, int] | None = None                      # Tuple of (row, col) indices of basin outlet pixel in DEM (-)

@dataclass
class ShadeLookupTable:
    """Tabulated solar, shade, and ancillary lookup matrices."""

    shade_calc_flag: bool = False                           # Flag to perform shade lookup calculations
    shade_table: np.ndarray | None = None                   # Binary shade lookup matrix (-)
    azimuth: np.ndarray | None = None                       # Discrete solar azimuth angles (degrees)
    zenith: np.ndarray | None = None                        # Discrete solar zenith angles (degrees)


@dataclass
class MetForcingData:
    """Meteorological forcing timeseries and station attributes."""

    # Time index, 365*24*4=35040 values (15 min temporal resolution)
    time: np.ndarray | None = None                          # Meteorological forcing timestamp array (datetime stamp)
    elapsed_days: np.ndarray | None = None                  # Elapsed days (including fractional days) from start of met data (days)
    DOY: np.ndarray | None = None                           # day of year (including fractional days) from start of met data (DOY)
    start_date_time: str | None = None                      # start date time string in format yyyy-mm-dd HH:MM:SS
    gage_elev: float | None = None                          # Weather station elevation (m)

    # Meteorological Inputs (same shape as met_time)
    Ta: np.ndarray | None = None                            # Near-surface air temperature array (K)
    qa: np.ndarray | None = None                            # Specific humidity array (kg/kg)
    Psfc: np.ndarray | None = None                          # Surface atmospheric pressure array (Pa)
    U: np.ndarray | None = None                             # Wind speed array (m/s)
    SW: np.ndarray | None = None                            # Downward shortwave solar radiation array (W/m^2)
    PPT: np.ndarray | None = None                           # Precipitation rate array (mm/h)

    # Solar index (same shape as met_time)
    solar_index: np.ndarray | None = None

@dataclass
class ModelState:
    """Active physical states updated continuously and carried across timesteps (t -> t+1)."""

    nx: int
    ny: int

    # Hydrologic States (2D: nx x ny)
    Srz: np.ndarray = field(init=False)                     # Rootzone soil moisture storage (m)
    Suz: np.ndarray = field(init=False)                     # Unsaturated zone storage (m)
    SD: np.ndarray = field(init=False)                      # Saturation deficit (m)

    # Thermal States (2D: nx x ny)
    Tsurf: np.ndarray = field(init=False)                   # Surface temperature (K)
    Td: np.ndarray = field(init=False)                      # Deep soil temperature (K)

    # Snow States (2D: nx x ny)
    SWE: np.ndarray = field(init=False)                     # Snow water equivalent (m)
    snowdepth: np.ndarray = field(init=False)               # Snow depth (m)
    snowdens: np.ndarray = field(init=False)                # Snow density (kg/m^3)
    snowfrac: np.ndarray = field(init=False)                # Snow cover fraction (-)

    # Tracking / Memory Variables (2D: nx x ny)
    NDayLastSnow: np.ndarray = field(init=False)            # Days since last major snowfall (-)

    # Lateral Routing Flow Buffers (2D: nx x ny)
    INFLOW_old: np.ndarray = field(init=False)              # Previous timestep inflow rate (m^3/s)
    OUTFLOW_old: np.ndarray = field(init=False)             # Previous timestep outflow rate (m^3/s)
    NEW_INFLOWS: np.ndarray = field(init=False)             # Current timestep calculated inflow rate (m^3/s)

    def __post_init__(self) -> None:
        shape = (self.nx, self.ny)
        for name in self.__dataclass_fields__:
            if name not in ("nx", "ny"):
                setattr(self, name, np.zeros(shape, dtype=np.float64))

@dataclass
class StepVariables:
    """Transient 2D intermediate buffers and instantaneous fluxes computed during a single timestep."""

    nx: int
    ny: int

    # Surface Properties
    albedo_out: np.ndarray = field(init=False)              # Effective surface albedo (-)

    # Intermediate Updated States (2D: nx x ny)
    Tsurf_new: np.ndarray = field(init=False)               # Updated surface temperature (K)
    SWE_new: np.ndarray = field(init=False)                 # Updated snow water equivalent (m)
    Td_new: np.ndarray = field(init=False)                  # Updated deep soil temperature (K)
    snowdens_new: np.ndarray = field(init=False)            # Updated snow density (kg/m^3)
    snowdepth_new: np.ndarray = field(init=False)           # Updated snow depth (m)
    snowfrac_new: np.ndarray = field(init=False)            # Updated snow cover fraction (-)

    # NEW: snow melt output (2D: nx x ny)
    melt_out: np.ndarray = field(init=False)                # Instantaneous snowmelt rate (m/h)

    # NEW: infiltration model output (2D: nx x ny)
    qie_new: np.ndarray = field(init=False)                 # Infiltration excess runoff (in cm)
    
    # NEW: topmodel output (2D: nx x ny)
    Srz_new: np.ndarray = field(init=False)                 # Rootzone storage at end of time step (m)
    Suz_new: np.ndarray = field(init=False)                 # Unsaturated zone storage at end of time step (m)
    SD_new: np.ndarray = field(init=False)                  # Storage deficit at end of time step (m)
    qv_new: np.ndarray = field(init=False)                  # Recharge flux (m)
    qb_new: np.ndarray = field(init=False)                  # Baseflow (m^2/hr)
    qse_new: np.ndarray = field(init=False)                 # Saturation excess runoff (m)
    Qv: float = field(init=False)                           # Total basin-averaged (recharge) flux to groundwater (m)
    Qb: float = field(init=False)                           # Total basin-averaged baseflow (m)

    # Instantaneous Energy Fluxes (2D: nx x ny)
    Rn_out: np.ndarray = field(init=False)                  # Net radiation (W/m^2)
    LE_out: np.ndarray = field(init=False)                  # Latent heat flux (W/m^2)
    H_out: np.ndarray = field(init=False)                   # Sensible heat flux (W/m^2)
    G_out: np.ndarray = field(init=False)                   # Ground heat flux (W/m^2)
    Rlup_out: np.ndarray = field(init=False)                # Upwelling longwave radiation (W/m^2)

    # NEW: Energy Fluxes (2D: nx x ny)
    ET_out: np.ndarray = field(init=False)                  # evaporative flux (m/hr)

    def __post_init__(self) -> None:
        shape = (self.nx, self.ny)
        for name in self.__dataclass_fields__:
            if name not in ("nx", "ny", "Qv", "Qb"):
                setattr(self, name, np.zeros(shape, dtype=np.float64))
            elif name in ("Qv", "Qb"):
                setattr(self, name, 0.0)

@dataclass
class Accumulators:
    """Transient 2D running tallies reset daily to calculate time-averaged or accumulated outputs."""

    nx: int
    ny: int

    # State Accumulators (2D: nx x ny)
    cumulSrz: np.ndarray = field(init=False)                # Accumulated rootzone soil moisture (m)
    cumulSuz: np.ndarray = field(init=False)                # Accumulated unsaturated zone storage (m)
    cumulSD: np.ndarray = field(init=False)                 # Accumulated saturation deficit (m)
    cumulTsurf: np.ndarray = field(init=False)              # Accumulated surface temperature (K)
    cumulSWE: np.ndarray = field(init=False)                # Accumulated snow water equivalent (m)
    cumulsnowdepth: np.ndarray = field(init=False)          # Accumulated snow depth (m)
    cumulsnowdens: np.ndarray = field(init=False)           # Accumulated snow density (kg/m^3)
    cumulsnowfrac: np.ndarray = field(init=False)           # Accumulated snow cover fraction (-)
    cumulTd: np.ndarray = field(init=False)                 # Accumulated deep soil temperature (K)

    # Flux Accumulators (2D: nx x ny)
    cumulsnowmelt: np.ndarray = field(init=False)           # Accumulated snowmelt (m)
    cumulRn: np.ndarray = field(init=False)                 # Accumulated net radiation (W/m^2)
    cumulLE: np.ndarray = field(init=False)                 # Accumulated latent heat flux (W/m^2)
    cumulET: np.ndarray = field(init=False)                 # Accumulated evapotranspiration (m)
    cumulH: np.ndarray = field(init=False)                  # Accumulated sensible heat flux (W/m^2)
    cumulqie: np.ndarray = field(init=False)                # Accumulated infiltration excess runoff (m)
    cumulqse: np.ndarray = field(init=False)                # Accumulated saturation excess runoff (m)
    cumulqb: np.ndarray = field(init=False)                 # Accumulated baseflow (m)
    cumulqv: np.ndarray = field(init=False)                 # Accumulated recharge to saturated zone (m)
    cumulinfil: np.ndarray = field(init=False)              # Accumulated infiltration (m)
    cumulRlup: np.ndarray = field(init=False)               # Accumulated upwelling longwave radiation (W/m^2)

    # Disaggregated Forcing Accumulators (2D: nx x ny)
    cumulTair: np.ndarray = field(init=False)               # Accumulated air temperature (K)
    cumulalbedo: np.ndarray = field(init=False)             # Accumulated surface albedo (-)
    cumulRldown: np.ndarray = field(init=False)             # Accumulated downwelling longwave radiation (W/m^2)
    cumulRs: np.ndarray = field(init=False)                 # Accumulated solar radiation (W/m^2)
    cumulqair: np.ndarray = field(init=False)               # Accumulated specific humidity (kg/kg)
    cumulPsfc: np.ndarray = field(init=False)               # Accumulated surface pressure (Pa)
    cumulPPT: np.ndarray = field(init=False)                # Accumulated precipitation (m)

    def __post_init__(self) -> None:
        shape = (self.nx, self.ny)
        for name in self.__dataclass_fields__:
            if name not in ("nx", "ny"):
                setattr(self, name, np.zeros(shape, dtype=np.float64))

    def reset(self) -> None:
        """Reset all accumulators to zero at the start of a new daily loop."""
        for name in self.__dataclass_fields__:
            if name not in ("nx", "ny"):
                getattr(self, name).fill(0.0)

@dataclass
class MapOutputs:
    """Historical daily 3D spatial grids (n_map, nx, ny) stored for analysis and export."""

    n_map: int
    nx: int
    ny: int

    # State Daily Maps (3D: n_map x nx x ny)
    Srz: np.ndarray = field(init=False)                     # Daily rootzone soil moisture (m)
    Suz: np.ndarray = field(init=False)                     # Daily unsaturated zone storage (m)
    SD: np.ndarray = field(init=False)                      # Daily saturation deficit (m)
    Tsurf: np.ndarray = field(init=False)                   # Daily surface temperature (K)
    SWE: np.ndarray = field(init=False)                     # Daily snow water equivalent (m)
    snowdepth: np.ndarray = field(init=False)               # Daily snow depth (m)
    snowdens: np.ndarray = field(init=False)                # Daily snow density (kg/m^3)
    snowfrac: np.ndarray = field(init=False)                # Daily snow cover fraction (-)
    Td: np.ndarray = field(init=False)                      # Daily deep soil temperature (K)

    # Flux Daily Maps (3D: n_map x nx x ny)
    snowmelt: np.ndarray = field(init=False)                # Daily snowmelt (m)
    Rn: np.ndarray = field(init=False)                      # Daily net radiation (W/m^2)
    LE: np.ndarray = field(init=False)                      # Daily latent heat flux (W/m^2)
    ET: np.ndarray = field(init=False)                      # Daily evapotranspiration (m/h)
    H: np.ndarray = field(init=False)                       # Daily sensible heat flux (W/m^2)
    qie: np.ndarray = field(init=False)                     # Daily infiltration excess runoff (m)
    qse: np.ndarray = field(init=False)                     # Daily saturation excess runoff (m)
    qb: np.ndarray = field(init=False)                      # Daily baseflow (m)
    qv: np.ndarray = field(init=False)                      # Daily recharge to saturated zone (m)
    infil: np.ndarray = field(init=False)                   # Daily infiltration (m)
    Rlup: np.ndarray = field(init=False)                    # Daily upwelling longwave radiation (W/m^2)

    # Disaggregated Forcing Daily Maps (3D: n_map x nx x ny)
    Tair: np.ndarray = field(init=False)                    # Daily air temperature (K)
    albedo: np.ndarray = field(init=False)                  # Daily surface albedo (-)
    Rldown: np.ndarray = field(init=False)                  # Daily downwelling longwave radiation (W/m^2)
    Rs: np.ndarray = field(init=False)                      # Daily solar radiation (W/m^2)
    qair: np.ndarray = field(init=False)                    # Daily specific humidity (kg/kg)
    Psfc: np.ndarray = field(init=False)                    # Daily surface pressure (Pa)
    PPT: np.ndarray = field(init=False)                     # Daily precipitation (m)

    # NEW: Tracking / Memory Variables (2D: nx x ny)
    NDayLastSnow: np.ndarray = field(init=False)            # Days since last major snowfall (-)

    def __post_init__(self) -> None:
        shape = (self.n_map, self.nx, self.ny)
        for name in self.__dataclass_fields__:
            if name not in ("n_map", "nx", "ny"):
                setattr(self, name, np.full(shape, np.nan, dtype=np.float64))

@dataclass
class TimeSeriesOutputs:
    """Historical time-series vectors (1D) and point observations (2D) recorded at sub-daily timesteps."""

    nt: int
    nx: int
    ny: int
    special_pixels: Tuple[Sequence[int], Sequence[int]] = ((), ())
    stream_pixels: Tuple[Sequence[int], Sequence[int]] = ((), ())

    # Derived Pixel Counts
    n_special_pixels: int = field(init=False)
    n_stream_pixels: int = field(init=False)

    # Basin-Averaged States (1D: nt + 1)
    Srz: np.ndarray = field(init=False)                     # Basin-avg rootzone soil moisture (m)
    Suz: np.ndarray = field(init=False)                     # Basin-avg unsaturated storage (m)
    SD: np.ndarray = field(init=False)                      # Basin-avg saturation deficit (m)
    Tsurf: np.ndarray = field(init=False)                   # Basin-avg surface temperature (K)
    SWE: np.ndarray = field(init=False)                     # Basin-avg snow water equivalent (m)
    snowdepth: np.ndarray = field(init=False)               # Basin-avg snow depth (m)
    snowdens: np.ndarray = field(init=False)                # Basin-avg snow density (kg/m^3)
    snowfrac: np.ndarray = field(init=False)                # Basin-avg snow cover fraction (-)
    Td: np.ndarray = field(init=False)                      # Basin-avg deep soil temperature (K)

    # Basin-Averaged Fluxes (1D: nt)
    snowmelt: np.ndarray = field(init=False)                # Basin-avg snowmelt (m/h)
    Rn: np.ndarray = field(init=False)                      # Basin-avg net radiation (W/m^2)
    LE: np.ndarray = field(init=False)                      # Basin-avg latent heat flux (W/m^2)
    ET: np.ndarray = field(init=False)                      # Basin-avg evapotranspiration (m/h)
    H: np.ndarray = field(init=False)                       # Basin-avg sensible heat flux (W/m^2)
    qie: np.ndarray = field(init=False)                     # Basin-avg infiltration excess runoff (m/h)
    qse: np.ndarray = field(init=False)                     # Basin-avg saturation excess runoff (m/h)
    qb: np.ndarray = field(init=False)                      # Basin-avg baseflow (m/h)
    qv: np.ndarray = field(init=False)                      # Basin-avg recharge to saturated zone (m/h)
    outlet_hydrograph: np.ndarray = field(init=False)       # Outlet hydrograph (m^3/s)
    Rlup: np.ndarray = field(init=False)                    # Basin-avg upwelling longwave radiation (W/m^2)
    # NEW
    infil: np.ndarray = field(init=False)                   # Basin-ave. infiltration (m/h)

    # Basin-Averaged Disaggregated Forcings (1D: nt)
    Rs: np.ndarray = field(init=False)                      # Basin-avg solar radiation (W/m^2)
    Tair: np.ndarray = field(init=False)                    # Basin-avg air temperature (K)
    albedo: np.ndarray = field(init=False)                  # Basin-avg surface albedo (-)
    qair: np.ndarray = field(init=False)                    # Basin-avg specific humidity (kg/kg)
    Psfc: np.ndarray = field(init=False)                    # Basin-avg surface pressure (Pa)
    Rldown: np.ndarray = field(init=False)                  # Basin-avg downwelling longwave radiation (W/m^2)
    PPT: np.ndarray = field(init=False)                     # Basin-avg precipitation (m)

    # Pixel storage structure needs to be corrected, it's time x num_pixels, not nx x ny
    # Stream Pixel Outputs (3D: nt x nx x ny)
    pixel_stream_hydrograph: np.ndarray = field(init=False) # Stream pixel hydrographs (m^3/s)

    # Special Pixel States (3D: (nt+1) x nx x ny)
    pixel_Srz: np.ndarray = field(init=False)               # Special-pixel rootzone moisture (m)
    pixel_Suz: np.ndarray = field(init=False)               # Special-pixel unsaturated storage (m)
    pixel_SD: np.ndarray = field(init=False)                # Special-pixel saturation deficit (m)
    pixel_Tsurf: np.ndarray = field(init=False)             # Special-pixel surface temperature (K)
    pixel_SWE: np.ndarray = field(init=False)               # Special-pixel snow water equivalent (m)
    pixel_snowdepth: np.ndarray = field(init=False)         # Special-pixel snow depth (m)
    pixel_snowdens: np.ndarray = field(init=False)          # Special-pixel snow density (kg/m^3)
    pixel_snowfrac: np.ndarray = field(init=False)          # Special-pixel snow cover fraction (-)
    pixel_Td: np.ndarray = field(init=False)                # Special-pixel deep soil temperature (K)

    # Special Pixel Fluxes (3D: nt x nx x ny)
    pixel_snowmelt: np.ndarray = field(init=False)          # Special-pixel snowmelt (m/h)
    pixel_Rn: np.ndarray = field(init=False)                # Special-pixel net radiation (W/m^2)
    pixel_LE: np.ndarray = field(init=False)                # Special-pixel latent heat flux (W/m^2)
    pixel_ET: np.ndarray = field(init=False)                # Special-pixel evapotranspiration (m/h)
    pixel_H: np.ndarray = field(init=False)                 # Special-pixel sensible heat flux (W/m^2)
    pixel_qie: np.ndarray = field(init=False)               # Special-pixel infiltration excess runoff (m/h)
    pixel_qse: np.ndarray = field(init=False)               # Special-pixel saturation excess runoff (m/h)
    pixel_qb: np.ndarray = field(init=False)                # Special-pixel baseflow (m/h)
    pixel_qv: np.ndarray = field(init=False)                # Special-pixel recharge (m/h)
    pixel_Rlup: np.ndarray = field(init=False)              # Special-pixel upwelling longwave radiation (W/m^2)
    pixel_infil: np.ndarray = field(init=False)             # Special-pixel infiltration (m/h)

    def __post_init__(self) -> None:
        # Compute total count of selected pixel coordinates
        self.n_special_pixels = len(self.special_pixels[0]) if self.special_pixels else 0
        self.n_stream_pixels = len(self.stream_pixels[0]) if self.stream_pixels else 0

        # 1D Basin States (nt + 1)
        basin_states = ("Srz", "Suz", "SD", "Tsurf", "SWE", "snowdepth", "snowdens", "snowfrac", "Td")
        for field_name in basin_states:
            setattr(self, field_name, np.zeros(self.nt + 1, dtype=np.float64))

        # 1D Basin Fluxes & Forcings (nt)
        basin_step_fields = (
            "snowmelt", "Rn", "LE", "ET", "H", "qie", "qse", "qb", "qv", "outlet_hydrograph", "Rlup", "infil",
            "Rs", "Tair", "albedo", "qair", "Psfc", "Rldown", "PPT"
        )
        for field_name in basin_step_fields:
            setattr(self, field_name, np.zeros(self.nt, dtype=np.float64))

        # Stream Pixels (nt x nx x ny)
        self.pixel_stream_hydrograph = np.zeros((self.nt, self.nx, self.ny), dtype=np.float64)

        # Special Pixel States ((nt+1) x nx x ny)
        pixel_state_fields = (
            "pixel_Srz", "pixel_Suz", "pixel_SD", "pixel_Tsurf", "pixel_SWE",
            "pixel_snowdepth", "pixel_snowdens", "pixel_snowfrac", "pixel_Td"
        )
        for field_name in pixel_state_fields:
            setattr(self, field_name, np.zeros((self.nt + 1, self.nx, self.ny), dtype=np.float64))

        # Special Pixel Fluxes (nt x nx x ny)
        pixel_flux_fields = (
            "pixel_snowmelt", "pixel_Rn", "pixel_LE", "pixel_ET", "pixel_H",
            "pixel_qie", "pixel_qse", "pixel_qb", "pixel_qv", "pixel_Rlup", "pixel_infil"
        )
        for field_name in pixel_flux_fields:
            setattr(self, field_name, np.zeros((self.nt, self.nx, self.ny), dtype=np.float64))

