# import base libraries
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import xarray as xr
import pandas as pd
import calendar
import matplotlib.pyplot as plt

from src.plot_utils.plot_functions import (
    get_day_of_year_flexible, 
    montly_mean, 
    plot_format, 
    plot_spatial_data, 
    plot_time_series, 
    time_series_aggregation
)

# Similar to evaluate_mod_wet_simulation.m from original MATLAB MOD-WET
def evaluate_simulation(output_filepath: str | Path, plots_path: str | Path,
                        save_plots: bool = True, display_plots: bool = False,
                        file_name_prefix: Optional[str] = None):
    """
    Evaluate simulation results and develop key results plots.

    Parameters
    ----------
    output_filepath : str | Path
        File path of NetCDF simulation results.
    plots_path: 
        location to save simulation results plots
    save_plots: 
        boolean, if True, develop and save plots to plot_path
    display_plots: 
        boolean, if True, displays plots to screen (develop_plots must be set to True)
    """
    # Create directory if it doesn't exist
    if isinstance(plots_path, (str, Path)):
        plots_path = Path(plots_path)
        plots_path.mkdir(parents=True, exist_ok=True)

    if file_name_prefix is None:
        file_name_prefix = ""
    else:
        file_name_prefix = file_name_prefix + "_"

    # read in simulation output and static input data
    ds = xr.open_dataset(output_filepath, decode_timedelta=False)

    # spatial data
    northing = ds['northing'].values
    easting = ds['easting'].values
    mask = ds['static_map_mask'].values
    maskNaN = np.where(mask == 1, 1.0, np.nan)

    # temporal data
    time_flux = ds['time_fluxes'].values
    time_state = ds['time_states'].values
    time_map = ds['time_maps'].values
    water_year = time_map[-1].astype('datetime64[Y]').astype(int) + 1970

    # extract static data
    dx = ds.dx                                      # m
    dy = ds.dy                                      # m
    basin_area = ds.basin_area_m2;                  # m^2
    basin_area_km2 = basin_area / 1000 / 1000;      # km^2

    # Generate Figures

    # Figure 1: Elevation Map
    elev = ds['static_map_elev'].values * maskNaN
    fig1, ax1 = plt.subplots(num=1) # define a new figure
    plot_spatial_data(fig1, ax1, easting, northing, elev, 
                      "Easting (m)", "Northing (m)", None, 
                      plot_cmap="terrain", sci_not=True,
                      plot_title="Elevation (m)")

    # Figure 2: Elevation Histogram
    fig2, ax2 = plt.subplots(num=2)
    hist_data = (elev * maskNaN).flat
    bin_width = 50
    bin_min = round(np.nanmin(hist_data) - bin_width, -2)
    bin_max = round(np.nanmax(hist_data) + bin_width, -2)
    ax2.hist(hist_data, color='skyblue', edgecolor='black', 
             bins=np.arange(bin_min, bin_max + bin_width, bin_width))
    plot_format(ax2, "Elevation (m)", "Count (# of Pixels)", grid=False)
    ax2.set_title("Watershed Elevation Histogram", fontsize=12)

    # Figure 3: Soil Saturated Hydraulic Conductivity
    K0_map = ds['static_map_K0'].values * maskNaN
    fig3, ax3 = plt.subplots(num=3) # define a new figure
    plot_spatial_data(fig3, ax3, easting, northing, K0_map, 
                      "Easting (m)", "Northing (m)", None, 
                      plot_cmap="viridis", sci_not=True,
                      plot_title="Soil Saturated Hydraulic Conductivity (m/h)")

    # Figure 4: Air Temperature
    Tair = ds['basin_Tair'].values - 273.15 # convert to Celsius
    time_daily, temp_daily = time_series_aggregation(time_vector=time_flux, data_vector=Tair, agg_func="mean", output_freq="D")
    fig4, ax4 = plt.subplots(num=4) # define a new figure
    plot_time_series(fig4, ax4, time_daily, [temp_daily], x_lab=None, y_lab=r"Average Daily Air Temperature ($^{\circ}$C)")

    # Figure 5: Daily Precip
    PPT = ds['basin_PPT'].values
    time_daily, precip_daily = time_series_aggregation(time_vector=time_flux, data_vector=PPT, agg_func="mean", output_freq="D")
    precip_daily = precip_daily * 24 * 1000 # convert to mm
    fig5, ax5 = plt.subplots(num=5) # define a new figure
    plot_time_series(fig5, ax5, time_daily, [precip_daily], x_lab=None, y_lab="Total Daily Precip (mm)")

    # Figure 6: Annual Average Temperature Map
    map_Tair = np.nanmean(ds['map_Tair'].values, axis=0) - 273.15
    fig6, ax6 = plt.subplots(num=6) # define a new figure
    plot_spatial_data(fig6, ax6, easting, northing, map_Tair, 
                      "Easting (m)", "Northing (m)", None, 
                      plot_cmap="RdYlBu_r", sci_not=True,
                      plot_title=r"Annual Average Air Temperature ($^{\circ}$C)")

    # Figure 7: Daily Albedo
    albedo = ds['basin_albedo'].values
    time_daily, albedo_daily = time_series_aggregation(time_vector=time_flux, data_vector=albedo, agg_func="mean", output_freq="D")
    fig7, ax7 = plt.subplots(num=7) # define a new figure
    plot_time_series(fig7, ax7, time_daily, [albedo_daily], x_lab=None, y_lab="Average Daily Albedo (-)")

    # Figure 8: Shorwave Radiation
    # extract data and aggregate from hourly to monthly
    time_monthly, Rs_monthly = montly_mean(time_flux, ds["basin_Rs"].values)
    fig8, ax8 = plt.subplots(num=8) # define a new figure
    plot_time_series(fig8, ax8, time_monthly, [Rs_monthly], x_lab=None, y_lab=r"Shortwave Radiation (W/m$^2$)")

    # Figure 9: Downwelling Longwave Radiation
    # extract data and aggregate from hourly to monthly
    time_monthly, Rldown_monthly = montly_mean(time_flux, ds["basin_Rldown"].values)
    fig9, ax9 = plt.subplots(num=9) # define a new figure
    plot_time_series(fig9, ax9, time_monthly, [Rldown_monthly], x_lab=None, y_lab=r"Downwelling Longwave Radiation (W/m$^2$)")

    # Figure 10: Net Radiation
    # extract data and aggregate from hourly to monthly
    time_monthly, Rn_monthly = montly_mean(time_flux, ds["basin_Rn"].values)
    fig10, ax10 = plt.subplots(num=10) # define a new figure
    plot_time_series(fig10, ax10, time_monthly, [Rn_monthly], x_lab=None, y_lab=r"Net Radiation (W/m$^2$)")

    # Figure 11: Annual Snowfall Map
    map_snowfall = ds['map_PPT'].values
    map_snowfall[map_snowfall>273.15] = 0 # remove rainfall
    map_snowfall = map_snowfall * 1000 # convert m to mm
    map_snowfall = np.nansum(map_snowfall, axis = 0) * maskNaN # take annual total
    fig11, ax11 = plt.subplots(num=11) # define a new figure
    plot_spatial_data(fig11, ax11, easting, northing, map_snowfall, 
                      "Easting (m)", "Northing (m)", None, 
                      plot_cmap="cool_r", sci_not=True,
                      plot_title=r"Annual Total Snowfall (mm)")

    # Figure 12: Daily SWE
    SWE = ds['basin_SWE'].values
    time_daily, swe_daily = time_series_aggregation(time_vector=time_state, data_vector=SWE, agg_func="mean", output_freq="D")
    fig12, ax12 = plt.subplots(num=12) # define a new figure
    plot_time_series(fig12, ax12, time_daily, [swe_daily], x_lab=None, y_lab="Basin-Averaged Daily SWE (m)")

    # Figure 13: SWE Map on April 1st
    target_date = np.datetime64(f"{water_year}-04-01 00:00:00")
    target_date_idx = np.where(time_map == target_date)[0]
    if len(target_date_idx) != 0:
        dowy_f13 = get_day_of_year_flexible(target_date, start_month=10)
        py_dt = target_date.astype(object)
        year = py_dt.year
        month = calendar.month_abbr[py_dt.month]
        day = py_dt.day
        map_SWE = ds['map_SWE'].values[target_date_idx[0]] * maskNaN
        fig13, ax13 = plt.subplots(num=13) # define a new figure
        plot_spatial_data(fig13, ax13, easting, northing, map_SWE, 
                        "Easting (m)", "Northing (m)", None, 
                        plot_cmap="cool_r", sci_not=True,
                        plot_title=f"SWE (m) on {year}-{month}-{day} (DOWY {dowy_f13})")
    else:
        print("April 1st not found. Check the meteorological forcing data.")

    # Soil Root Zone Moisture Maps

    # Figure 14: DOWY 250 Srz map
    dowy_f14 = 250
    target_date_idx = dowy_f14 - 1
    if len(time_map) - 1 >= target_date_idx:
        target_date = time_map[target_date_idx]
        py_dt = target_date.astype("M8[D]").astype(object)
        year = py_dt.year
        month = calendar.month_abbr[py_dt.month]
        day = py_dt.day
        map_Srz = ds['map_Srz'].values[target_date_idx] * maskNaN
        fig14, ax14 = plt.subplots(num=14) # define a new figure
        plot_spatial_data(fig14, ax14, easting, northing, map_Srz, 
                        "Easting (m)", "Northing (m)", None, 
                        plot_cmap="viridis", sci_not=True,
                        plot_title=f"Soil Root Zone Moisture (m) on {year}-{month}-{day} (DOWY {dowy_f14})")
    else:
        print(f"DOWY {dowy_f14} not found. Check the meteorological forcing data.")

    # Figure 15: DOWY 275 Srz map
    dowy_f15 = 275
    target_date_idx = dowy_f15 - 1
    if len(time_map) - 1 >= target_date_idx:
        target_date = time_map[target_date_idx]
        py_dt = target_date.astype("M8[D]").astype(object)
        year = py_dt.year
        month = calendar.month_abbr[py_dt.month]
        day = py_dt.day
        map_Srz = ds['map_Srz'].values[target_date_idx] * maskNaN
        fig15, ax15 = plt.subplots(num=15) # define a new figure
        plot_spatial_data(fig15, ax15, easting, northing, map_Srz, 
                        "Easting (m)", "Northing (m)", None, 
                        plot_cmap="viridis", sci_not=True,
                        plot_title=f"Soil Root Zone Moisture (m) on {year}-{month}-{day} (DOWY {dowy_f15})")
    else:
        print(f"DOWY {dowy_f15} not found. Check the meteorological forcing data.")

    # Figure 16: DOWY 300 Srz map
    dowy_f16 = 300
    target_date_idx = dowy_f16 - 1
    if len(time_map) - 1 >= target_date_idx:
        target_date = time_map[target_date_idx]
        py_dt = target_date.astype("M8[D]").astype(object)
        year = py_dt.year
        month = calendar.month_abbr[py_dt.month]
        day = py_dt.day
        map_Srz = ds['map_Srz'].values[target_date_idx] * maskNaN
        fig16, ax16 = plt.subplots(num=16) # define a new figure
        plot_spatial_data(fig16, ax16, easting, northing, map_Srz, 
                        "Easting (m)", "Northing (m)", None, 
                        plot_cmap="viridis", sci_not=True,
                        plot_title=f"Soil Root Zone Moisture (m) on {year}-{month}-{day} (DOWY {dowy_f16})")
    else:
        print(f"DOWY {dowy_f16} not found. Check the meteorological forcing data.")

    # Figure 17: Surface Energy Balance
    G = ds["basin_Rn"].values - ds["basin_LE"].values - ds["basin_H"].values
    time_monthly, Rn_monthly = montly_mean(time_flux, ds["basin_Rn"].values)
    _, LE_monthly            = montly_mean(time_flux, ds["basin_LE"].values)
    _, H_monthly             = montly_mean(time_flux, ds["basin_H"].values)
    _, G_monthly             = montly_mean(time_flux, G)
    fig17, ax17 = plt.subplots(num=17) # define a new figure
    plot_time_series(fig17, ax17, time_monthly, [Rn_monthly, LE_monthly, H_monthly, G_monthly], 
                     x_lab=None, y_lab=r"Radiation (W/m$^2$)", 
                     data_labels=["Net Radiation", "Latent Heat", "Sensible Heat", "Ground Heat"],
                     data_colors=["b", "r", "g", "k"])

    # Figure 18: Water Balance
    
    # Convert water fluxes to daily cumulative totals in cubic meters
    # Cumulative Precipitation
    PPT = ds['basin_PPT'].values # m/hr
    time_daily, precip_daily = time_series_aggregation(time_vector=time_flux, data_vector=PPT, agg_func="mean", output_freq="D")
    precip_daily = (precip_daily * 24) * basin_area # convert to daily total in m^3 
    precip_daily_cumulative = np.cumsum(precip_daily)

    # Cumulative Runoff
    Q = ds['basin_outlet_hydrograph'].values * 3600 # m^3/s to m^3/h
    _, Q_daily = time_series_aggregation(time_vector=time_flux, data_vector=Q, agg_func="mean", output_freq="D")
    Q_daily = (Q_daily * 24) # convert to daily total in m^3 
    Q_daily_cumulative = np.cumsum(Q_daily)

    # Cumulative ET
    ET = ds['basin_ET'].values # m/hr
    _, ET_daily = time_series_aggregation(time_vector=time_flux, data_vector=ET, agg_func="mean", output_freq="D")
    ET_daily = (ET_daily * 24) * basin_area # convert to daily total in m^3 
    ET_daily_cumulative = np.cumsum(ET_daily)

    # Overall basin storage change (diagnosed from fluxes)
    storage_change=precip_daily_cumulative-ET_daily_cumulative-Q_daily_cumulative # in m^3

    # Plot Results
    fig18, ax18 = plt.subplots(num=18) # define a new figure
    plot_time_series(fig18, ax18, time_daily, [precip_daily_cumulative/1000**3, Q_daily_cumulative/1000**3, 
                                               ET_daily_cumulative/1000**3, storage_change/1000**3], 
                     x_lab=None, y_lab=r"Cumulative Volume (km$^3$)", sci_not=True,
                     data_labels=["Precipitation", "Runoff at Outlet", "Evaporation", "Storage Change"],
                     data_colors=["b", "r", "g", "k"], plot_title="Annual Cumulative Water Balance")

    # Print Key Values to terminal
    print(f"Cumulative precipitation over the course of the water year is: {precip_daily_cumulative[-1] / 1000**3} km^3")
    print(f"Cumulative outlet runoff over the course of the water year is: {Q_daily_cumulative[-1] / 1000**3} km^3")
    print(f"Cumulative evaporation over the course of the water year is: {ET_daily_cumulative[-1] / 1000**3} km^3")
    print(f"Storage change over the course of the water year is: { (storage_change[-1] - storage_change[0]) / 1000**3} km^3")

    # Figure 19: Saturation Excess Runoff
    qse = ds['basin_qse'].values
    time_daily, qse_daily = time_series_aggregation(time_vector=time_flux, data_vector=qse, agg_func="mean", output_freq="D")
    qse_daily = qse_daily * 1000 # convert m to mm
    fig19, ax19 = plt.subplots(num=19) # define a new figure
    plot_time_series(fig19, ax19, time_daily, [qse_daily], x_lab=None, y_lab="Basin-Averaged Daily Saturation Excess Runoff (mm/day)")

    # Figure 20: Infiltration Excess Runoff
    qie = ds['basin_qie'].values
    time_daily, qie_daily = time_series_aggregation(time_vector=time_flux, data_vector=qie, agg_func="mean", output_freq="D")
    qie_daily = qie_daily * 1000 # convert m to mm
    fig20, ax20 = plt.subplots(num=20) # define a new figure
    plot_time_series(fig20, ax20, time_daily, [qie_daily], x_lab=None, y_lab="Basin-Averaged Daily Infiltration Excess Runoff (mm/day)")

    # Figure 21: Baseflow Runoff
    qb = ds['basin_qb'].values
    time_daily, qb_daily = time_series_aggregation(time_vector=time_flux, data_vector=qb, agg_func="mean", output_freq="D")
    qb_daily = qb_daily * 1000 # convert m to mm
    fig21, ax21 = plt.subplots(num=21) # define a new figure
    plot_time_series(fig21, ax21, time_daily, [qb_daily], x_lab=None, y_lab="Basin-Averaged Daily Baseflow Runoff (mm/day)")

    # Figure 22: Annual Average Saturation Excess Runoff
    map_qse = np.nanmean(ds['map_qse'].values, axis=0) * 1000 # convert m to mm
    fig22, ax22 = plt.subplots(num=22) # define a new figure
    plot_spatial_data(fig22, ax22, easting, northing, map_qse, 
                      "Easting (m)", "Northing (m)", None, 
                      plot_cmap="viridis", sci_not=True,
                      plot_title="Annual Average Saturation Excess Runoff (mm/day)")

    # Figure 23: Annual Average Saturation Excess Runoff
    map_qie = np.nanmean(ds['map_qie'].values, axis=0) * 1000 # convert m to mm
    fig23, ax23 = plt.subplots(num=23) # define a new figure
    plot_spatial_data(fig23, ax23, easting, northing, map_qie, 
                      "Easting (m)", "Northing (m)", None, 
                      plot_cmap="viridis", sci_not=True,
                      plot_title="Annual Average Infiltration Excess Runoff (mm/day)")

    # Figure 24: Outlet Hydrograph
    Q = ds['basin_outlet_hydrograph'].values
    time_daily, Q_daily = time_series_aggregation(time_vector=time_flux, data_vector=Q, agg_func="mean", output_freq="D")
    fig24, ax24 = plt.subplots(num=24) # define a new figure
    plot_time_series(fig24, ax24, time_daily, [Q_daily], x_lab=None, y_lab=r"Streamflow (m$^3$/s)", plot_title="Outlet Hydrograph")

    if save_plots:
        fig1.savefig(plots_path / f"{file_name_prefix}Elevation_Map.png", dpi=300,  bbox_inches="tight")
        fig2.savefig(plots_path / f"{file_name_prefix}Elevation_Histogram.png", dpi=300,  bbox_inches="tight")
        fig3.savefig(plots_path / f"{file_name_prefix}K0_Map.png", dpi=300,  bbox_inches="tight")
        fig4.savefig(plots_path / f"{file_name_prefix}Daily_Air_Temp_Plot.png", dpi=300,  bbox_inches="tight")
        fig5.savefig(plots_path / f"{file_name_prefix}Daily_Precip_Plot.png", dpi=300,  bbox_inches="tight")
        fig6.savefig(plots_path / f"{file_name_prefix}Annual_Air_Temp_Map.png", dpi=300,  bbox_inches="tight")
        fig7.savefig(plots_path / f"{file_name_prefix}Daily_Albedo_Plot.png", dpi=300,  bbox_inches="tight")
        fig8.savefig(plots_path / f"{file_name_prefix}Monthly_Rs_Plot.png", dpi=300,  bbox_inches="tight")
        fig9.savefig(plots_path / f"{file_name_prefix}Monthly_Rldown_Plot.png", dpi=300,  bbox_inches="tight")
        fig10.savefig(plots_path / f"{file_name_prefix}Monthly_Rn_Plot.png", dpi=300,  bbox_inches="tight")
        fig11.savefig(plots_path / f"{file_name_prefix}Annual_Snowfall_Map.png", dpi=300,  bbox_inches="tight")
        fig12.savefig(plots_path / f"{file_name_prefix}Daily_SWE_Plot.png", dpi=300,  bbox_inches="tight")
        fig13.savefig(plots_path / f"{file_name_prefix}SWE_Map_on_DOWY_{dowy_f13}.png", dpi=300,  bbox_inches="tight")
        fig14.savefig(plots_path / f"{file_name_prefix}Srz_Map_on_DOWY_{dowy_f14}.png", dpi=300,  bbox_inches="tight")
        fig15.savefig(plots_path / f"{file_name_prefix}Srz_Map_on_DOWY_{dowy_f15}.png", dpi=300,  bbox_inches="tight")
        fig16.savefig(plots_path / f"{file_name_prefix}Srz_Map_on_DOWY_{dowy_f16}.png", dpi=300,  bbox_inches="tight")
        fig17.savefig(plots_path / f"{file_name_prefix}Surface_Energy_Balance.png", dpi=300,  bbox_inches="tight")
        fig18.savefig(plots_path / f"{file_name_prefix}Water_Balance.png", dpi=300,  bbox_inches="tight")
        fig19.savefig(plots_path / f"{file_name_prefix}Daily_qse_Plot.png", dpi=300,  bbox_inches="tight")
        fig20.savefig(plots_path / f"{file_name_prefix}Daily_qie_Plot.png", dpi=300,  bbox_inches="tight")
        fig21.savefig(plots_path / f"{file_name_prefix}Daily_qb_Plot.png", dpi=300,  bbox_inches="tight")
        fig22.savefig(plots_path / f"{file_name_prefix}Annual_qse_Map.png", dpi=300,  bbox_inches="tight")
        fig23.savefig(plots_path / f"{file_name_prefix}Annual_qie_Map.png", dpi=300,  bbox_inches="tight")
        fig24.savefig(plots_path / f"{file_name_prefix}Outlet_Hydrograph.png", dpi=300,  bbox_inches="tight")

    if display_plots:
        plt.show()












