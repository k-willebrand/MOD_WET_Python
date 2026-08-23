# import base libraries
import sys
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def get_day_of_year_flexible(target_dt64, start_month=10):
    """
    Returns day of water year based on specified start month.
    If start_month = 10, day of year is day of water year where October 1st = 1.
    If start_month = 1, day of year is the default day of year were January 1st = 1.
    """
    # Extract the calendar year and month
    dt_parsed = target_dt64.astype('M8[M]').astype(str)
    year = int(dt_parsed[:4])
    month = int(dt_parsed[5:7])
    
    # Determine the starting year of the water year
    water_year_start_year = year - 1 if start_month >= month else year
    
    # Create the October 1st reference date
    start_str = f"{water_year_start_year}-{start_month:02d}-01"
    start_dt64 = np.datetime64(start_str, 'D')
    
    # Compute the delta in days and add 1
    day_of_water_year = (target_dt64.astype('M8[D]') - start_dt64).astype(int) + 1

    return day_of_water_year

def time_series_aggregation(time_vector, data_vector, agg_func = "mean", output_freq = "D"):
    # Create datetime data series
    series = pd.Series(data_vector, index=pd.DatetimeIndex(time_vector))
    # Aggregate input data time resolution to daily (mean, sum, max, etc.)

    # if output_freq not in ['D', 'ME', 'QE', 'YE']:
    #     sys.exit("'output_freq' must be 'D' (daily),  'ME' (month-end), 'QE', (quarter-end), or 'YE' (year-end).")

    if agg_func == "mean":
        daily_series = series.resample(output_freq).mean()
    elif agg_func == "sum":
        daily_series = series.resample(output_freq).sum()
    else:
        sys.exit("'agg_func' must be 'mean' or 'sum'.")
    # Extract back to NumPy arrays
    daily_time = daily_series.index.values
    daily_data = daily_series.values
    return daily_time, daily_data

def montly_mean(time_vector, data_vector):
    df = pd.DataFrame({
        "datetime": time_vector,
        "data": data_vector
    })
    monthly = (
        df.groupby(df["datetime"].dt.to_period("M"))["data"]
          .mean()
          .reset_index()
    )
    # Convert YYYY-MM periods to NumPy datetime64
    monthly_datetime = monthly["datetime"].dt.to_timestamp().to_numpy()
    monthly_data = monthly["data"].to_numpy()

    return monthly_datetime, monthly_data

def plot_format(axes, xlab, ylab, grid=True, majcol='#D5D8DC', mincol='#EAECEE'):
    axes.set(xlabel=xlab, ylabel=ylab)
    axes.tick_params(axis="y", which="major", direction="in")
    axes.tick_params(axis="x", which="major", direction="in")
    axes.tick_params(axis="y", which="minor", direction="in")
    axes.tick_params(axis="x", which="minor", direction="in")
    if grid:
        # axes.grid()
        axes.grid(visible=True, which="major", color=majcol, linestyle="-", zorder=0)
        axes.grid(visible=True, which="minor", color=mincol, linestyle="-", zorder=0)
        axes.minorticks_on()
    return axes

def plot_spatial_data(fig, ax, x_data, y_data, plot_data, x_lab, y_lab, bar_label, 
                      plot_title: Optional[str] = None, 
                      plot_cmap: Optional[str] = "viridis", sci_not: Optional[bool] = False):
    # sort data as a fail-safe
    x_data = np.sort(x_data)
    y_data = np.sort(y_data)
    im = ax.imshow(plot_data, 
                   extent=[x_data[0], x_data[-1], y_data[-1], y_data[0]],
                   cmap=plot_cmap,
                   zorder=5
    )
    # add color bar
    cbar = fig.colorbar(im, ax=ax, shrink=1.0)
    cbar.set_label(bar_label, fontsize=12)

    # Formatting
    plot_format(ax, x_lab, y_lab, grid=False)
    if plot_title is not None:
        ax.set_title(plot_title, fontsize=12)
    if sci_not:
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True)

def plot_time_series(fig, ax, time_vector: np.ndarray, data_list: list, x_lab, y_lab,
                     data_labels: Optional[list] = None, data_colors: Optional[list] = None,
                     plot_title: Optional[str] = None, sci_not: Optional[bool] = False):
    num_var = len(data_list)

    # Error handling block
    if data_labels is not None:
        if len(data_labels) != num_var:
            sys.exit("Incorred number of data labels.")
    if data_colors is not None:
        if len(data_colors) != num_var:
            sys.exit("Incorred number of data colors.")

    # plot data
    for v in range(num_var):
        if data_labels is not None and data_colors is not None:
            ax.plot(time_vector, data_list[v], 
                    label=data_labels[v], color=data_colors[v], zorder=v+2)
        elif data_labels is not None and data_colors is None:
            ax.plot(time_vector, data_list[v], 
                    label=data_labels[v], zorder=v+2)
        elif data_labels is None and data_colors is not None:
            ax.plot(time_vector, data_list[v], 
                    color=data_colors[v], zorder=v+2)
        else:
            ax.plot(time_vector, data_list[v], zorder=v+2)

    # plot formatting
    plot_format(ax, x_lab, y_lab, grid=True)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=(1, 15)))
    if plot_title is not None:
        ax.set_title(plot_title, fontsize=12)
    if sci_not:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    if data_labels is not None:
        ax.legend(loc="best", frameon=True, framealpha=1)

    
    

