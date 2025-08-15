"""
CMIP6 Seasonal Graphing Script

Overview:
    This script generates seasonal graphs for Southern South America (SSA) using bias-corrected CMIP6 climate model outputs,
    meteorological forcing dataset (WFDE5) and observational weather station data from the local weather stations.

Workflow:
    - Loads and processes measured weather variables from a local weather station.
    - Loads bias-corrected CMIP6 model outputs for precipitation, daily maximum, and minimum temperature.
    - Calculates seasonal and decadal statistics for both historical and future periods.
    - Plots seasonal graphs for all models and multi-model means, including scenario comparisons.
    - Saves figures for further analysis and reporting.

Data Sources:
    - Observational reference: WFDE5_CRU (Cucchi et al., 2020, DOI: https://doi.org/10.5194/essd-12-2097-2020)
    - Bias-corrected CMIP6 outputs based on: (Spuler et al., 2024, DOI: 10.5194/gmd-17-1249-2024)

Developed by Howard van Meer (2025).
Contact: howard.vanmeer@wur.nl ; vanmeer.howard@inta.gob.ar ; https://www.linkedin.com/in/howardvanmeer/
"""

# =============================================================================
# Imports and Setup
# ============================================================================
import os
import pandas as pd
import numpy as np
import urllib3
import xarray as xr
import warnings
import datetime
import cftime
import itertools
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# Configuration: Models, Scenarios, Variables, Directories
# =============================================================================
region = 'Ssa'  # Southern South America
climategen = 'WFDE5_CRU'
model = ['miroc6','cmcc_esm2','mpi_esm1_2_lr','gfdl_esm4','noresm2_mm','ipsl_cm6a_lr','ec_earth3_cc','hadgem3_gc31_ll']
scenario = ['historical', 'ssp2_4_5', "ssp5_8_5"]
scenario_adjusted = ['rcp45','rcp85']
variable = ['precipitation',"daily_maximum_near_surface_air_temperature", "daily_minimum_near_surface_air_temperature"]
variable_adjusted = ['tmax','tmin','precipitation']

# Data directory for CMIP6 files
CMIP6DIR = f"...../Downloads/CMIP6/"
#CMIP6DIR = f'..../Downloads/CMIP6/'  
if not os.path.exists(CMIP6DIR):
    os.makedirs(CMIP6DIR)

# Year ranges for historical and future periods
histyears = list(map(str, range(1985, 2015)))
years = list(map(str, range(2025, 2100)))
obs_years = list(map(str, range(1990, 2020)))
# =============================================================================
# OOpen and process measured weather variables from weatherstation
# =============================================================================
lamaria = pd.read_csv('...../Downloads/lamaria8824.csv')
#lamaria = pd.read_csv('...../Lamaria8824.csv')
lamaria = lamaria.replace(-9999, np.nan)
lamaria.rename(columns={'AA': 'year', 'MM': 'month', 'DD': 'day'}, inplace=True)
lamaria['time'] = pd.to_datetime(lamaria[['year', 'month', 'day']])
lamaria['Tmean'] = (lamaria['Tmax'] + lamaria['Tmin']) / 2
# Cut lamaria for obs_years time range (1990-2019)
lamaria = lamaria[(lamaria['time'] >= pd.Timestamp('1989-12-01')) & (lamaria['time'] <= pd.Timestamp('2019-11-30'))]
# Prepare lamaria seasonal/yearly summaries for each variable
# Assign season, ensuring December (month 12) is counted as part of the next year's DJF
lamaria['season'] = lamaria['time'].dt.month.map(
    lambda m: 'DJF' if m in [12, 1, 2] else
              'MAM' if m in [3, 4, 5] else
              'JJA' if m in [6, 7, 8] else
              'SON'
)
# For DJF, increment year by 1 for December to group Dec-Jan-Feb correctly
lamaria['season_year'] = lamaria.apply(
    lambda row: row['year']  + 1 if row['season'] == 'DJF' and row['month'] == 12 else row['year'],
    axis=1
)
meteostations = pd.read_csv('...../Downloads/Meteostations_INTA.csv', encoding='latin1')
#meteostations = pd.read_csv('...../Data/Meteostations_INTA.csv', encoding='latin1')
# Options for Name in meteostations:
print(list(meteostations['Name'].unique()))
name_weatherstation = meteostations['Name']== 'INTA LA MARIA EMC'
station_row = meteostations.loc[name_weatherstation].iloc[0]
station_lat = station_row['Lat']
station_lon = station_row['Lon']+360
# Open the last saved bias-corrected NetCDF file in xarray for inspection
# Map season codes to full names
season_name_map = {'DJF': 'Summer', 'MAM': 'Autumn', 'JJA': 'Winter', 'SON': 'Spring'}
seasons = ['DJF', 'MAM', 'JJA', 'SON']
# Define decades to plot (including 2090s as 2090-2099)
decades = [(2030, 2039), (2040, 2049), (2050, 2059), (2060, 2069), (2070, 2079), (2080, 2089), (2090, 2099)]
decade_labels = [f"{start}s" for start, _ in decades]
decade_centers = [start for start, _ in decades]

# Assign a color to each model for plotting
colors = plt.get_cmap('tab10', len(model))
model_colors = {mod: colors(i) for i, mod in enumerate(model)}

# =============================================================================
# Step 1: Make seasonal graphs with all models
# =============================================================================
for var in variable_adjusted:
    lamaria_obs_sum_hist = {}
    if var == "precipitation":
        lamaria_obs_sum_hist[var] = lamaria.groupby(['season', 'season_year'])['Pr'].sum().reset_index()
    elif var == "tmax":
        lamaria_obs_sum_hist[var] = lamaria.groupby(['season', 'season_year'])['Tmax'].mean().reset_index()
    elif var == "tmin":
        lamaria_obs_sum_hist[var] = lamaria.groupby(['season', 'season_year'])['Tmin'].mean().reset_index()
    for scen in scenario_adjusted:
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
        axes = axes.flatten()
        for i, season in enumerate(seasons):
            all_decadal_means = []
            all_hist_decadal_means = []
            for mod_idx, mod in enumerate(model):
                # Special cases for some models/variables
                if mod == "ipsl_cm6a_lr" and scen == "rcp85":
                    vars_adjusted_to_use = ["precipitation", "tmin"]
                elif mod == "hadgem3_gc31_ll":
                    vars_adjusted_to_use = ["precipitation", "tmin"]
                elif mod == "gfdl_esm4" and scen == "rcp45":
                    vars_adjusted_to_use = ["precipitation", "tmax"]
                else:
                    vars_adjusted_to_use = variable_adjusted

                if var not in vars_adjusted_to_use:
                    continue
                DATADIR = f"C:/MyData/Ibicus/Downloads/{str(climategen)}/"
                obs_file = f"{DATADIR}/{climategen}_19902019_Ssa_{var}.nc"
                # Convert tasmin to Celsius if needed
                obs = xr.open_dataset(obs_file)
                obs = obs.assign_coords(year=obs['time'].dt.year)

                future_file = f"{CMIP6DIR}/{mod}/{mod}_{scen}_{min(years)}-{max(years)}_{var}_BC.nc"
                future = xr.open_dataset(future_file)
                future = future.assign_coords(year=future['time'].dt.year)

                # Group by season and year, then sum or mean over time
                if var == "precipitation":
                    season_year_sum_future = future.groupby(["time.season", "year"]).sum(dim="time")
                    data = season_year_sum_future.sel(lat=station_lat, lon=station_lon, method='nearest')
                    season_year_sum_hist = obs.groupby(["time.season", "year"]).sum(dim="time")
                    data_hist = season_year_sum_hist.sel(lat=station_lat, lon=station_lon-360, method='nearest')
                    data_obs = lamaria_obs_sum_hist
                elif var in ["tmax", "tmin"]:
                    season_year_mean_future = future.groupby(["time.season", "year"]).mean(dim="time")
                    data = season_year_mean_future.sel(lat=station_lat, lon=station_lon, method='nearest')
                    season_year_mean_hist = obs.groupby(["time.season", "year"]).mean(dim="time")
                    data_hist = season_year_mean_hist.sel(lat=station_lat, lon=station_lon-360, method='nearest')
                    data_obs = lamaria_obs_sum_hist
                else:
                    continue

                data = data.sel(season=season)
                years_plot = data['year'].values
                var_bc_name = f"{var.lower()}_bc"
                values_plot = data[var_bc_name].values if var_bc_name in data else data[list(data.data_vars)[0]].values

                # Calculate decadal means for future
                decadal_means = []
                for start, end in decades:
                    mask = (years_plot >= start) & (years_plot <= end)
                    if np.any(mask):
                        decadal_means.append(np.nanmean(values_plot[mask]))
                    else:
                        decadal_means.append(np.nan)
                all_decadal_means.append(decadal_means)

                # Plot per-year values as light dots (future)
                axes[i].plot(
                    years_plot,
                    values_plot,
                    'o',
                    color=model_colors[mod],
                    alpha=0.3,
                    markersize=1.5,
                    markeredgecolor='black',
                    markeredgewidth=0.7
                )
                # Plot decadal means as line (future)
                decade_x = decade_centers + [2100]
                decade_y = decadal_means + [decadal_means[-1]]
                axes[i].plot(decade_x, decade_y, marker='o', linestyle='-', color=model_colors[mod], linewidth=1, label=f'{mod}' if i == 0 else "")
                # Add a dot at 2100
                axes[i].plot(2100, decadal_means[-1], marker='o', color=model_colors[mod])

                # --- Plot history ---
                data_hist = data_hist.sel(season=season)
                years_hist = data_hist['year'].values
                values_hist = data_hist[var_bc_name].values if var_bc_name in data_hist else data_hist[list(data_hist.data_vars)[0]].values

                # Calculate decadal means for history (1990-2019)
                hist_decades = []
                hist_decade_centers = []
                for start in range(int(min(obs_years)), int(max(obs_years))+1, 10):
                    end = min(start+9, int(max(obs_years)))
                    mask = (years_hist >= start) & (years_hist <= end)
                    if np.any(mask):
                        hist_decades.append(np.nanmean(values_hist[mask]))
                        hist_decade_centers.append(start)
                all_hist_decadal_means.append(hist_decades)

                # Plot per-year values as light dots (history)
                axes[i].plot(years_hist, values_hist, 'o', color=model_colors[mod], alpha=0.3, markersize=1.5)
                # Plot decadal means as line (history)
                if hist_decade_centers:
                    axes[i].plot(hist_decade_centers, hist_decades, marker='o', linestyle='--', color=model_colors[mod], linewidth=1.5, alpha=0.7)
                    if hist_decades:
                        # Plot a dot at 2019 and connect with a line to the last decade mean
                        color = 'red' if var in ['tmax', 'tmin'] else 'blue'
                        axes[i].plot([hist_decade_centers[-1], 2019], [hist_decades[-1], hist_decades[-1]], linestyle='--', color=color, linewidth=4)
                        axes[i].plot(
                            2019, hist_decades[-1], 
                            marker='o', color=color, linewidth=4, 
                            markeredgecolor='black', markeredgewidth=1.5
                        )

                # --- Plot obs (station) ---
                data_obs_season = lamaria_obs_sum_hist[var][lamaria_obs_sum_hist[var]['season'] == season]
                years_obs = data_obs_season['season_year'].values
                values_obs = data_obs_season[var.capitalize() if var != "precipitation" else "Pr"].values

                # Calculate decadal means for obs (1990-2019)
                obs_decades = []
                obs_decade_centers = []
                for start in range(int(min(obs_years)), int(max(obs_years))+1, 10):
                    end = min(start+9, int(max(obs_years)))
                    mask = (years_obs >= start) & (years_obs <= end)
                    if np.any(mask):
                        obs_decades.append(np.nanmean(values_obs[mask]))
                        obs_decade_centers.append(start)

                # Plot per-year values as black dots (obs)
                axes[i].plot(years_obs, values_obs, 'o', color='black', alpha=0.5, markersize=2 if i == 0 else 1)
                # Plot decadal means as thick black dashed line (obs)
                if obs_decade_centers:
                    axes[i].plot(obs_decade_centers, obs_decades, marker='o', linestyle='--', color='black', linewidth=2.5, label='Station observations' if mod == 'ec_earth3_cc' and i == 0 else "")
                    if obs_decades:
                        axes[i].plot([obs_decade_centers[-1], 2019], [obs_decades[-1], obs_decades[-1]], linestyle='--', color='black', linewidth=3)
                        axes[i].plot(
                            2019, obs_decades[-1], 
                            marker='o', color='black', linewidth=3, 
                            markeredgecolor='black', markeredgewidth=1.5
                        )

            # Plot multi-model mean with thick line (future)
            if all_decadal_means:
                all_decadal_means_np = np.array(all_decadal_means)
                mean_decadal = np.nanmean(all_decadal_means_np, axis=0)
                mean_decadal_y = mean_decadal.tolist() + [mean_decadal[-1]]
                color = 'red' if var in ['tmax', 'tmin'] else 'blue'
                axes[i].plot(
                    decade_x, mean_decadal_y, 
                    marker='o', linestyle='-', color=color, linewidth=4, 
                    label='Multi-model mean',
                    markeredgecolor='black', markeredgewidth=1.5
                )
                # Plot a dot at 2100 for the multi-model mean with a black edge
                axes[i].plot(
                    2100, mean_decadal[-1], 
                    marker='o', color=color, linewidth=4, 
                    markeredgecolor='black', markeredgewidth=1.5
                )

            # Plot multi-model mean for history (dashed)
            if all_hist_decadal_means:
                # Only include decades up to 2019
                valid_hist_decades = [x for x in all_hist_decadal_means if len(x) == len(hist_decade_centers)]
                if valid_hist_decades:
                    all_hist_decadal_means_np = np.array(valid_hist_decades)
                    mean_hist_decadal = np.nanmean(all_hist_decadal_means_np, axis=0)
                    color = 'red' if var in ['tmax', 'tmin'] else 'blue'
                    # Only plot up to 2020
                    plot_centers = [c for c in hist_decade_centers if c <= 2020]
                    plot_means = mean_hist_decadal[:len(plot_centers)]
                    axes[i].plot(
                        plot_centers, plot_means, 
                        marker='o', linestyle='--', color=color, linewidth=4, 
                        label='Historical (WFDE5)',
                        markeredgecolor='black', markeredgewidth=1.5
                    )
            # Set x-ticks and labels
            # Set x-ticks and labels for both historical (1990-2020) and future (2030-2100) periods
            axes[i].set_xticks(
                np.concatenate([
                np.arange(1990, 2021, 5),  # historical ticks
                np.arange(2030, 2101, 5)   # future ticks
                ])
            )
            axes[i].set_xticklabels(
                [str(y) for y in np.concatenate([np.arange(1990, 2021, 5), np.arange(2030, 2101, 5)])],
                rotation=45, fontsize=8
            )
            # Add decade labels above the plot (only for first row)
            if i < 2:
                pad = 5  # horizontal pad in years
                for idx, (center, label) in enumerate(zip(decade_centers, decade_labels)):
                    axes[i].text(center + pad, axes[i].get_ylim()[1]*1.02, f"$\\bf{{{label}}}$", ha='center', va='bottom', fontsize=10, color='darkred')
            # Use full season name in lower part of subplot
            axes[i].set_ylabel('Precipitation (mm)' if var == 'precipitation' else f"{var.capitalize()} (°C)")
            axes[i].set_xlabel('Year')
            # Only show legend in the first subplot
            if i == 0:
                axes[i].legend(loc='upper left', fontsize=8)
            # Lower season name inside figure
            axes[i].text(0.5, 0.925, season_name_map.get(season, season), ha='center', va='bottom', fontsize=12, color='black', transform=axes[i].transAxes)
            # Ensure precipitation y-axis minimum is never negative
            if var == 'precipitation':
                ymin, ymax = axes[i].get_ylim()
                if ymin < 0:
                    axes[i].set_ylim(bottom=0, top=ymax)
        plt.tight_layout()
        # Adjust longitude for display if needed (assuming 0-360 input, convert to -180 to 180)
        display_lon = station_lon if station_lon <= 180 else station_lon - 360
        if var in ['tmax', 'tmin']:
            plt.suptitle(f"Seasonal average of {var} at lat={station_lat:.2f}, lon={display_lon:.2f} (all models) - Scenario: {scen}", y=1.04)
        else:
            plt.suptitle(f"Seasonal sum of {var} at lat={station_lat:.2f}, lon={display_lon:.2f} (all models) - Scenario: {scen}", y=1.04)
        # Save figure as PNG in CMIP6DIR/graph
        graph_dir = os.path.join(CMIP6DIR, "graph")
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
        fname = f"{var}_{scen}_seasonal_lamaria.png"
        plt.savefig(os.path.join(graph_dir, fname), dpi=150, bbox_inches='tight')
        plt.show()    
# =============================================================================
# Step 2: Make seasonal graphs with all multi model mean
# =============================================================================
for var in variable_adjusted:
    lamaria_obs_sum_hist = {}
    if var == "precipitation":
        lamaria_obs_sum_hist[var] = lamaria.groupby(['season', 'season_year'])['Pr'].sum().reset_index()
    elif var == "tmax":
        lamaria_obs_sum_hist[var] = lamaria.groupby(['season', 'season_year'])['Tmax'].mean().reset_index()
    elif var == "tmin":
        lamaria_obs_sum_hist[var] = lamaria.groupby(['season', 'season_year'])['Tmin'].mean().reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()
    for i, season in enumerate(seasons):
        # Historical (same for both scenarios)
        all_hist_decadal_means = []
        hist_decade_centers = []
        for mod in model:
            DATADIR = f"C:/MyData/Ibicus/Downloads/{str(climategen)}/"
            obs_file = f"{DATADIR}/{climategen}_19902019_Ssa_{var}.nc"
            obs = xr.open_dataset(obs_file)
            obs = obs.assign_coords(year=obs['time'].dt.year)
            if var == "precipitation":
                season_year_sum_hist = obs.groupby(["time.season", "year"]).sum(dim="time")
                data_hist = season_year_sum_hist.sel(lat=station_lat, lon=station_lon-360, method='nearest')
            elif var in ["tmax", "tmin"]:
                season_year_mean_hist = obs.groupby(["time.season", "year"]).mean(dim="time")
                data_hist = season_year_mean_hist.sel(lat=station_lat, lon=station_lon-360, method='nearest')
            else:
                continue
            data_hist = data_hist.sel(season=season)
            years_hist = data_hist['year'].values
            var_bc_name = f"{var.lower()}_bc"
            values_hist = data_hist[var_bc_name].values if var_bc_name in data_hist else data_hist[list(data_hist.data_vars)[0]].values
            # Calculate decadal means for history (1990-2019)
            hist_decades = []
            hist_decade_centers = []
            for start in range(int(min(obs_years)), int(max(obs_years))+1, 10):
                end = min(start+9, int(max(obs_years)))
                mask = (years_hist >= start) & (years_hist <= end)
                if np.any(mask):
                    hist_decades.append(np.nanmean(values_hist[mask]))
                    hist_decade_centers.append(start)
            all_hist_decadal_means.append(hist_decades)
        # Plot multi-model mean for history (dashed)
        if all_hist_decadal_means:
            valid_hist_decades = [x for x in all_hist_decadal_means if len(x) == len(hist_decade_centers)]
            if valid_hist_decades:
                all_hist_decadal_means_np = np.array(valid_hist_decades)
                mean_hist_decadal = np.nanmean(all_hist_decadal_means_np, axis=0)
                color = 'red' if var in ['tmax', 'tmin'] else 'blue'
                plot_centers = [c for c in hist_decade_centers if c <= 2020]
                plot_means = mean_hist_decadal[:len(plot_centers)]
                axes[i].plot(
                    plot_centers, plot_means,
                    marker='o', linestyle='--', color=color, linewidth=3,
                    label='Historical (WFDE5)',
                    markeredgecolor='black', markeredgewidth=1.5
                )
                # Add a dot at 2019 and connect with a line to the last decade mean
                if plot_centers:
                    axes[i].plot([plot_centers[-1], 2019], [plot_means[-1], plot_means[-1]], linestyle='--', color=color, linewidth=4)
                    axes[i].plot(
                        2019, plot_means[-1],
                        marker='o', color=color, linewidth=3,
                        markeredgecolor='black', markeredgewidth=1.5
                    )

        # --- Incorporate data_obs (station obs) ---
        data_obs_season = lamaria_obs_sum_hist[var][lamaria_obs_sum_hist[var]['season'] == season]
        years_obs = data_obs_season['season_year'].values
        values_obs = data_obs_season[var.capitalize() if var != "precipitation" else "Pr"].values
        # Calculate decadal means for obs (1990-2019)
        obs_decades = []
        obs_decade_centers = []
        for start in range(int(min(obs_years)), int(max(obs_years))+1, 10):
            end = min(start+9, int(max(obs_years)))
            mask = (years_obs >= start) & (years_obs <= end)
            if np.any(mask):
                obs_decades.append(np.nanmean(values_obs[mask]))
                obs_decade_centers.append(start)
        # Plot decadal means as thick black dashed line (obs)
        if obs_decade_centers:
            axes[i].plot(obs_decade_centers, obs_decades, marker='o', linestyle='--', color='black', linewidth=2.5, label='Station observations' if i == 0 else "")
            if obs_decades:
                axes[i].plot([obs_decade_centers[-1], 2019], [obs_decades[-1], obs_decades[-1]], linestyle='--', color='black', linewidth=3)
                axes[i].plot(
                    2019, obs_decades[-1],
                    marker='o', color='black', linewidth=3,
                    markeredgecolor='black', markeredgewidth=1.5
                )

        # Future scenarios: plot both on same axes
        for scen_idx, scen in enumerate(scenario_adjusted):
            all_decadal_means = []
            for mod in model:
                # Special cases for some models/variables
                if mod == "ipsl_cm6a_lr" and scen == "rcp85":
                    vars_adjusted_to_use = ["precipitation", "tmin"]
                elif mod == "hadgem3_gc31_ll":
                    vars_adjusted_to_use = ["precipitation", "tmin"]
                elif mod == "gfdl_esm4" and scen == "rcp45":
                    vars_adjusted_to_use = ["precipitation", "tmax"]
                else:
                    vars_adjusted_to_use = variable_adjusted
                if var not in vars_adjusted_to_use:
                    continue
                future_file = f"{CMIP6DIR}/{mod}/{mod}_{scen}_{min(years)}-{max(years)}_{var}_BC.nc"
                future = xr.open_dataset(future_file)
                future = future.assign_coords(year=future['time'].dt.year)
                if var == "precipitation":
                    season_year_sum_future = future.groupby(["time.season", "year"]).sum(dim="time")
                    data = season_year_sum_future.sel(lat=station_lat, lon=station_lon, method='nearest')
                elif var in ["tmax", "tmin"]:
                    season_year_mean_future = future.groupby(["time.season", "year"]).mean(dim="time")
                    data = season_year_mean_future.sel(lat=station_lat, lon=station_lon, method='nearest')
                else:
                    continue
                data = data.sel(season=season)
                years_plot = data['year'].values
                var_bc_name = f"{var.lower()}_bc"
                values_plot = data[var_bc_name].values if var_bc_name in data else data[list(data.data_vars)[0]].values
                # Calculate decadal means for future
                decadal_means = []
                for start, end in decades:
                    mask = (years_plot >= start) & (years_plot <= end)
                    if np.any(mask):
                        decadal_means.append(np.nanmean(values_plot[mask]))
                    else:
                        decadal_means.append(np.nan)
                all_decadal_means.append(decadal_means)
            # Plot multi-model mean and spread (cloud) for this scenario
            if all_decadal_means:
                all_decadal_means_np = np.array(all_decadal_means)
                mean_decadal = np.nanmean(all_decadal_means_np, axis=0)
                std_decadal = np.nanstd(all_decadal_means_np, axis=0)
                mean_decadal_y = mean_decadal.tolist() + [mean_decadal[-1]]
                std_decadal_y = std_decadal.tolist() + [std_decadal[-1]]
                decade_x = decade_centers + [2100]
                color = 'red' if var in ['tmax', 'tmin'] else 'blue'
                # Use different alpha for each scenario
                alpha = 0.15 if scen_idx == 0 else 0.25
                if var in ['tmax', 'tmin']:
                    color = '#ff9999' if scen_idx == 0 else '#b20000'  # light red, dark red
                else:
                    color = '#99ccff' if scen_idx == 0 else '#003366'  # light blue, dark blue
                line_style = '-'
                label = f'Multi-model mean {scen}'
                axes[i].plot(
                    decade_x, mean_decadal_y,
                    marker='o', linestyle=line_style, color=color, linewidth=3,
                    label=label,
                    markeredgecolor='black', markeredgewidth=1.5
                )
                # Plot spread as shaded area (same alpha for all plots)
                axes[i].fill_between(
                    decade_x,
                    np.array(mean_decadal_y) - np.array(std_decadal_y),
                    np.array(mean_decadal_y) + np.array(std_decadal_y),
                    color=color, alpha=0.2
                )
        # Set x-ticks and labels
        # Set x-ticks and labels for both historical (1990-2020) and future (2030-2100) periods
        axes[i].set_xticks(
            np.concatenate([
            np.arange(1990, 2021, 5),  # historical ticks
            np.arange(2030, 2101, 5)   # future ticks
            ])
        )
        axes[i].set_xticklabels(
            [str(y) for y in np.concatenate([np.arange(1990, 2021, 5), np.arange(2030, 2101, 5)])],
            rotation=45, fontsize=8
        )
        # Add decade labels above the plot (only for first row)
        if i < 2:
            pad = 5  # horizontal pad in years
            for idx, (center, label) in enumerate(zip(decade_centers, decade_labels)):
                axes[i].text(center + pad, axes[i].get_ylim()[1]*1.02, f"$\\bf{{{label}}}$", ha='center', va='bottom', fontsize=10, color='darkred')
        axes[i].set_ylabel('Precipitation (mm)' if var == 'precipitation' else f"{var.capitalize()} (°C)")
        axes[i].set_xlabel('Year')
        if i == 0:
            axes[i].legend(loc='lower right', fontsize=10)
        axes[i].text(0.5, 0.925, season_name_map.get(season, season), ha='center', va='bottom', fontsize=12, color='black', transform=axes[i].transAxes)
        # Add vertical line at year 2024
        axes[i].axvline(x=2024, color='black', linewidth=4, linestyle='-', alpha=0.7)
        # Add left arrow and label for "Historic"
        if i == 0:
            if var in ['tmax', 'tmin']:
                global_ymax = max(ax.get_ylim()[1] for ax in axes)
                y_annot = global_ymax * 0.95
            else:
                global_ymax = max(ax.get_ylim()[1] for ax in axes)
                y_annot = global_ymax * 0.875
        axes[i].annotate(
            'Historic',
            xy=(2023, y_annot),
            xytext=(2010, y_annot),
            arrowprops=dict(arrowstyle='<-', color='black', lw=2),
            ha='right', va='center', fontsize=12, fontweight='bold', color='black'
        )
        axes[i].annotate(
            'Future',
            xy=(2025, y_annot),
            xytext=(2040, y_annot),
            arrowprops=dict(arrowstyle='<-', color='black', lw=2),
            ha='left', va='center', fontsize=12, fontweight='bold', color='black'
        )
        # Ensure precipitation y-axis minimum is never negative
        if var == 'precipitation':
            ymin, ymax = axes[i].get_ylim()
            if ymin < 0:
                axes[i].set_ylim(bottom=0, top=ymax)
    plt.tight_layout()
    display_lon = station_lon if station_lon <= 180 else station_lon - 360
    if var in ['tmax', 'tmin']:
        plt.suptitle(f"Seasonal average of {var} at lat={station_lat:.2f}, lon={display_lon:.2f} (multi-model mean, scenarios) - Historic + {', '.join(scenario_adjusted)}", y=1.04)
    else:
        plt.suptitle(f"Seasonal sum of {var} at lat={station_lat:.2f}, lon={display_lon:.2f} (multi-model mean, scenarios) - Historic + {', '.join(scenario_adjusted)}", y=1.04)
    graph_dir = os.path.join(CMIP6DIR, "graph")
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    fname = f"{var}_multi_model_mean_scenarios_seasonal_lamaria.png"
    plt.savefig(os.path.join(graph_dir, fname), dpi=150, bbox_inches='tight')

    plt.show()      
