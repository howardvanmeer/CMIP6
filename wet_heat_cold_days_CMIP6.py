"""
Seasonal Analysis of Wet, Hot, and Cold Days Using CMIP6 Climate Model Data and Weather Station Observations
Developed by Howard van Meer, August 2025
Contact: howard.vanmeer@wur.nl and vanmeer@inta.gob.ar
Description:
-------------
This script processes and visualizes seasonal statistics of wet days (>0.5 mm precipitation), hot days (Tmax >= 35°C), and cold days (Tmin < 0°C) for Southern South America (Ssa) using both bias-corrected CMIP6 climate model outputs and measured weather station data. It generates comparative plots for historical and future climate scenarios (rcp45, rcp85), showing individual model results, multi-model means, and observed station data across decades and seasons.
Main Steps:
-----------
1. Imports and configuration of models, scenarios, variables, directories, and time ranges.
2. Loads and processes weather station data, including seasonal/yearly summaries and assignment of seasons.
3. Loads station metadata and selects the target station coordinates.
4. For each variable (precipitation, tmax, tmin):
    - Calculates seasonal counts of wet, hot, and cold days from station data.
    - For each scenario and model:
        - Loads historical and future NetCDF data.
        - Applies thresholds to define wet/hot/cold days.
        - Aggregates data by season and year.
        - Plots per-year values and decadal means for each model, historical reference, and station observations.
        - Plots multi-model mean and spread for future scenarios.
    - Customizes plot appearance, legends, and saves figures to disk.
Outputs:
--------
- Seasonal graphs for each variable, scenario, and model, showing historical and future projections.
- Multi-model mean plots with spread (standard deviation) for each scenario.
- Figures saved as PNG files in the specified output directory.
Notes:
------
- The script assumes the presence of bias-corrected CMIP6 NetCDF files and weather station CSV files at specified paths.
- Longitude conventions are handled for compatibility between datasets.
- Decadal means are calculated for both historical (1990-2019) and future (2030-2100) periods.
- Custom color schemes and plot annotations are used for clarity.
"""
# =============================================================================
# Imports and Setup
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import os
import xarray as xr
import numpy as np
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
if not os.path.exists(CMIP6DIR):
    os.makedirs(CMIP6DIR)

# Year ranges for historical and future periods
histyears = list(map(str, range(1985, 2015)))
years = list(map(str, range(2025, 2100)))
obs_years = list(map(str, range(1990, 2020)))

# Open the last saved bias-corrected NetCDF file in xarray for inspection
model = ['miroc6','cmcc_esm2','mpi_esm1_2_lr','gfdl_esm4','noresm2_mm','ipsl_cm6a_lr','hadgem3_gc31_ll','ec_earth3_cc']
obs_years = list(map(str, range(1990, 2020)))
# =============================================================================
# Open and process measured weather variables from weatherstation
# =============================================================================
lamaria = pd.read_csv('......Downloads/lamaria8824.csv')
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
# Options for Name in meteostations:
print(list(meteostations['Name'].unique()))
name_weatherstation = meteostations['Name']== 'INTA LA MARIA EMC'
if name_weatherstation.any():
    station_row = meteostations.loc[name_weatherstation].iloc[0]
    station_lat = station_row['Lat']
    station_lon = station_row['Lon'] + 360  # CMIP6 data uses a different longitude convention please check beforehand
else:
    # Uses manual values stated below if station is not present in list meteostations
    station_lat = -28.02
    station_lon = 295.77  # Ensure this is numeric and matches your dataset's longitude convention
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

lamaria_sum_wetdays_hist = {}  # Initialize the dictionary before use
lamaria_sum_hotdays_hist = {}
lamaria_sum_colddays_hist = {}
# =============================================================================
# Step 1: Make seasonal graphs with all models
# =============================================================================
for var in variable_adjusted:
    lamaria_obs_sum_hist = {}
    if var == "precipitation":
        # Count wet days (Pr > 0.5 mm)
        lamaria_sum_wetdays_hist = lamaria.groupby(['season', 'season_year'])['Pr'].apply(lambda x: (x > 0.5).sum()).reset_index(name='wetdays')
    elif var == "tmax":
        lamaria_sum_hotdays_hist= lamaria.groupby(['season', 'season_year'])['Tmax'].apply(lambda x: (x >= 35).sum()).reset_index(name='hotdays')
    elif var == "tmin":
        lamaria_sum_colddays_hist= lamaria.groupby(['season', 'season_year'])['Tmin'].apply(lambda x: (x < 0).sum()).reset_index(name='colddays')

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
                DATADIR = f"..../Downloads/{str(climategen)}/"
                obs_file = f"{DATADIR}/{climategen}_19902019_Ssa_{var}.nc"                
                obs = xr.open_dataset(obs_file)
                if 'time_bnds' in obs:
                    obs = obs.drop_vars('time_bnds')
                obs = obs.assign_coords(year=obs['time'].dt.year)
                # Define criteria for wet, hot, and cold days, and assign a value of 1 when the respective criteria are met 
                if var == "precipitation" and 'precipitation' in obs:
                    obs = obs.assign(wetdays=(obs['precipitation'] > 0.5).astype(int))
                elif var == "tmax" and 'Temperature' in obs:
                    obs = obs.assign(hotdays=(obs['Temperature'] > 35).astype(int))
                elif var == "tmin" and 'Temperature' in obs:
                    obs = obs.assign(colddays=(obs['Temperature'] < 0).astype(int))

                future_file = f"{CMIP6DIR}/{mod}/{mod}_{scen}_{min(years)}-{max(years)}_{var}_BC.nc"
                future = xr.open_dataset(future_file)
                future = future.assign_coords(year=future['time'].dt.year)
                # Define criteria for wet, hot, and cold days, and assign a value of 1 when the respective criteria are met 
                if var == "precipitation" and 'precipitation_BC' in future:
                    future = future.assign(wetday=(future['precipitation_BC'] > 0.5).astype(int))
                elif var == "tmax" and 'tmax_BC' in future:
                    future = future.assign(hotday=(future['tmax_BC'] > 35).astype(int))
                elif var == "tmin" and 'tmin_BC' in future:
                    future = future.assign(colddays=(future['tmin_BC'] < 0).astype(int))
                # Delete variable from obs and _BC from future if present
                if var == "precipitation":
                    if 'precipitation' in obs:
                        obs = obs.drop_vars('precipitation')
                    if 'precipitation_BC' in future:
                        future = future.drop_vars('precipitation_BC')
                elif var == "tmax":
                    if 'Temperature' in obs:
                        obs = obs.drop_vars('Temperature')
                    if 'tmax_BC' in future:
                        future = future.drop_vars('tmax_BC')
                elif var == "tmin":
                    if 'Temperature' in obs:
                        obs = obs.drop_vars('Temperature')
                    if 'tmin_BC' in future:
                        future = future.drop_vars('tmin_BC')

                # Group by season and year, then sum over time to get total amount of days per season
                if var == "precipitation":
                    season_year_sum_future = future.groupby(["time.season", "year"]).sum(dim="time")            
                    data = season_year_sum_future.sel(lat=station_lat, lon=station_lon, method='nearest')

                    season_year_sum_hist = obs.groupby(["time.season", "year"]).sum(dim="time")
                    data_hist = season_year_sum_hist.sel(lat=station_lat, lon=station_lon-360, method='nearest')
                    data_obs = lamaria_sum_wetdays_hist
                elif var == "tmax":
                    season_year_sum_future = future.groupby(["time.season", "year"]).sum(dim="time")            
                    data = season_year_sum_future.sel(lat=station_lat, lon=station_lon, method='nearest')

                    season_year_sum_hist = obs.groupby(["time.season", "year"]).sum(dim="time")
                    data_hist = season_year_sum_hist.sel(lat=station_lat, lon=station_lon-360, method='nearest')
                    data_obs = lamaria_sum_hotdays_hist
                elif var == "tmin":
                    season_year_sum_future = future.groupby(["time.season", "year"]).sum(dim="time")            
                    data = season_year_sum_future.sel(lat=station_lat, lon=station_lon, method='nearest')

                    season_year_sum_hist = obs.groupby(["time.season", "year"]).sum(dim="time")
                    data_hist = season_year_sum_hist.sel(lat=station_lat, lon=station_lon-360, method='nearest')
                    data_obs = lamaria_sum_colddays_hist

                data = data.sel(season=season)
                years_plot = data['year'].values
                if var == "precipitation":
                    var_bc_name = "wetdays"
                elif var == "tmax":
                    var_bc_name = "hotdays"
                elif var == "tmin":
                    var_bc_name = "colddays"
                else:
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
                axes[i].plot(years_hist, values_hist, 'o', color=model_colors[mod], alpha=0.3, markersize=2)
                # Plot decadal means as line (history)
                if hist_decade_centers:
                    axes[i].plot(hist_decade_centers, hist_decades, marker='o', linestyle='--', color=model_colors[mod], linewidth=2, alpha=0.7)
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
                if var == "precipitation":
                    data_obs_season = lamaria_sum_wetdays_hist[lamaria_sum_wetdays_hist['season'] == season]
                    years_obs = data_obs_season['season_year'].values
                    values_obs = data_obs_season['wetdays'].values
                elif var == "tmax":
                    data_obs_season = lamaria_sum_hotdays_hist[lamaria_sum_hotdays_hist['season'] == season]
                    years_obs = data_obs_season['season_year'].values
                    values_obs = data_obs_season['hotdays'].values
                elif var == "tmin":
                    data_obs_season = lamaria_sum_colddays_hist[lamaria_sum_colddays_hist['season'] == season]
                    years_obs = data_obs_season['season_year'].values
                    values_obs = data_obs_season['colddays'].values
                    
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
            # Set y-ticks and labels every 2.5 units, label every 5 units
            ymin, ymax = axes[i].get_ylim()
            yticks = np.arange(np.floor(ymin / 2.5) * 2.5, np.ceil(ymax / 2.5) * 2.5 + 0.1, 2.5)
            axes[i].set_yticks(yticks)
            axes[i].set_yticklabels(
                [str(int(y)) if y % 5 == 0 else "" for y in yticks],
                fontsize=10
            )            
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
            if var == 'precipitation':
                axes[i].set_ylabel('Wet days (Days)')
            elif var == 'tmax':
                axes[i].set_ylabel('Hot days (Days)')
            else:
                axes[i].set_ylabel('Cold days (Days)')
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
        if var == 'tmax':
            plt.suptitle(f"Seasonal amount of hot days at lat={station_lat:.2f}, lon={display_lon:.2f} (all models) - Scenario: {scen}", y=1.04)
        elif var == 'tmin':
            plt.suptitle(f"Seasonal amount of cold days at lat={station_lat:.2f}, lon={display_lon:.2f} (all models) - Scenario: {scen}", y=1.04)
        else:
            plt.suptitle(f"Seasonal amount of wet days (>0.5 mm) at lat={station_lat:.2f}, lon={display_lon:.2f} (all models) - Scenario: {scen}", y=1.04)
            # Save figure as PNG in CMIP6DIR/graph
            graph_dir = os.path.join(CMIP6DIR, "graph")
            if not os.path.exists(graph_dir):
                os.makedirs(graph_dir)
            if var == 'tmax':
                fname = f"{var}_{scen}_seasonal_hotdays_lamaria.png"
            if var == 'tmin':
                fname = f"{var}_{scen}_seasonal_colddays_lamaria.png"   
            if var == 'precipitation':
                fname = f"{var}_{scen}_seasonal_wetdays_lamaria.png"
            plt.savefig(os.path.join(graph_dir, fname), dpi=150, bbox_inches='tight')
        plt.show()
# ==========================================================================================
# Step 2: Make seasonal graphs with all multi model mean (hot days, cold days and wet days)
# ==========================================================================================
for var in variable_adjusted:
    lamaria_obs_sum_hist = {}
    if var == "precipitation":
        lamaria_sum_wetdays_hist = lamaria.groupby(['season', 'season_year'])['Pr'].apply(lambda x: (x > 0.5).sum()).reset_index(name='wetdays')
    elif var == "tmax":
        lamaria_sum_hotdays_hist = lamaria.groupby(['season', 'season_year'])['Tmax'].apply(lambda x: (x >= 35).sum()).reset_index(name='hotdays')
    elif var == "tmin":
        lamaria_sum_colddays_hist = lamaria.groupby(['season', 'season_year'])['Tmin'].apply(lambda x: (x < 0).sum()).reset_index(name='colddays')
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()
    for i, season in enumerate(seasons):
        # Historical (same for both scenarios)
        all_hist_decadal_means = []
        hist_decade_centers = []
        for mod in model:
            DATADIR = f"..../Downloads/{str(climategen)}/"
            obs_file = f"{DATADIR}/{climategen}_19902019_Ssa_{var}.nc"
            obs = xr.open_dataset(obs_file)
            obs = obs.assign_coords(year=obs['time'].dt.year)
            if 'time_bnds' in obs:
                obs = obs.drop_vars('time_bnds')
            obs = obs.assign_coords(year=obs['time'].dt.year)
            if var == "precipitation" and 'precipitation' in obs:
                obs = obs.assign(wetdays=(obs['precipitation'] > 0.5).astype(int))
            elif var == "tmax" and 'Temperature' in obs:
                obs = obs.assign(hotdays=(obs['Temperature'] > 35).astype(int))
            elif var == "tmin" and 'Temperature' in obs:
                obs = obs.assign(colddays=(obs['Temperature'] < 0).astype(int))
            else:
                continue
            # Select the correct lat/lon for the station from obs for historical data
            if var == "precipitation":
                data_hist = obs.groupby(["time.season", "year"]).sum(dim="time").sel(lat=station_lat, lon=station_lon-360, method='nearest')
            elif var == "tmax":
                data_hist = obs.groupby(["time.season", "year"]).sum(dim="time").sel(lat=station_lat, lon=station_lon-360, method='nearest')
            elif var == "tmin":
                data_hist = obs.groupby(["time.season", "year"]).sum(dim="time").sel(lat=station_lat, lon=station_lon-360, method='nearest')
            data_hist = data_hist.sel(season=season)
            years_hist = data_hist['year'].values
            if var == "precipitation":
                var_bc_name = "wetdays"
            elif var == "tmax":
                var_bc_name = "hotdays"
            elif var == "tmin":
                var_bc_name = "colddays"
            else:
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
            # Plot per-year values as light blue dots for historical (WFDE5)
            axes[i].plot(years_hist, values_hist, 'o', color='dodgerblue', alpha=0.5, markersize=2.5, label='Historical yearly (WFDE5)' if mod == model[0] and i == 0 else "")
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
        # --- Plot obs (station) ---
        if var == "precipitation":
            data_obs_season = lamaria_sum_wetdays_hist[lamaria_sum_wetdays_hist['season'] == season]
            years_obs = data_obs_season['season_year'].values
            values_obs = data_obs_season['wetdays'].values
        elif var == "tmax":
            data_obs_season = lamaria_sum_hotdays_hist[lamaria_sum_hotdays_hist['season'] == season]
            years_obs = data_obs_season['season_year'].values
            values_obs = data_obs_season['hotdays'].values
        elif var == "tmin":
            data_obs_season = lamaria_sum_colddays_hist[lamaria_sum_colddays_hist['season'] == season]
            years_obs = data_obs_season['season_year'].values
            values_obs = data_obs_season['colddays'].values
        # Plot per-year values as black dots for station obs
        axes[i].plot(years_obs, values_obs, 'o', color='black', alpha=0.7, markersize=2.5, label='Station yearly values' if i == 0 else "")
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
                if var == "precipitation" and 'precipitation_BC' in future:
                    future = future.assign(wetdays=(future['precipitation_BC'] > 0.5).astype(int))
                    var_bc_name = "wetdays"
                elif var == "tmax" and 'tmax_BC' in future:
                    future = future.assign(hotdays=(future['tmax_BC'] > 35).astype(int))
                    var_bc_name = "hotdays"
                elif var == "tmin" and 'tmin_BC' in future:
                    future = future.assign(colddays=(future['tmin_BC'] < 0).astype(int))
                    var_bc_name = "colddays"
                else:
                    var_bc_name = f"{var.lower()}_bc"
                # Delete variable from obs and _BC from future if present
                if var == "precipitation":
                    if 'precipitation' in obs:
                        obs = obs.drop_vars('precipitation')
                    if 'precipitation_BC' in future:
                        future = future.drop_vars('precipitation_BC')
                elif var == "tmax":
                    if 'Temperature' in obs:
                        obs = obs.drop_vars('Temperature')
                    if 'tmax_BC' in future:
                        future = future.drop_vars('tmax_BC')
                elif var == "tmin":
                    if 'Temperature' in obs:
                        obs = obs.drop_vars('Temperature')
                    if 'tmin_BC' in future:
                        future = future.drop_vars('tmin_BC')
                data = future.groupby(["time.season", "year"]).sum(dim="time").sel(lat=station_lat, lon=station_lon, method='nearest')
                data = data.sel(season=season)
                years_plot = data['year'].values
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
        axes[i].set_xticks(
            np.concatenate([
                np.arange(1990, 2021, 5),  # historical ticks
                np.arange(2030, 2101, 5)   # future ticks
            ])
        )
        # Set y-ticks and labels every 2.5 units, label every 5 units
        ymin, ymax = axes[i].get_ylim()
        yticks = np.arange(np.floor(ymin / 2.5) * 2.5, np.ceil(ymax / 2.5) * 2.5 + 0.1, 2.5)
        axes[i].set_yticks(yticks)
        axes[i].set_yticklabels(
            [str(int(y)) if y % 5 == 0 else "" for y in yticks],
            fontsize=10
        )
        axes[i].set_xticklabels(
            [str(y) for y in np.concatenate([np.arange(1990, 2021, 5), np.arange(2030, 2101, 5)])],
            rotation=45, fontsize=9
        )
        # Add decade labels above the plot (only for first row)
        if i < 2:
            pad = 5  # horizontal pad in years
            for idx, (center, label) in enumerate(zip(decade_centers, decade_labels)):
                axes[i].text(center + pad, axes[i].get_ylim()[1]*1.02, f"$\\bf{{{label}}}$", ha='center', va='bottom', fontsize=10, color='darkred')
        # Use full season name in lower part of subplot
        if var == 'precipitation':
            axes[i].set_ylabel('Wet days (Days)')
        elif var == 'tmax':
            axes[i].set_ylabel('Hot days (Days)')
        else:
            axes[i].set_ylabel('Cold days (Days)')
        axes[i].set_xlabel('Year')
        if i == 0:
            axes[i].legend(loc='lower right', fontsize=10)
        axes[i].text(0.5, 0.925, season_name_map.get(season, season), ha='center', va='bottom', fontsize=12, color='black', transform=axes[i].transAxes)
        # Add vertical line at year 2024
        axes[i].axvline(x=2024, color='black', linewidth=4, linestyle='-', alpha=0.7)
        # Add left arrow and label for "Historic"
        if i == 0:
            if var == 'tmax':
                global_ymax = max(ax.get_ylim()[1] for ax in axes)
                y_annot = global_ymax * 0.875
                
            elif var == 'tmin':
                global_ymax = max(ax.get_ylim()[1] for ax in axes)
                y_annot = 1.855  
            elif var == 'precipitation' and season == 'DJF':
                global_ymax = max(ax.get_ylim()[1] for ax in axes)
                y_annot = global_ymax * 0.9
                
            elif var == 'precipitation' and season in ['MAM', 'JJA', 'SON']:
                global_ymax = max(ax.get_ylim()[1] for ax in axes)
                y_annot = global_ymax * 500  
                
            else:
                global_ymax = max(ax.get_ylim()[1] for ax in axes)
                y_annot = global_ymax * 500
                
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
    if var == 'tmax':
        plt.suptitle(f"Amount of hot days at lat={station_lat:.2f}, lon={display_lon:.2f} (multi-model mean, scenarios) - Historic + {', '.join(scenario_adjusted)}", y=1.04)
    elif var == 'tmin':
        plt.suptitle(f"Amount of cold days at lat={station_lat:.2f}, lon={display_lon:.2f} (multi-model mean, scenarios) - Historic + {', '.join(scenario_adjusted)}", y=1.04)
    else:
        plt.suptitle(f"Amount of wet days at lat={station_lat:.2f}, lon={display_lon:.2f} (multi-model mean, scenarios) - Historic + {', '.join(scenario_adjusted)}", y=1.04)
    graph_dir = os.path.join(CMIP6DIR, "graph")
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    if var == 'tmax':
        fname = f"{var}_multi_model_hotdays_scenarios_seasonal_lamaria.png"
    if var == 'tmin':
        fname = f"{var}_multi_model_colddays_scenarios_seasonal_lamaria.png"
    if var == 'precipitation':
        fname = f"{var}_multi_model_wetdays_scenarios_seasonal_lamaria.png"
    plt.savefig(os.path.join(graph_dir, fname), dpi=150, bbox_inches='tight')
    plt.show()

