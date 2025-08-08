"""
CMIP6 Bias Correction and Regridding Script

Overview:
    This script processes CMIP6 climate model outputs for Southern South America (SSA), focusing on bias correction and regridding to match the WFDE5 observational grid.
    Workflow:
        - Crop model files to a predefined Area of Interest (AOI) to ensure compatibility with the WFDE5 grid.
        - Remove extraneous variables and leap days for consistency.
        - Regrid both model and observational datasets to a common spatial grid.
        - Apply quantile mapping bias correction using the Ibicus package.
        - Save bias-corrected NetCDF files for precipitation, daily maximum and minimum temperature.

Data Sources:
    - Observational reference: WFDE5_CRU (Cucchi et al., 2020, DOI: https://doi.org/10.5194/essd-12-2097-2020, URL: https://essd.copernicus.org/articles/12/2097/2020/)
    - Bias correction method: Ibicus package (Spuler et al., 2024, DOI: 10.5194/gmd-17-1249-2024, URL: https://gmd.copernicus.org/articles/17/1249/2024/)

Developed by Howard van Meer (2025).
Contact: <a href="mailto:howard.vanmeer@wur.nl">howard.vanmeer@wur.nl</a> ; <a href="mailto:vanmeer.howard@inta.gob.ar">vanmeer.howard@inta.gob.ar</a>
"""

# =============================================================================
# Imports and Setup
# =============================================================================
import os
import iris
import numpy as np
import urllib3
import xarray as xr
import warnings
import ibicus
import datetime
import cftime

# =============================================================================
# Configuration: Models, Scenarios, Variables, Directories
# =============================================================================
region = 'Ssa'  # Southern South America
climategen = 'WFDE5_CRU'
model = ['miroc6','cmcc_esm2','mpi_esm1_2_lr','noresm2_mm','ipsl_cm6a_lr','ec_earth3_cc','hadgem3_gc31_ll']
scenario = ['historical', 'ssp2_4_5', "ssp5_8_5"]
scenario_adjusted = ['rcp45','rcp85']
variable = ['precipitation',"daily_maximum_near_surface_air_temperature", "daily_minimum_near_surface_air_temperature"]
variable_adjusted = ['precipitation', 'tmax', 'tmin']

# Data directory for CMIP6 files
CMIP6DIR = f"......./CMIP6/"
if not os.path.exists(CMIP6DIR):
    os.makedirs(CMIP6DIR)

# Year ranges for historical and future periods
histyears = list(map(str, range(1985, 2015)))
years = list(map(str, range(2025, 2100)))

# =============================================================================
# Step 1: Crop and Adjust Model Files to Fit WFDE5 Grid
# =============================================================================
# Reduce extent to fit inside WFDE5 grid, remove bounds, crop longitude/latitude
for mod in model:
    for scen in scenario_adjusted:
        # Special cases for some models/variables
        if mod == "ipsl_cm6a_lr" and scen == "rcp85":
            vars_adjusted_to_use = ["precipitation", "tmin"]
        elif mod == "hadgem3_gc31_ll":
            vars_adjusted_to_use = ["precipitation", "tmin"]
        else:
            vars_adjusted_to_use = variable_adjusted

        for var in vars_adjusted_to_use:
            hist_file = f"{CMIP6DIR}/{mod}/{mod}_historical_{min(histyears)}-{max(histyears)}_{var}.nc"
            future_file = f"{CMIP6DIR}/{mod}/{mod}_{scen}_{min(years)}-{max(years)}_{var}.nc"

            # --- Process historical file ---
            histdata = xr.open_dataset(hist_file, decode_cf=True)
            for bnd_var in ['time_bnds', 'lat_bnds', 'lon_bnds']:
                if bnd_var in histdata.variables:
                    histdata = histdata.drop_vars(bnd_var)
            if 'longitude' in histdata.coords and (histdata.longitude > -70+360).any():
                histdata = histdata.sel(latitude=histdata.latitude > -70+360)
            elif 'lon' in histdata.coords and (histdata.lon > -70+360).any():
                histdata = histdata.sel(lon=histdata.lon > -70+360)
            if 'longitude' in histdata.coords:
                histdata = histdata.sel(latitude=histdata.latitude > -70+360)
            elif 'lon' in histdata.coords:
                histdata = histdata.sel(lon=histdata.lon > -70+360)
            histdata.to_netcdf(hist_file.replace(".nc", "_adjusted.nc"), mode='w')

            # --- Process future file ---
            futuredata = xr.open_dataset(future_file, decode_cf=True)
            for bnd_var in ['time_bnds', 'lat_bnds', 'lon_bnds']:
                if bnd_var in futuredata.variables:
                    futuredata = futuredata.drop_vars(bnd_var)
            if 'longitude' in futuredata.coords and (futuredata.longitude > -70+360).any():
                futuredata = futuredata.sel(latitude=futuredata.latitude > -70+360)
            elif 'lon' in futuredata.coords and (futuredata.lon > -70+360).any():
                futuredata = futuredata.sel(lon=futuredata.lon > -70+360)
            if 'longitude' in futuredata.coords:
                futuredata = futuredata.sel(latitude=futuredata.latitude > -70+360)
            elif 'lon' in futuredata.coords:
                futuredata = futuredata.sel(lon=futuredata.lon > -70+360)
            futuredata.to_netcdf(future_file.replace(".nc", "_adjusted.nc"), mode='w')

            print(f"Hist file: {hist_file} for {var} adjusted to fit WFDE5 grid")
            print(f"Future file: for {scen} and {var}: {future_file} adjusted to fit WFDE5 grid")

# =============================================================================
# Step 2: Regridding and Bias Correction
# =============================================================================
# Regrid historical and future data to WFDE5 grid and apply bias correction
for mod in model:
    for scen in scenario_adjusted:
        if mod == "ipsl_cm6a_lr" and scen == "rcp85":
            vars_adjusted_to_use = ["precipitation", "tmin"]
        elif mod == "hadgem3_gc31_ll":
            vars_adjusted_to_use = ["precipitation", "tmin"]
        else:
            vars_adjusted_to_use = variable_adjusted

        for var in vars_adjusted_to_use:
            DATADIR = f"...../{str(climategen)}/"
            # Load observational reference (WFDE5)
            obs = iris.load_cube(f"{DATADIR}/{climategen}_19902019_Ssa_{var}.nc")
            # Print min/max for diagnostics
            if var in ["tmax", "tmin"]:
                print(f"Min value of {var} in obs: {np.round(obs.data.min(),2)} °C")
                print(f"Max value of {var} in obs: {np.round(obs.data.max(),2)} °C")
            elif var == "precipitation":
                print(f"Min value of {var} in obs: {np.floor(obs.data.min())} mm/day")
                print(f"Max value of {var} in obs: {np.floor(obs.data.max())} mm/day")
            else:
                print(f"Min value of {var} in obs: {obs.data.min()}")
                print(f"Max value of {var} in obs: {obs.data.max()}")

            # --- Remove leap days from observations ---
            def not_leap_day(cell):
                date = cell.point
                return not (date.month == 2 and date.day == 29)
            obs = obs.extract(iris.Constraint(time=not_leap_day))
            start_date = datetime.datetime(1990, 1, 1)
            end_date = datetime.datetime(int(max(histyears)), 12, 31)
            time_obs_overlap = iris.Constraint(time=lambda cell: start_date <= cell.point <= end_date)
            obs = obs.extract(time_obs_overlap)

            # --- Load historical data ---
            if var == "precipitation":
                hist = iris.load_cube(f"{CMIP6DIR}/{mod}/{mod}_historical_{min(histyears)}-{max(histyears)}_{var}_adjusted.nc", "pr")
                hist.data = np.where(hist.data < 0, 0, hist.data)
                print(f"Min value of {var} in hist: {np.floor(hist.data.min() * 86400)} mm/day")
                print(f"Max value of {var} in hist: {np.floor(hist.data.max() * 86400)} mm/day")
                hist = hist.extract(iris.Constraint(time=not_leap_day))
                # Get calendar type and set time overlap
                calendar_type = hist.coord('time').units.calendar
                print(f"Calendar type for {mod} {var}: {calendar_type}")
                if calendar_type == 'standard' or calendar_type == 'proleptic_gregorian':
                    start_date = datetime.datetime(1990, 1, 1)
                    end_date = datetime.datetime(int(max(histyears)), 12, 31)
                elif calendar_type == '365_day':
                    start_date = cftime.DatetimeNoLeap(1990, 1, 1, 0, 0)
                    end_date = cftime.DatetimeNoLeap(int(max(histyears)), 12, 31, 0, 0)
                elif calendar_type == '360_day':
                    start_date = cftime.Datetime360Day(1990, 1, 1, 0, 0)
                    end_date = cftime.Datetime360Day(int(max(histyears)), 12, 30, 0, 0)
                else:
                    raise ValueError(f"Unsupported calendar type: {calendar_type}")
                time_hist_overlap = iris.Constraint(time=lambda cell: start_date <= cell.point <= end_date)
                hist = hist.extract(time_hist_overlap)

            elif var == "tmax":
                hist = iris.load_cube(f"{CMIP6DIR}/{mod}/{mod}_historical_{min(histyears)}-{max(histyears)}_{var}_adjusted.nc", "tasmax")
                print(f"Min value of {var} in hist: {np.round(hist.data.min() - 273.15, 2)}°C")
                print(f"Max value of {var} in hist: {np.round(hist.data.max() - 273.15, 2)}°C")
                hist = hist.extract(iris.Constraint(time=not_leap_day))
                calendar_type = hist.coord('time').units.calendar
                print(f"Calendar type for {mod} {var}: {calendar_type}")
                if calendar_type == 'standard' or calendar_type == 'proleptic_gregorian':
                    start_date = datetime.datetime(1990, 1, 1)
                    end_date = datetime.datetime(int(max(histyears)), 12, 31)
                elif calendar_type == '365_day':
                    start_date = cftime.DatetimeNoLeap(1990, 1, 1, 0, 0)
                    end_date = cftime.DatetimeNoLeap(int(max(histyears)), 12, 31, 0, 0)
                elif calendar_type == '360_day':
                    start_date = cftime.Datetime360Day(1990, 1, 1, 0, 0)
                    end_date = cftime.Datetime360Day(int(max(histyears)), 12, 30, 0, 0)
                else:
                    raise ValueError(f"Unsupported calendar type: {calendar_type}")
                time_hist_overlap = iris.Constraint(time=lambda cell: start_date <= cell.point <= end_date)
                hist = hist.extract(time_hist_overlap)

            elif var == "tmin":
                hist = iris.load_cube(f"{CMIP6DIR}/{mod}/{mod}_historical_{min(histyears)}-{max(histyears)}_{var}_adjusted.nc", "tasmin")
                print(f"Min value of {var} in hist: {np.round(hist.data.min() - 273.15, 2)}°C")
                print(f"Max value of {var} in hist: {np.round(hist.data.max() - 273.15, 2)}°C")
                hist = hist.extract(iris.Constraint(time=not_leap_day))
                calendar_type = hist.coord('time').units.calendar
                print(f"Calendar type for {mod} {var}: {calendar_type}")
                if calendar_type == 'standard' or calendar_type == 'proleptic_gregorian':
                    start_date = datetime.datetime(1990, 1, 1)
                    end_date = datetime.datetime(int(max(histyears)), 12, 31)
                elif calendar_type == '365_day':
                    start_date = cftime.DatetimeNoLeap(1990, 1, 1, 0, 0)
                    end_date = cftime.DatetimeNoLeap(int(max(histyears)), 12, 31, 0, 0)
                elif calendar_type == '360_day':
                    start_date = cftime.Datetime360Day(1990, 1, 1, 0, 0)
                    end_date = cftime.Datetime360Day(int(max(histyears)), 12, 30, 0, 0)
                else:
                    raise ValueError(f"Unsupported calendar type: {calendar_type}")
                time_hist_overlap = iris.Constraint(time=lambda cell: start_date <= cell.point <= end_date)
                hist = hist.extract(time_hist_overlap)
            else:
                raise ValueError(f"Unknown variable: {var}")

            # --- Load future data ---
            if var == "precipitation":
                future = iris.load_cube(f"{CMIP6DIR}/{mod}/{mod}_{scen}_{min(years)}-{max(years)}_{var}_adjusted.nc", "pr")
                future.data = np.where(future.data < 0, 0, future.data)
                print(f"Min value of {var} in future {scen}: {np.floor(future.data.min() * 86400)} mm/day")
                print(f"Max value of {var} in future {scen}: {np.floor(future.data.max() * 86400)} mm/day")
            elif var == "tmax":
                future = iris.load_cube(f"{CMIP6DIR}/{mod}/{mod}_{scen}_{min(years)}-{max(years)}_{var}_adjusted.nc", "tasmax")
                print(f"Min value of {var} in future {scen}: {np.round(future.data.min() - 273.15, 2)}°C")
                print(f"Max value of {var} in future {scen}: {np.round(future.data.max() - 273.15, 2)}°C")
            elif var == "tmin":
                future = iris.load_cube(f"{CMIP6DIR}/{mod}/{mod}_{scen}_{min(years)}-{max(years)}_{var}_adjusted.nc", "tasmin")
                print(f"Min value of {var} in future {scen}: {np.round(future.data.min() - 273.15, 2)}°C")
                print(f"Max value of {var} in future {scen}: {np.round(future.data.max() - 273.15, 2)}°C")
            else:
                raise ValueError(f"Unknown variable: {var}")
            futuredata = xr.open_dataset(f"{CMIP6DIR}/{mod}/{mod}_{scen}_{min(years)}-{max(years)}_{var}_adjusted.nc", decode_cf=True)

            # --- Regrid obs to hist grid ---
            obs = obs.regrid(hist, iris.analysis.Nearest())

            # --- Extract dates for all datasets ---
            def get_dates(x):
                time_dimension = x.coords()[0]
                dates = time_dimension.units.num2date(time_dimension.points)
                return dates
            dates_hist = get_dates(hist)
            dates_future = get_dates(future)
            dates_obs = get_dates(obs)

            # --- Convert data to numpy arrays, fill masked values ---
            hist = hist.data
            if np.ma.isMaskedArray(hist):
                hist = np.ma.filled(hist, np.nan)
            future = future.data
            if np.ma.isMaskedArray(future):
                future = np.ma.filled(future, np.nan)
            obs = obs.data
            if np.ma.isMaskedArray(obs):
                obs = np.ma.filled(obs, np.nan)

            print(f"Shape hist files: {hist.shape}")
            print(f"Shape future files: {future.shape}")
            print(f"Shape obs_WFDE5: {obs.shape}")

            # --- Unit conversions for bias correction ---
            if var in ['tmax', 'tmin', 'tmean', 'tdew']:
                print(f"Min value of {var} in obs before conversion: {np.round(obs.min(), 2)} °C")
                print(f"Max value of {var} in obs before conversion: {np.round(obs.max(), 2)} °C")
                obs = obs + 273.15
                print(f"Min value of {var} in obs after conversion: {np.round(obs.min(), 2)} K")
                print(f"Max value of {var} in obs after conversion: {np.round(obs.max(), 2)} K")
            elif var == 'precipitation':
                print(f"Min value of {var} in obs before conversion: {np.floor(obs.min())} mm/day")
                print(f"Max value of {var} in obs before conversion: {np.floor(obs.max())} mm/day")
                obs = (obs / 86400)
                print(f"Min value of {var} in obs after conversion: {np.round(obs.min(), 5)} (kg m-2 s-1)")
                print(f"Max value of {var} in obs after conversion: {np.round(obs.max(), 5)} (kg m-2 s-1)")

            # =============================================================================
            # Step 3: Bias Correction with Ibicus Quantile Mapping
            # =============================================================================
            from ibicus.debias import QuantileMapping

            # Select debiaser based on variable type
            if var in ['tmax', 'tmin', 'tmean', 'tdew']:
                debiaser = QuantileMapping.from_variable("tas")
            elif var == 'precipitation':
                debiaser = QuantileMapping.from_variable("pr")

            debiased_cm_future = debiaser.apply(obs, hist, future)

            # --- Convert units back for output ---
            if var in ['tmax', 'tmin', 'tmean', 'tdew']:
                debiased_cm_future = debiased_cm_future - 273.15
            elif var in ['precipitation', 'rad']:
                debiased_cm_future = (debiased_cm_future * 86400)

            # =============================================================================
            # Step 4: Save Bias-Corrected Output as NetCDF
            # =============================================================================
            future_biascorrected = xr.Dataset(
                {str(var).lower()+'_BC': (['time', 'latitude', 'longitude'], debiased_cm_future)},
                coords={
                    'time': futuredata['time'],
                    'latitude': futuredata['latitude'] if 'latitude' in futuredata.coords else futuredata['lat'],
                    'longitude': futuredata['longitude'] if 'longitude' in futuredata.coords else futuredata['lon']
                }
            )
            # Add metadata
            if var in ['tmax', 'tmin', 'tmean', 'tdew']:
                future_biascorrected[str(var).lower()+'_BC'] = future_biascorrected[str(var).lower()+'_BC'].assign_attrs(
                    units="DegC", description=f"Daily {var} (biascorrected) for {min(years)}-{max(years)} period")
            else:
                future_biascorrected[str(var).lower()+'_BC'] = future_biascorrected[str(var).lower()+'_BC'].assign_attrs(
                    units="mm", description=f"Daily {var} (biascorrected) for {min(years)}-{max(years)} period")
            future_biascorrected.attrs['Conventions'] = 'CF-1.7 (modified to be compatible with tools like CDO and NCO that work with CF-1.6)'
            future_biascorrected.attrs['Institution'] = 'ECMWF, Wageningen University (WUR), Instituto Nacional de Tecnología Agropecuaria (INTA)'
            future_biascorrected.attrs['Region'] = 'Southern South America (SSA) including Pampas and Gran Chaco Region, Argentina AOI 15S°-39°S and 49W°-70°W'
            actual_time = datetime.datetime.now()
            future_biascorrected.attrs['Processing date'] = actual_time.strftime("%d/%m/%Y %H:%M:%S")
            future_biascorrected.attrs['Contact'] = 'howard.vanmeer@wur.nl ; vanmeer.howard@inta.gob.ar'
            future_biascorrected.attrs['History'] = (
                f"Developed by Howard van Meer and Iwan Supit (WUR) Biascorrected (quantilemapping) bias correction of daily weather variables {var} "
                f"based on CMIP6 https://cds.climate.copernicus.eu/datasets/projections-cmip6?tab=overview"
            )
            # Save to NetCDF
            future_biascorrected.to_netcdf(
                f"{CMIP6DIR}/{mod}/{mod}_{scen}_{min(years)}-{max(years)}_{var}_BC.nc"
            )
            print(
                f"Biascorrection for {var}, {mod}, {scen} and all corresponding longitudes for given AOI end has been performed on {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')} by using Ibicus"
            )
