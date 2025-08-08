"""
Automated CMIP6 Climate Projections Downloader

This script automates the download of CMIP6 climate projections from the Copernicus Climate Data Store (CDS)
(https://cds.climate.copernicus.eu/datasets/projections-cmip6?tab=overview). Developed by Howard van Meer (2025),
it streamlines the process of requesting, downloading, and organizing daily climate data for multiple models,
scenarios, and variables over a specified region (Southern South America).

"""
# =========================
# 1. Imports and Setup
# =========================
import cdsapi
import os
import zipfile
import numpy as np
import urllib3
import xarray as xr
import warnings
warnings.filterwarnings("ignore")
# =========================
# 2. Configuration
# =========================

# Define region, climate generator, models, scenarios, and variables
region = 'Ssa'
climategen = 'WFDE5_CRU'
model = ['miroc6','cmcc_esm2','mpi_esm1_2_lr','noresm2_mm','ipsl_cm6a_lr','ec_earth3_cc','hadgem3_gc31_ll']
scenario = ['historical', 'ssp2_4_5', "ssp5_8_5"]
scenario_adjusted = ['rcp45','rcp85']
variable = ['precipitation',"daily_maximum_near_surface_air_temperature", "daily_minimum_near_surface_air_temperature"]
variable_adjusted = ['precipitation', 'tmax', 'tmin']
area = [-15, -70, -39, -49]  # [lat_max, lon_min, lat_min, lon_max] for South America

# Output directory for downloads
CMIP6DIR = f"....../CMIP6/"
if not os.path.exists(CMIP6DIR):
    os.makedirs(CMIP6DIR)

# CDS API credentials
URL = 'https://cds.climate.copernicus.eu/api'
KEY = '******'

# Define years, months, and days for data requests
histyears = list(map(str, range(1985, 2015)))
years = list(map(str, range(2025, 2100)))
months = list(map(str, range(1, 13)))
days = list(map(str, range(1, 32)))

# =========================
# 3. Data Download
# =========================

# Initialize CDS API client
c = cdsapi.Client(url=URL, key=KEY)

# Loop through models, scenarios, and variables to request data
for mod in model:
    for scen in scenario:
        # Select variables based on model and scenario
        if mod == "ipsl_cm6a_lr" and scen == "ssp5_8_5":
            vars_to_use = ["precipitation", "daily_minimum_near_surface_air_temperature"]
        elif mod == "hadgem3_gc31_ll":
            vars_to_use = ["precipitation", "daily_minimum_near_surface_air_temperature"]
        else:
            vars_to_use = variable
        for var in vars_to_use:
            print(f"----- Requesting model: {mod} -----")
            print(f"----- Requesting scenario: {scen} -----")
            print(f"----- Requesting variable: {var} -----")
            output_dir = os.path.join(CMIP6DIR, str(mod))
            os.makedirs(output_dir, exist_ok=True)
            # Select years based on scenario
            selected_years = histyears if scen == 'historical' else years
            if scen == 'historical':
                output_file = os.path.join(output_dir, f"cmip6_historical_daily_{selected_years[0]}-{selected_years[-1]}_{scen}_{mod}_{var}.zip")
            else:
                output_file = os.path.join(output_dir, f"cmip6_daily_{selected_years[0]}-{selected_years[-1]}_{scen}_{mod}_{var}.zip")
            # Request data from CDS API
            result = c.retrieve(
                "projections-cmip6",
                {
                    "temporal_resolution": "daily",
                    "experiment": scen,
                    "variable": var,
                    "model": mod,
                    "year": selected_years,
                    "month": months,
                    "day": days,
                    "area": area,
                },
                output_file
            )
# =========================
# 4. Unzipping Downloaded Files
# =========================

# Extract .nc files from downloaded zip archives
for mod in model:
    output_dir = os.path.join(CMIP6DIR, str(mod))
    for file in os.listdir(output_dir):
        if file.endswith(".zip"):
            zip_path = os.path.join(output_dir, file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    if member.endswith('.nc'):
                        zip_ref.extract(member, output_dir)

# =========================
# 5. Renaming NetCDF Files
# =========================

# Rename .nc files to a more readable format
for mod in model:
    output_dir = os.path.join(CMIP6DIR, str(mod))
    for file in os.listdir(output_dir):
        if file.endswith(".nc"):
            src = os.path.join(output_dir, file)
            # Determine variable type and scenario
            if "tasmax" in file:
                var_type = "tmax"
            elif "tasmin" in file:
                var_type = "tmin"
            elif "pr_day" in file:
                var_type = "precipitation"
            else:
                continue  # Skip files that don't match
            if "historical" in file:
                scenario_str = f"historical_{min(histyears)}-{max(histyears)}"
            elif "45" in file:
                scenario_str = f"rcp45_{min(years)}-{max(years)}"
            elif "85" in file:
                scenario_str = f"rcp85_{min(years)}-{max(years)}"
            else:
                continue  # Skip files that don't match
            new_name = f"{mod}_{scenario_str}_{var_type}.nc"
            dst = os.path.join(output_dir, new_name)
            os.rename(src, dst)
