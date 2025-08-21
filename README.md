
This repository provides Python scripts for downloading, processing (bias correction), analyzing, and visualizing seasonal and decadal climate statistics. The workflow leverages bias-corrected CMIP6 model outputs, meteorological forcing datasets (WFDE5), and local weather station observations for Southern South America (SSA).

## Script Overview

1. **download_CMIP6.py**
    - Automates the download of CMIP6 data for predefined models from the Copernicus Climate Data Store ([CDS](https://cds.climate.copernicus.eu/)).
    - Supports batch downloading for user-defined time periods.

2. **CMIP6_process_biascorrection.py**
    - Preprocesses raw CMIP6 NetCDF files from CDS.
    - Performs bias correction, spatial subsetting, and variable extraction using the Ibicus package.
    - Outputs processed NetCDF files for downstream analysis.

3. **CMIP6_graphs.py** and **wet_heat_cold_days_CMIP6.py**
    - Generate seasonal graphs for precipitation, daily maximum and minimum temperature.
    - Compare historical (WFDE5) and future (CMIP6 scenarios) periods.
    - Analyze and visualize seasonal temperature and rainfall data and number of (hot/cold days) and wet days.
    - Plot results for individual models, multi-model means, and local station observations.
    - Save figures for reporting and further analysis.

## Data Sources

- **Observational Reference:** WFDE5_CRU (Cucchi et al., 2020)  
  [DOI: 10.5194/essd-12-2097-2020](https://doi.org/10.5194/essd-12-2097-2020)
- **Bias-corrected CMIP6 Outputs:** Spuler et al., 2024  
  [DOI: 10.5194/gmd-17-1249-2024](https://doi.org/10.5194/gmd-17-1249-2024)

## Usage

1. Ensure all required data files (NetCDF, CSV) are available in the specified directories.
2. Install dependencies:
    - `pandas`
    - `numpy`
    - `xarray`
    - `matplotlib`
    - `seaborn`
    - (and others as needed)
3. Run `CMIP6_graphs.py` to generate seasonal plots for selected variables and scenarios.
4. Output figures are saved in the `CMIP6/graph` directory.

## Contact

Developed by Howard van Meer (2025)  
Email: howard.vanmeer@wur.nl | vanmeer.howard@inta.gob.ar  
[LinkedIn](https://www.linkedin.com/in/howardvanmeer/)

## License

For academic use only. Please cite the relevant data sources and publications.
