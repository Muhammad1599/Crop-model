import os
import pandas as pd
from datetime import datetime, timedelta
import yaml
import logging
import time
from tqdm import tqdm
import numpy as np
from pcse.fileinput import CABOFileReader, YAMLAgroManagementReader
from pcse.util import WOFOST71SiteDataProvider
from pcse.base import ParameterProvider
from pcse.db import NASAPowerWeatherDataProvider
from pcse.models import Wofost71_WLP_FD
import matplotlib.pyplot as plt
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze

# Disable logging
logging.disable(logging.CRITICAL)

# Custom representer to format dates correctly without time
def represent_date(self, data):
    value = data.strftime('%Y-%m-%d')
    return self.represent_scalar('tag:yaml.org,2002:str', value)

yaml.add_representer(datetime.date, represent_date)

# Setup paths and load data
crop_info_path = r'C:\data copy\WOFOST MODEL\Model input files\crop and agro\Crop info ID.xlsx'
output_directory = r'C:\data copy\WOFOST MODEL\Model input files\crop and agro\yaml agro files'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

crop_info = pd.read_excel(crop_info_path)
crop_info['Sowing date'] = pd.to_datetime(crop_info['Sowing date'], format='%d.%m.%Y').dt.date
crop_info['Harvest date'] = pd.to_datetime(crop_info['Harvest date'], format='%d.%m.%Y').dt.date

# Define the directories
soil_location_dir = r'C:\data copy\WOFOST MODEL\Model input files\Soil and Location'
agro_files_dir = r'C:\data copy\WOFOST MODEL\Model input files\crop and agro\yaml agro files'
id_soil_location_path = os.path.join(soil_location_dir, 'ID SOIL and Location.xlsx')
crop_parameters_file = os.path.join(agro_files_dir, '..', 'practice_relevantcrop.crop')

# Load the soil and crop information
id_soil_location = pd.read_excel(id_soil_location_path)
crop_info = pd.read_excel(crop_info_path)
crop_parameters = CABOFileReader(crop_parameters_file)

# Prepare additional columns for RRMSE
crop_info['RRMSE TWSO %'] = np.nan

# Define functions
def calculate_rrmse(observed, estimated):
    observed = np.array(observed, dtype=float)
    estimated = np.array(estimated, dtype=float)
    if observed.size == 0 or estimated.size == 0 or np.isnan(observed).all() or np.isnan(estimated).all():
        return np.nan
    return np.sqrt(np.nanmean((observed - estimated) ** 2)) / np.nanmean(observed) * 100

def update_crop_file(params, crop_parameters_file):
    with open(crop_parameters_file, 'r') as file:
        lines = file.readlines()

    with open(crop_parameters_file, 'w') as file:
        for line in lines:
            if line.startswith("TSUMEM"):
                value = params[0] if not np.isnan(params[0]) else 300  # Provide a default value if nan
                file.write(f"TSUMEM   = {value:.1f}    ! temperature sum from sowing to emergence [cel d]\n")
            elif line.startswith("TSUM1"):
                value = params[1] if not np.isnan(params[1]) else 1100  # Provide a default value if nan
                file.write(f"TSUM1    = {value:.1f}    ! temperature sum from emergence to anthesis [cel d]\n")
            elif line.startswith("TSUM2"):
                value = params[2] if not np.isnan(params[2]) else 680  # Provide a default value if nan
                file.write(f"TSUM2    = {value:.1f}    ! temperature sum from anthesis to maturity [cel d]\n")
            elif line.startswith("TDWI"):
                value = params[3] if not np.isnan(params[3]) else 140  # Provide a default value if nan
                file.write(f"TDWI     = {value:.1f}    ! initial total crop dry weight [kg ha-1]\n")
            elif line.startswith("LAIEM"):
                value = params[4] if not np.isnan(params[4]) else 0.1665  # Provide a default value if nan
                file.write(f"LAIEM    = {value:.4f}    ! leaf area index at emergence [ha ha-1]\n")
            elif line.startswith("SPAN"):
                value = params[5] if not np.isnan(params[5]) else 26.3  # Provide a default value if nan
                file.write(f"SPAN     = {value:.1f}    ! life span of leaves growing at 35 Celsius [d]\n")
            elif line.startswith("RGRLAI"):
                value = params[6] if not np.isnan(params[6]) else 0.0075  # Provide a default value if nan
                file.write(f"RGRLAI   = {value:.4f}    ! maximum relative increase in LAI [ha ha-1 d-1]\n")
            elif line.startswith("SLATB"):
                value = params[7] if not np.isnan(params[7]) else 0.0020  # Provide a default value if nan
                file.write(f"SLATB    = 0.00, {value:.4f},   ! specific leaf area\n")
            elif line.startswith("CFET"):
                value = params[8] if not np.isnan(params[8]) else 1.00  # Provide a default value if nan
                file.write(f"CFET     = {value:.2f}   ! correction factor transpiration rate [-]\n")
            elif line.startswith("DEPNR"):
                value = params[9] if not np.isnan(params[9]) else 4.5  # Provide a default value if nan
                file.write(f"DEPNR    = {value:.1f}    ! crop group number for soil water depletion [-]\n")
            elif line.startswith("RDMCR"):
                value = params[10] if not np.isnan(params[10]) else 125  # Provide a default value if nan
                file.write(f"RDMCR    = {value:.1f}     ! maximum rooting depth [cm]\n")
            else:
                file.write(line)

def run_model_and_get_rrmse(params, id_soil_location, crop_info, crop_parameters_file, agro_files_dir):
    update_crop_file(params, crop_parameters_file)
    
    errors_twso = []
    print(f"Running model with params: {params}")

    for index, crop_row in tqdm(crop_info.iterrows(), total=crop_info.shape[0], desc="Running simulations"):
        id_number = crop_row['ID']
        row_number = index + 1
        agro_file_path = os.path.join(agro_files_dir, f'agrorelevantcrop_ID_{id_number}_row_{row_number}.agro')

        if not os.path.exists(agro_file_path):
            print(f"Agro file not found: {agro_file_path}")
            continue
        
        soil_info = id_soil_location[id_soil_location['ID'] == id_number].iloc[0]
        soil_file = os.path.join(soil_location_dir, f"{soil_info['Soil File']}")
        if not os.path.exists(soil_file):
            print(f"Soil file not found: {soil_file}")
            continue

        soildata = CABOFileReader(soil_file)
        sitedata = WOFOST71SiteDataProvider(WAV=100)
        parameters = ParameterProvider(soildata=soildata, cropdata=CABOFileReader(crop_parameters_file), sitedata=sitedata)
        wdp = NASAPowerWeatherDataProvider(latitude=soil_info['Latitude'], longitude=soil_info['Longitude'])
        agromanagement = YAMLAgroManagementReader(agro_file_path)
        wofsim = Wofost71_WLP_FD(parameters, wdp, agromanagement)

        try:
            wofsim.run_till_terminate()
            df = pd.DataFrame(wofsim.get_output())
            if df.empty:
                raise ValueError("Model did not produce any output.")
        except Exception as e:
            print(f"Error running model for ID {id_number}, row {row_number}: {e}")
            continue

        estimated_yield = df['TWSO'].iloc[-1]
        crop_info.at[index, 'Dry matter yield (TWSO) kg/ha (estimated)'] = estimated_yield

        # Calculate RRMSE and store errors
        observed_yield = crop_row['Dry matter yield (TWSO) kg/ha (observed)']
        rrms_twso = calculate_rrmse([observed_yield], [estimated_yield])
        crop_info.at[index, 'RRMSE TWSO %'] = rrms_twso

        errors_twso.append(np.abs(observed_yield - estimated_yield))
    
    mean_rrmse_twso = crop_info['RRMSE TWSO %'].mean()
    print(f"mean_rrmse_twso: {mean_rrmse_twso}")
    return mean_rrmse_twso

# Sensitivity Analysis

# Define the problem for sensitivity analysis
problem = {
    'num_vars': 11,
    'names': ['TSUMEM', 'TSUM1', 'TSUM2', 'TDWI', 'LAIEM', 'SPAN', 'RGRLAI', 'SLATB', 'CFET', 'DEPNR', 'RDMCR'],
    'bounds': [
        (80, 140),
        (150, 300),
        (800, 1100),
        (180, 250),
        (0.1, 0.3),
        (25, 40),
        (0.005, 0.009),
        (0.0015, 0.0030),
        (0.7, 1.1),
        (2.5, 5.5),
        (110, 150)
    ]
}

# Number of trajectories (adjust based on computational capacity)
num_trajectories = 5

# Total number of simulations for Morris method
total_simulations = (problem['num_vars'] + 1) * num_trajectories
print(f"Total simulations for sensitivity analysis: {total_simulations}")

# Generate samples
param_values = morris_sample.sample(problem, N=num_trajectories, num_levels=4)

# Progress tracking
start_time = time.time()
last_update_time = start_time
total_samples = len(param_values)

# Evaluate the model for each sample with progress tracking
Y = []
for i, params in enumerate(param_values):
    result = run_model_and_get_rrmse(params, id_soil_location, crop_info, crop_parameters_file, agro_files_dir)
    if not np.isnan(result):
        Y.append(result)
    
    current_time = time.time()
    elapsed_time = current_time - start_time
    if current_time - last_update_time > 300:  # 5 minutes
        percentage_complete = (i + 1) / total_samples * 100
        estimated_total_time = elapsed_time / (i + 1) * total_samples
        estimated_remaining_time = estimated_total_time - elapsed_time
        print(f"Sensitivity Analysis Progress: {percentage_complete:.2f}% complete, "
              f"Estimated time remaining: {timedelta(seconds=estimated_remaining_time)}")
        last_update_time = current_time

Y = np.array(Y)

# Check if Y is not empty
if Y.size == 0:
    raise ValueError("All simulations resulted in NaN values. Please check the model and parameter ranges.")

# Perform the sensitivity analysis
Si = morris_analyze.analyze(problem, param_values, Y, conf_level=0.95, print_to_console=False)

# Plot the sensitivity analysis results
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(problem['names'], Si['mu_star'])
ax.set_title('Sensitivity Analysis using Morris Method')
ax.set_xlabel('Parameters')
ax.set_ylabel('Mu* (Mean of the absolute value of the elementary effects)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print results
for name, mu_star in zip(problem['names'], Si['mu_star']):
    print(f"{name}: {mu_star:.4f}")

# Calculate correlations between parameters
correlation_matrix = np.corrcoef(param_values.T)

# Plot the correlation matrix
fig, ax = plt.subplots(figsize=(12, 8))
cax = ax.matshow(correlation_matrix, cmap='coolwarm')
fig.colorbar(cax)
ax.set_xticks(np.arange(len(problem['names'])))
ax.set_yticks(np.arange(len(problem['names'])))
ax.set_xticklabels(problem['names'], rotation=45)
ax.set_yticklabels(problem['names'])
plt.title('Correlation Matrix of Parameters')
plt.show()

# Print the correlation matrix
correlation_df = pd.DataFrame(correlation_matrix, index=problem['names'], columns=problem['names'])
print(correlation_df)
