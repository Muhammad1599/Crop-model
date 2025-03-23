import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product, combinations

import pcse
from pcse.models import Wofost72_WLP_FD
from pcse.base import ParameterProvider
from pcse.db import NASAPowerWeatherDataProvider
from pcse.fileinput import CABOFileReader, YAMLAgroManagementReader
from pcse.util import WOFOST72SiteDataProvider

# Import SALib for advanced sensitivity analysis
from SALib.sample.morris import sample
from SALib.analyze import morris

# Use matplotlib style if available
if hasattr(matplotlib, 'style'):
    matplotlib.style.use("ggplot")

print("This script was built with:")
print(f"Python version: {sys.version}")
print(f"PCSE version: {pcse.__version__}")

# Define data directory
data_dir = r'C:\Users\user\Documents\ind set 2'

# Read crop data
cropfile = os.path.join(data_dir, 'practice_sugarbeet.crop')
cropdata = CABOFileReader(cropfile)
print("Crop Data:")
print(cropdata)

# Read soil data
soilfile = os.path.join(data_dir, 'Medium.soil')
soildata = CABOFileReader(soilfile)
print("Soil Data:")
print(soildata)

# Define site data
sitedata = WOFOST72SiteDataProvider(WAV=100, CO2=360)
print("Site Data:")
print(sitedata)

# Combine parameters
parameters = ParameterProvider(cropdata=cropdata, soildata=soildata, sitedata=sitedata)

# Read agromanagement data
agromanagement_file = os.path.join(data_dir, 'agrofile_npk.agro')
agromanagement = YAMLAgroManagementReader(agromanagement_file)
print("Agromanagement Data:")
print(agromanagement)

# Define weather data
wdp = NASAPowerWeatherDataProvider(latitude=53.1525327612394, longitude=10.7421664705163)
print("Weather Data Provider:")
print(wdp)

# Switches to control which parameters to include in the sensitivity analysis
include_conversion = True

# Conversion parameters initial values
default_values_conversion = {
    "CVL": 0.608,
    "CVO": 0.591,
    "CVR": 0.659,
    "CVS": 0.631
}

# Set conversion parameters
for param, value in default_values_conversion.items():
    parameters.set_override(param, value)

# Run initial WOFOST model with default parameters
wofost_initial = Wofost72_WLP_FD(parameters, wdp, agromanagement)
wofost_initial.run_till_terminate()
df_initial = pd.DataFrame(wofost_initial.get_output())
df_initial.index = pd.to_datetime(df_initial.day)
print("Initial WOFOST Output:")
print(df_initial.tail())

# Define parameters to include in the sensitivity analysis
parameters_to_analyze = []

if include_conversion:
    # Include conversion parameters
    parameters_to_analyze.extend(["CVL", "CVO", "CVR", "CVS"])

# Extract the parameters to analyze based on switches
parameter_switches = {
    "TSUMEM": True,
    "TSUM1": True,
    "TSUM2": True,
    "TDWI": True,
    "SPAN": True,
    "RGRLAI": True,
    "DEPNR": True,
    "RDMCR": True,
    "CFET": True,
    "DLC": True,
    "DLO": True,
    "PERDL": True,
    "Q10": True,
    "RDI": True,
    "RML": True,
    "RMO": True,
    "RMR": True,
    "RMS": True,
    "SPA": True,
    "TBASEM": True,
    "TEFFMX": True
}

parameters_to_analyze_other = [param for param, switch in parameter_switches.items() if switch]
parameters_to_analyze.extend(parameters_to_analyze_other)

print("Parameters to analyze:", parameters_to_analyze)

# Create default values for parameters
default_values = []

if include_conversion:
    default_values.extend([default_values_conversion[param] for param in ["CVL", "CVO", "CVR", "CVS"]])

# For other parameters
defaults = {param: cropdata[param] for param in parameter_switches.keys()}
default_values.extend([defaults[param] for param in parameters_to_analyze_other])

# Now define parameter bounds
parameter_bounds = []

if include_conversion:
    # Bounds for conversion parameters
    conversion_bounds = {
        "CVL": (0.55, 0.65),
        "CVO": (0.55, 0.65),
        "CVR": (0.60, 0.70),
        "CVS": (0.60, 0.70)
    }
    parameter_bounds.extend([conversion_bounds[param] for param in ["CVL", "CVO", "CVR", "CVS"]])

# Bounds for other parameters
parameter_ranges = {
    "TSUMEM": (50, 150),
    "TSUM1": (700, 900),
    "TSUM2": (1200, 1500),
    "TDWI": (30, 60),
    "SPAN": (15, 40),
    "RGRLAI": (0.008, 0.015),
    "DEPNR": (3.5, 4.5),
    "RDMCR": (50, 150),
    "CFET": (0.5, 1.5),
    "DLC": (-99.0, 15.0),
    "DLO": (-99.0, 18.0),
    "PERDL": (0.01, 0.05),
    "Q10": (1.5, 3.0),
    "RDI": (5.0, 20.0),
    "RML": (0.02, 0.04),
    "RMO": (0.003, 0.01),
    "RMR": (0.005, 0.02),
    "RMS": (0.01, 0.02),
    "SPA": (0.0, 0.001),
    "TBASEM": (0.0, 5.0),
    "TEFFMX": (15.0, 25.0)
}
parameter_bounds.extend([parameter_ranges[param] for param in parameters_to_analyze_other])

# Now define the ModelRerunner class
class ModelRerunner:
    def __init__(self, params, wdp, agro, parameters_to_analyze):
        self.params = params
        self.wdp = wdp
        self.agro = agro
        self.parameters_to_analyze = parameters_to_analyze

    def run(self, par_values):
        # Clear any existing overrides
        self.params.clear_override()

        idx = 0  # Index to keep track of parameter positions

        # Set conversion parameters
        if include_conversion:
            conversion_params = ["CVL", "CVO", "CVR", "CVS"]
            for param in conversion_params:
                value = par_values[idx]
                self.params.set_override(param, value)
                idx += 1
        else:
            for param, value in default_values_conversion.items():
                self.params.set_override(param, value)

        # Set other parameters
        for param in parameters_to_analyze_other:
            value = par_values[idx]
            self.params.set_override(param, value)
            idx += 1

        # Run the model
        wofost = Wofost72_WLP_FD(self.params, self.wdp, self.agro)
        wofost.run_till_terminate()
        df_sim = pd.DataFrame(wofost.get_output())
        df_sim.index = pd.to_datetime(df_sim.day)
        return df_sim

# Create an instance of the ModelRerunner
modelrerunner = ModelRerunner(parameters, wdp, agromanagement, parameters_to_analyze)

# Prepare the problem definition for SALib
problem = {
    'num_vars': len(parameters_to_analyze),
    'names': parameters_to_analyze,
    'bounds': parameter_bounds
}

# Generate parameter samples using Morris method
N = 100  # Number of trajectories (adjust based on computational resources)
num_levels = 4  # Number of levels (typically 4 or 6)
param_values = sample(problem, N, num_levels=num_levels, optimal_trajectories=None)

# Run simulations and collect outputs
num_samples = param_values.shape[0]
outputs = np.zeros((num_samples, 4))  # Assuming 4 outputs: LAI_max, DVS_at_LAI_max, TAGP, TWSO

for i in range(num_samples):
    par_values = param_values[i, :]
    df_sim = modelrerunner.run(par_values)
    # Extract outputs
    LAI_max = df_sim['LAI'].max()
    DVS_at_LAI_max = df_sim.loc[df_sim['LAI'].idxmax(), 'DVS']
    TAGP = df_sim['TAGP'].iloc[-1]  # Total above-ground biomass at harvest
    TWSO = df_sim['TWSO'].iloc[-1]  # Total weight storage organ at harvest
    outputs[i, :] = [LAI_max, DVS_at_LAI_max, TAGP, TWSO]
    print(f"Simulation {i+1}/{num_samples} completed.")

# Perform Morris sensitivity analysis
Si_LAI_max = morris.analyze(problem, param_values, outputs[:, 0], conf_level=0.95, print_to_console=False)
Si_DVS_LAI_max = morris.analyze(problem, param_values, outputs[:, 1], conf_level=0.95, print_to_console=False)
Si_TAGP = morris.analyze(problem, param_values, outputs[:, 2], conf_level=0.95, print_to_console=False)
Si_TWSO = morris.analyze(problem, param_values, outputs[:, 3], conf_level=0.95, print_to_console=False)

# Function to display and sort Morris indices
def display_sorted_morris_indices(Si, parameter_names, output_name):
    print(f"\nMorris sensitivity indices for {output_name} (sorted by sensitivity):")
    mu_star = Si['mu_star']
    sigma = Si['sigma']
    sorted_indices = np.argsort(mu_star)[::-1]  # Sort in descending order
    for idx in sorted_indices:
        print(f"{parameter_names[idx]}: mu* = {mu_star[idx]:.4f}, sigma = {sigma[idx]:.4f}")

# Display Morris indices for each output (sorted by sensitivity)
display_sorted_morris_indices(Si_LAI_max, parameters_to_analyze, 'LAI_max')
display_sorted_morris_indices(Si_DVS_LAI_max, parameters_to_analyze, 'DVS_at_LAI_max')
display_sorted_morris_indices(Si_TAGP, parameters_to_analyze, 'TAGP')
display_sorted_morris_indices(Si_TWSO, parameters_to_analyze, 'TWSO')

# Function to visualize individual sensitivity analysis
def plot_individual_sensitivity(Si, parameter_names, output_name):
    mu_star = Si['mu_star']
    sigma = Si['sigma']
    sorted_indices = np.argsort(mu_star)[::-1]  # Sort in descending order
    sorted_mu_star = mu_star[sorted_indices]
    sorted_sigma = sigma[sorted_indices]
    sorted_parameters = [parameter_names[i] for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_parameters, sorted_mu_star, xerr=sorted_sigma, color='skyblue', ecolor='black')
    plt.xlabel('mu* (Mean of absolute EE)')
    plt.title(f'Individual Sensitivity Analysis for {output_name}')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()

# Plot individual sensitivity for each output
plot_individual_sensitivity(Si_LAI_max, parameters_to_analyze, 'LAI_max')
plot_individual_sensitivity(Si_TAGP, parameters_to_analyze, 'TAGP')
plot_individual_sensitivity(Si_TWSO, parameters_to_analyze, 'TWSO')

# Function to calculate combined sensitivity and visualize as a matrix
def plot_combined_sensitivity_matrix(Si_list, parameter_names, output_name):
    num_params = len(parameter_names)
    combined_sensitivities = np.zeros((num_params, num_params))

    for Si in Si_list:
        mu_star = Si['mu_star']
        for i in range(num_params):
            for j in range(num_params):
                combined_sensitivities[i, j] += mu_star[i] + mu_star[j]

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(combined_sensitivities, cmap='viridis')
    plt.xticks(range(num_params), parameter_names, rotation=90)
    plt.yticks(range(num_params), parameter_names)
    plt.colorbar(cax)
    plt.title(f'Combined Sensitivity Matrix for {output_name}')
    plt.show()

# Plot combined sensitivity matrix for each output
plot_combined_sensitivity_matrix([Si_LAI_max], parameters_to_analyze, 'LAI_max')
plot_combined_sensitivity_matrix([Si_TAGP], parameters_to_analyze, 'TAGP')
plot_combined_sensitivity_matrix([Si_TWSO], parameters_to_analyze, 'TWSO')

# Save the sensitivity analysis results
# Combine parameter values and outputs
results = np.hstack((param_values, outputs))
columns = parameters_to_analyze + ['LAI_max', 'DVS_at_LAI_max', 'TAGP', 'TWSO']
df_results = pd.DataFrame(results, columns=columns)

output_file = os.path.join(data_dir, 'morris_sensitivity_analysis_results.csv')
df_results.to_csv(output_file, index=False)
print(f"Sensitivity analysis results saved to {output_file}")

# === Added Code Starts Here ===

# Function to compute combined sensitivity indices
def compute_combined_sensitivity(Si_list, parameter_names):
    num_params = len(parameter_names)
    combined_mu_star = np.zeros(num_params)
    for Si in Si_list:
        combined_mu_star += Si['mu_star']
    combined_mu_star /= len(Si_list)  # Average over all outputs
    return combined_mu_star

# Compute combined sensitivity indices across all outputs
Si_list = [Si_LAI_max, Si_DVS_LAI_max, Si_TAGP, Si_TWSO]
combined_mu_star = compute_combined_sensitivity(Si_list, parameters_to_analyze)

# Sort parameters by combined_mu_star
sorted_indices = np.argsort(combined_mu_star)[::-1]  # Sort in descending order
print("\nCombined Sensitivity Indices (average mu* across all outputs):")
for idx in sorted_indices:
    print(f"{parameters_to_analyze[idx]}: combined mu* = {combined_mu_star[idx]:.4f}")

# Plot combined sensitivity
def plot_combined_sensitivity(combined_mu_star, parameter_names):
    sorted_indices = np.argsort(combined_mu_star)[::-1]  # Sort in descending order
    sorted_mu_star = combined_mu_star[sorted_indices]
    sorted_parameters = [parameter_names[i] for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_parameters, sorted_mu_star, color='skyblue')
    plt.xlabel('Average mu* (Mean of absolute EE across outputs)')
    plt.title('Combined Sensitivity Analysis')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()

# Plot the combined sensitivity
plot_combined_sensitivity(combined_mu_star, parameters_to_analyze)

# Suggest parameters to calibrate based on combined sensitivity
# You can adjust the threshold_percentage or number of parameters as needed
threshold_percentage = 80  # Percentage of total sensitivity to cover
total_mu_star = np.sum(combined_mu_star)
sorted_mu_star = combined_mu_star[sorted_indices]
cumulative_mu_star = np.cumsum(sorted_mu_star)
cumulative_percentage = cumulative_mu_star / total_mu_star * 100

# Determine parameters that contribute to the specified percentage of total sensitivity
num_top_params = np.argmax(cumulative_percentage >= threshold_percentage) + 1

print(f"\nParameters contributing to {threshold_percentage}% of total sensitivity:")
for i in range(num_top_params):
    idx = sorted_indices[i]
    print(f"{parameters_to_analyze[idx]}: combined mu* = {combined_mu_star[idx]:.4f}")

print("\nBased on the combined sensitivity analysis, it is suggested to prioritize the calibration of these parameters.")

# === Added Code Ends Here ===
