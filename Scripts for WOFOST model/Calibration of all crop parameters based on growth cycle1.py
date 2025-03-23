import os
import sys
import matplotlib
# Use the Agg backend to avoid memory issues with TkAgg
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nlopt
import traceback
import logging  # Import logging module
import gc  # Garbage collector
import itertools  # For grid search combinations
import progressbar  # For displaying progress

import pcse
from pcse.models import Wofost72_WLP_FD
from pcse.base import ParameterProvider
from pcse.db import NASAPowerWeatherDataProvider
from pcse.fileinput import CABOFileReader, YAMLAgroManagementReader
from pcse.util import WOFOST72SiteDataProvider

# Adjust logging configuration to prevent file access errors
# Remove all handlers associated with the root logger object
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Reconfigure logging to output to console only
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Set the logging level for PCSE to CRITICAL to suppress less severe messages
logging.getLogger('pcse').setLevel(logging.CRITICAL)

# Use matplotlib style if available
if hasattr(matplotlib, 'style'):
    matplotlib.style.use("ggplot")

print("This script was built with:")
print(f"Python version: {sys.version}")
print(f"PCSE version: {pcse.__version__}")

# Define data directory
data_dir = r'C:\Users\user\Documents\ind set 2\ind test for majro crops\Test 2'  # Update this path as needed

# Read crop data
cropfile = os.path.join(data_dir, 'practice_sugarbeet.crop')
try:
    cropdata = CABOFileReader(cropfile)
    print("Crop Data:")
    print(cropdata)
except Exception as e:
    print(f"Failed to read crop data: {e}")
    sys.exit(1)

# Read soil data
soilfile = os.path.join(data_dir, 'Coarse.soil')
try:
    soildata = CABOFileReader(soilfile)
    print("Soil Data:")
    print(soildata)
except Exception as e:
    print(f"Failed to read soil data: {e}")
    sys.exit(1)

# Define site data
sitedata = WOFOST72SiteDataProvider(WAV=100, CO2=360)
print("Site Data:")
print(sitedata)

# Combine parameters
parameters = ParameterProvider(cropdata=cropdata, soildata=soildata, sitedata=sitedata)

# Read agromanagement data
agromanagement_file = os.path.join(data_dir, 'agrofile_npk.agro')
try:
    agromanagement = YAMLAgroManagementReader(agromanagement_file)
    print("Agromanagement Data:")
    print(agromanagement)
except Exception as e:
    print(f"Failed to read agromanagement data: {e}")
    sys.exit(1)

# Define weather data
try:
    wdp = NASAPowerWeatherDataProvider(latitude=54.0231470051597, longitude=12.8092511197906)
    print("Weather Data Provider:")
    print(wdp)
except Exception as e:
    print(f"Failed to initialize weather data provider: {e}")
    sys.exit(1)

# Load observed partitioning data from Excel file
observed_partitioning_file = os.path.join(data_dir, 'Observed partitioning data.xlsx')
try:
    observed_partitioning_df = pd.read_excel(observed_partitioning_file, parse_dates=['Date'], index_col='Date')
    print("Observed Partitioning Data:")
    print(observed_partitioning_df.head())
except Exception as e:
    print(f"Failed to read observed partitioning data: {e}")
    sys.exit(1)

# Define the variables to be used throughout the script
variables = ['LAI', 'TAGP', 'TWSO', 'TWLV', 'TWST', 'TWRT', 'SM']

# Check for NaN values and handle them
if observed_partitioning_df.isnull().values.any():
    print("Warning: Observed data contains NaN values.")
    print(observed_partitioning_df.isnull().sum())
    # Drop rows with NaN values in the relevant columns
    df_observations = observed_partitioning_df.dropna(subset=variables, how='all')
    if df_observations.empty:
        print("Error: No valid observed data after dropping NaN values.")
        sys.exit(1)
else:
    df_observations = observed_partitioning_df

print("Filtered Observed Partitioning Data:")
print(df_observations.head())

# Ensure that the index of observations is sorted
df_observations.sort_index(inplace=True)

# Switches to control which parameters to optimize
optimize_partitioning = False
optimize_FRTB = True
optimize_conversion = True
optimize_AMAXTB = False
optimize_other_params = False  # Change to True if you want to optimize other parameters

# Switches to control which parameters to include in the grid search
grid_search_partitioning = False
grid_search_FRTB = False
grid_search_conversion = True
grid_search_AMAXTB = False
grid_search_other_params = False  # Set to True for grid search on other parameters

# Define DVS stages for partitioning parameters
DVS_stages = [0.00, 0.54, 1.00, 1.20, 1.42, 1.51, 1.71, 2.00]
# Initial FLTB and FSTB values
FLTB_values_initial = [1.000000, 0.594312, 0.449823, 0.418516, 0.020000, 0.000000, 0.000000, 0.000000]
FSTB_values_initial = [0.000000, 0.405688, 0.550177, 0.193750, 0.140000, 0.010000, 0.000000, 0.000000]

# For FRTB
DVS_stages_FRTB = [0.00, 0.54, 1.00, 1.20, 2.00]
FRTB_values_initial = [0.025000, 0.625000, 0.200000, 0.000000, 0.000000]

# For AMAXTB
DVS_stages_AMAXTB = [0.00, 1.50, 1.90, 2.00]
AMAXTB_values_initial = [35.00, 35.00, 0.00, 0.00]

# Conversion parameters initial values
default_values_conversion = {
    "CVL": 0.458,
    "CVO": 0.951,
    "CVR": 0.951,
    "CVS": 0.831
}

# Extract the parameters to analyze based on switches
parameter_switches = {
    "TSUMEM": False,
    "TSUM1": True,
    "TSUM2": False,
    "TDWI": False,
    "SPAN": False,
    "RGRLAI": True,
    "DEPNR": False,
    "RDMCR": True,
    "CFET": False,
    "DLC": False,
    "DLO": False,
    "PERDL": False,
    "Q10": False,
    "RDI": False,
    "RML": False,
    "RMO": False,
    "RMR": False,
    "RMS": True,
    "SPA": False,
    "TBASEM": False,
    "TEFFMX": False
}

# Create function to generate data lists
def create_afgen_data(dvs_stages, values):
    afgen_list = [float(item) for pair in zip(dvs_stages, values) for item in pair]
    return afgen_list

# Set initial parameters in the model
FLTB_afgen_data_initial = create_afgen_data(DVS_stages, FLTB_values_initial)
FSTB_afgen_data_initial = create_afgen_data(DVS_stages, FSTB_values_initial)
FOTB_values_initial = [1.0 - fltb - fstb for fltb, fstb in zip(FLTB_values_initial, FSTB_values_initial)]
FOTB_afgen_data_initial = create_afgen_data(DVS_stages, FOTB_values_initial)
parameters.set_override("FLTB", FLTB_afgen_data_initial)
parameters.set_override("FSTB", FSTB_afgen_data_initial)
parameters.set_override("FOTB", FOTB_afgen_data_initial)

# Set initial FRTB
FRTB_afgen_data_initial = create_afgen_data(DVS_stages_FRTB, FRTB_values_initial)
parameters.set_override("FRTB", FRTB_afgen_data_initial)

# Set initial AMAXTB
AMAXTB_afgen_data_initial = create_afgen_data(DVS_stages_AMAXTB, AMAXTB_values_initial)
parameters.set_override("AMAXTB", AMAXTB_afgen_data_initial)

# Set conversion parameters
for param, value in default_values_conversion.items():
    parameters.set_override(param, value)

# Set other parameters
defaults = {}
for param in parameter_switches.keys():
    if param in cropdata:
        defaults[param] = cropdata[param]
    elif param in soildata:
        defaults[param] = soildata[param]
    elif param in sitedata:
        defaults[param] = sitedata[param]
    else:
        defaults[param] = parameters[param]  # Fetch from parameters if available

for param, value in defaults.items():
    parameters.set_override(param, value)

# Run initial WOFOST model with default parameters
try:
    wofost_initial = Wofost72_WLP_FD(parameters, wdp, agromanagement)
    wofost_initial.run_till_terminate()
    df_initial = pd.DataFrame(wofost_initial.get_output())
    df_initial.index = pd.to_datetime(df_initial.day)
    print("Initial WOFOST Output:")
    print(df_initial.tail())
except Exception as e:
    print(f"Initial model run failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Define parameters to optimize based on switches
parameters_to_optimize = []

if optimize_partitioning:
    # Include FLTB and FSTB parameters
    for i in range(len(DVS_stages)):
        parameters_to_optimize.append(f"FLTB_{i+1}")
    for i in range(len(DVS_stages)):
        parameters_to_optimize.append(f"FSTB_{i+1}")

if optimize_FRTB:
    # Include FRTB parameters
    for i in range(len(DVS_stages_FRTB)):
        parameters_to_optimize.append(f"FRTB_{i+1}")

if optimize_conversion:
    # Include conversion parameters
    parameters_to_optimize.extend(["CVL", "CVO", "CVR", "CVS"])

if optimize_AMAXTB:
    # Include AMAXTB parameters
    for i in range(len(DVS_stages_AMAXTB)):
        parameters_to_optimize.append(f"AMAXTB_{i+1}")

if optimize_other_params:
    # Include other parameters
    parameters_to_optimize.extend([param for param, switch in parameter_switches.items() if switch])

print("Parameters to optimize:", parameters_to_optimize)

# Create default values for parameters
default_values = []

if optimize_partitioning:
    default_values.extend(FLTB_values_initial)
    default_values.extend(FSTB_values_initial)

if optimize_FRTB:
    default_values.extend(FRTB_values_initial)

if optimize_conversion:
    default_values.extend([default_values_conversion[param] for param in ["CVL", "CVO", "CVR", "CVS"]])

if optimize_AMAXTB:
    default_values.extend(AMAXTB_values_initial)

if optimize_other_params:
    default_values.extend([defaults[param] for param in parameters_to_optimize if param in defaults])

# Now define parameter bounds
parameter_bounds = []

if optimize_partitioning:
    # Bounds between 0 and 1 for FLTB and FSTB
    parameter_bounds.extend([(0.0, 1.0)] * len(DVS_stages) * 2)

if optimize_FRTB:
    # Bounds between 0 and 1 for FRTB
    parameter_bounds.extend([(0.0, 1.0)] * len(DVS_stages_FRTB))

if optimize_conversion:
    # Bounds for conversion parameters
    conversion_bounds = {
        "CVL": (0.30, 0.50),
        "CVO": (0.85, 0.985),
        "CVR": (0.90, 0.99),
        "CVS": (0.80, 0.90)
    }
    parameter_bounds.extend([conversion_bounds[param] for param in ["CVL", "CVO", "CVR", "CVS"]])

if optimize_AMAXTB:
    # Bounds for AMAXTB values
    parameter_bounds.extend([(0.0, 35.0)] * len(DVS_stages_AMAXTB))

if optimize_other_params:
    parameter_ranges = {
        "TSUMEM": (500, 650),
        "TSUM1": (500, 1000),
        "TSUM2": (800, 1000),
        "TDWI": (30, 60),
        "SPAN": (20, 45),
        "RGRLAI": (0.0080, 0.011),
        "DEPNR": (3.5, 4.5),
        "RDMCR": (110, 120),
        "CFET": (0.5, 1.5),
        "DLC": (-99.0, 15.0),
        "DLO": (-99.0, 18.0),
        "PERDL": (0.03, 0.04),
        "Q10": False,
        "RDI": False,
        "RML": False,
        "RMO": False,
        "RMR": False,
        "RMS": (0.01, 0.02),
        "SPA": False,
        "TBASEM": False,
        "TEFFMX": False
    }
    parameter_bounds.extend([parameter_ranges[param] for param in parameters_to_optimize if param in parameter_ranges])

# Extract lower and upper bounds
lower_bounds = [bound[0] for bound in parameter_bounds]
upper_bounds = [bound[1] for bound in parameter_bounds]

# Define the ModelRerunner class (ensure only one definition exists)
class ModelRerunner:
    def __init__(self, params, wdp, agro, DVS_stages, DVS_stages_FRTB, DVS_stages_AMAXTB, parameters_to_optimize):
        self.params = params
        self.wdp = wdp
        self.agro = agro
        self.DVS_stages = DVS_stages
        self.DVS_stages_FRTB = DVS_stages_FRTB
        self.DVS_stages_AMAXTB = DVS_stages_AMAXTB
        self.parameters_to_optimize = parameters_to_optimize
        self.adjusted_parameters = {}  # To store adjusted parameters

    def run(self, par_values):
        # Clear any existing overrides
        self.params.clear_override()

        idx = 0  # Index to keep track of parameter positions

        # Set partitioning parameters
        if optimize_partitioning:
            num_DVS = len(self.DVS_stages)
            FLTB_values = list(par_values[idx:idx + num_DVS])
            idx += num_DVS
            FSTB_values = list(par_values[idx:idx + num_DVS])
            idx += num_DVS

            # Enforce constraints and compute FOTB_values
            FOTB_values = []
            for i in range(len(self.DVS_stages)):
                DVS = self.DVS_stages[i]
                FLTB = FLTB_values[i]
                FSTB = FSTB_values[i]

                if DVS <= 1.00:
                    # FOTB is 0
                    FOTB = 0.0
                    total = FLTB + FSTB
                    if total > 0:
                        # Normalize FLTB and FSTB to sum to 1
                        FLTB /= total
                        FSTB /= total
                    else:
                        # Default partitioning if total is zero
                        FLTB = 0.5
                        FSTB = 0.5
                elif DVS == 2.00:
                    # For DVS = 2.00, FLTB and FSTB are 0, FOTB is 1
                    FLTB = 0.0
                    FSTB = 0.0
                    FOTB = 1.0
                else:
                    # For other DVS, calculate FOTB = 1 - FLTB - FSTB
                    FOTB = 1.0 - FLTB - FSTB
                    # Ensure FOTB is within [0,1]
                    FOTB = min(max(FOTB, 0.0), 1.0)
                    total = FLTB + FSTB + FOTB
                    if abs(total - 1.0) > 1e-6:
                        # Adjust to ensure sum to 1
                        FLTB /= total
                        FSTB /= total
                        FOTB /= total

                # Store the adjusted values
                FLTB_values[i] = FLTB
                FSTB_values[i] = FSTB
                FOTB_values.append(FOTB)

                # Store adjusted parameters
                self.adjusted_parameters[f"FLTB_{i+1}"] = FLTB
                self.adjusted_parameters[f"FSTB_{i+1}"] = FSTB
                self.adjusted_parameters[f"FOTB_{i+1}"] = FOTB

            # Create data lists
            FLTB_afgen_data = create_afgen_data(self.DVS_stages, FLTB_values)
            FSTB_afgen_data = create_afgen_data(self.DVS_stages, FSTB_values)
            FOTB_afgen_data = create_afgen_data(self.DVS_stages, FOTB_values)

            # Override parameters
            self.params.set_override("FLTB", FLTB_afgen_data)
            self.params.set_override("FSTB", FSTB_afgen_data)
            self.params.set_override("FOTB", FOTB_afgen_data)
        else:
            # Use initial values
            self.params.set_override("FLTB", FLTB_afgen_data_initial)
            self.params.set_override("FSTB", FSTB_afgen_data_initial)
            self.params.set_override("FOTB", FOTB_afgen_data_initial)
            # Store initial parameters
            for i in range(len(self.DVS_stages)):
                self.adjusted_parameters[f"FLTB_{i+1}"] = FLTB_values_initial[i]
                self.adjusted_parameters[f"FSTB_{i+1}"] = FSTB_values_initial[i]
                self.adjusted_parameters[f"FOTB_{i+1}"] = FOTB_values_initial[i]

        # Set FRTB parameters
        if optimize_FRTB:
            num_DVS_FRTB = len(self.DVS_stages_FRTB)
            FRTB_values = par_values[idx:idx + num_DVS_FRTB]
            idx += num_DVS_FRTB
            # Enforce bounds [0,1]
            FRTB_values = [min(max(val, 0.0), 1.0) for val in FRTB_values]
            FRTB_afgen_data = create_afgen_data(self.DVS_stages_FRTB, FRTB_values)
            self.params.set_override("FRTB", FRTB_afgen_data)
            # Store adjusted parameters
            for i in range(num_DVS_FRTB):
                self.adjusted_parameters[f"FRTB_{i+1}"] = FRTB_values[i]
        else:
            self.params.set_override("FRTB", FRTB_afgen_data_initial)
            for i in range(len(self.DVS_stages_FRTB)):
                self.adjusted_parameters[f"FRTB_{i+1}"] = FRTB_values_initial[i]

        # Set conversion parameters
        if optimize_conversion:
            conversion_params = ["CVL", "CVO", "CVR", "CVS"]
            for param in conversion_params:
                value = par_values[idx]
                self.params.set_override(param, value)
                self.adjusted_parameters[param] = value
                idx += 1
        else:
            for param, value in default_values_conversion.items():
                self.params.set_override(param, value)
                self.adjusted_parameters[param] = value

        # Set AMAXTB parameters
        if optimize_AMAXTB:
            num_DVS_AMAXTB = len(self.DVS_stages_AMAXTB)
            AMAXTB_values = list(par_values[idx:idx + num_DVS_AMAXTB])
            idx += num_DVS_AMAXTB

            # Enforce non-increasing rule for AMAXTB
            for i in range(1, len(AMAXTB_values)):
                if AMAXTB_values[i] > AMAXTB_values[i - 1]:
                    AMAXTB_values[i] = AMAXTB_values[i - 1]

            AMAXTB_afgen_data = create_afgen_data(self.DVS_stages_AMAXTB, AMAXTB_values)
            self.params.set_override("AMAXTB", AMAXTB_afgen_data)
            # Store adjusted parameters
            for i in range(num_DVS_AMAXTB):
                self.adjusted_parameters[f"AMAXTB_{i+1}"] = AMAXTB_values[i]
        else:
            self.params.set_override("AMAXTB", AMAXTB_afgen_data_initial)
            for i in range(len(self.DVS_stages_AMAXTB)):
                self.adjusted_parameters[f"AMAXTB_{i+1}"] = AMAXTB_values_initial[i]

        # Set other parameters
        for param in self.parameters_to_optimize:
            # Skip parameters already handled
            if param.startswith("FLTB_") or param.startswith("FSTB_") or param.startswith("FOTB_") \
               or param.startswith("FRTB_") or param.startswith("AMAXTB_") \
               or param in ["CVL", "CVO", "CVR", "CVS"]:
                continue
            if param in defaults:
                value = par_values[idx]
                self.params.set_override(param, value)
                self.adjusted_parameters[param] = value
                idx += 1

        # Run the model
        try:
            wofost = Wofost72_WLP_FD(self.params, self.wdp, self.agro)
            wofost.run_till_terminate()
            df_sim = pd.DataFrame(wofost.get_output())
            df_sim.index = pd.to_datetime(df_sim.day)
            return df_sim
        except Exception as e:
            print(f"Model run failed with error: {e}")
            traceback.print_exc()
            return None  # Return None if the model fails

# Define the ObjectiveFunctionCalculator class
class ObjectiveFunctionCalculator:
    def __init__(self, params, wdp, agro, observations, DVS_stages, DVS_stages_FRTB, DVS_stages_AMAXTB, parameters_to_optimize):
        self.modelrerunner = ModelRerunner(params, wdp, agro, DVS_stages, DVS_stages_FRTB, DVS_stages_AMAXTB, parameters_to_optimize)
        self.df_observations = observations
        self.n_calls = 0
        self.df_simulations = None  # To store the last simulation results
        self.best_parameters = None  # To store the best parameters found
        self.minimum_error = np.inf
        self.best_call = None  # To store the function call number of the lowest error
        self.parameters_to_optimize = parameters_to_optimize  # Store the parameter names

    def __call__(self, par_values, grad=None):
        self.n_calls += 1
        print(f"Function call {self.n_calls}")

        # Run the model and collect output
        self.df_simulations = self.modelrerunner.run(par_values)

        # Print adjusted parameters and their values, including FOTB
        print("Adjusted Parameters for this run:")
        adjusted_params = self.modelrerunner.adjusted_parameters
        for param, value in adjusted_params.items():
            if param.startswith("FLTB_") or param.startswith("FSTB_") or param.startswith("FOTB_"):
                idx = int(param.split("_")[1]) - 1
                dvs = self.modelrerunner.DVS_stages[idx]
                print(f"{param} (DVS {dvs}): {value:.6f}")
            elif param.startswith("FRTB_"):
                idx = int(param.split("_")[1]) - 1
                dvs = self.modelrerunner.DVS_stages_FRTB[idx]
                print(f"{param} (DVS {dvs}): {value:.6f}")
            elif param.startswith("AMAXTB_"):
                idx = int(param.split("_")[1]) - 1
                dvs = self.modelrerunner.DVS_stages_AMAXTB[idx]
                print(f"{param} (DVS {dvs}): {value:.6f}")
            else:
                print(f"{param}: {value:.6f}")
        print("\n")  # Add an extra newline for readability

        if self.df_simulations is None:
            # Model run failed, return a high error
            return 1e6

        # Merge simulated and observed data
        df_merge = pd.merge(self.df_observations, self.df_simulations, left_index=True, right_index=True, how='inner')
        if df_merge.empty:
            # If no overlapping dates, try to use the last date from observations
            last_obs_date = self.df_observations.index[-1]
            if last_obs_date in self.df_simulations.index:
                df_merge = pd.merge(self.df_observations.loc[[last_obs_date]], self.df_simulations, left_index=True, right_index=True, how='inner')
            else:
                # If last observed date is not in simulation, find the closest date
                closest_date = self.df_simulations.index.get_loc(last_obs_date, method='nearest')
                df_sim_closest = self.df_simulations.iloc[[closest_date]]
                df_merge = pd.merge(self.df_observations.loc[[last_obs_date]], df_sim_closest, left_index=True, right_index=True, how='inner')

        if df_merge.empty:
            print("Merged DataFrame is empty.")
            return 1e6  # Return a high error if no overlapping dates

        errors = []
        variables = ['LAI', 'TAGP', 'TWSO', 'TWLV', 'TWST', 'TWRT', 'SM']
        weights = {
            'LAI': 1.0,
            'TAGP': 2.0,
            'TWSO': 2.0,
            'TWLV': 1.0,
            'TWST': 1.0,
            'TWRT': 1.0,
            'SM': 1.0  # Adjust the weight for SM as needed
        }

        for var in variables:
            if var + '_x' in df_merge.columns and var + '_y' in df_merge.columns:
                observed = df_merge[f'{var}_x']
                simulated = df_merge[f'{var}_y']
                error = np.sqrt(np.mean((observed - simulated) ** 2))
                if not np.isfinite(error):
                    error = 1e6  # Assign a large penalty
                errors.append(weights[var] * error)
            else:
                print(f"Variable {var} not found in merged data.")

        if not errors:
            print("No variables to compute error.")
            return 1e6

        mean_error = np.mean(errors)
        if not np.isfinite(mean_error):
            mean_error = 1e6

        # Update the best parameters if the current error is the lowest
        if mean_error < self.minimum_error:
            self.minimum_error = mean_error
            self.best_parameters = par_values.copy()
            self.best_call = self.n_calls
            print(f"New best parameters found with error: {self.minimum_error:.6f}")

        return mean_error

# Create an instance of the ObjectiveFunctionCalculator
objfunc_calculator = ObjectiveFunctionCalculator(
    parameters, wdp, agromanagement, df_observations,
    DVS_stages, DVS_stages_FRTB, DVS_stages_AMAXTB, parameters_to_optimize
)

# --- Start of Grid Search ---

# Define switches for grid search parameters
grid_search_parameters = []

if grid_search_partitioning:
    # Include FLTB and FSTB parameters
    for i in range(len(DVS_stages)):
        grid_search_parameters.append(f"FLTB_{i+1}")
    for i in range(len(DVS_stages)):
        grid_search_parameters.append(f"FSTB_{i+1}")

if grid_search_FRTB:
    # Include FRTB parameters
    for i in range(len(DVS_stages_FRTB)):
        grid_search_parameters.append(f"FRTB_{i+1}")

if grid_search_conversion:
    # Include conversion parameters
    grid_search_parameters.extend(["CVL", "CVO", "CVR", "CVS"])

if grid_search_AMAXTB:
    # Include AMAXTB parameters
    for i in range(len(DVS_stages_AMAXTB)):
        grid_search_parameters.append(f"AMAXTB_{i+1}")

if grid_search_other_params:
    # Include other parameters
    grid_search_parameters.extend([param for param, switch in parameter_switches.items() if switch])

print("Parameters to include in grid search:", grid_search_parameters)

# Create a list of indices for the grid search parameters
grid_parameter_indices = [parameters_to_optimize.index(param) for param in grid_search_parameters]

# Define parameter ranges and steps for grid search
parameter_ranges_grid = {}
for param in grid_search_parameters:
    idx = parameters_to_optimize.index(param)
    lb, ub = parameter_bounds[idx]
    # Define a small number of steps to keep computational load manageable
    if 'FLTB' in param or 'FSTB' in param or 'FRTB' in param:
        parameter_ranges_grid[param] = np.linspace(lb, ub, 3)  # 3 values between lb and ub
    elif param in ["CVL", "CVO", "CVR", "CVS"]:
        parameter_ranges_grid[param] = np.linspace(lb, ub, 3)
    elif 'AMAXTB' in param:
        parameter_ranges_grid[param] = np.linspace(lb, ub, 3)
    else:
        # For other parameters, use 3 steps
        parameter_ranges_grid[param] = np.linspace(lb, ub, 3)

# Generate all combinations using itertools.product
all_combinations = list(itertools.product(*parameter_ranges_grid.values()))
nruns = len(all_combinations)

if nruns == 0:
    raise ValueError("No parameter combinations to run. Check your parameter ranges and steps.")

print(f"Total grid search runs: {nruns}")

# Initialize variables to track the best parameters
lowest_error = np.inf
best_grid_values = None

# Use progressbar to track progress
widgets = [progressbar.Percentage(), progressbar.Bar()]
bar = progressbar.ProgressBar(widgets=widgets, maxval=nruns).start()
ncalls = 0

for values in all_combinations:
    # Create a copy of default_values to modify
    par_values = default_values.copy()
    # Update the selected parameters
    for idx, value in zip(grid_parameter_indices, values):
        par_values[idx] = value
    # Evaluate the objective function
    error = objfunc_calculator(par_values)
    ncalls += 1
    if error < lowest_error:
        best_grid_values = par_values.copy()
        lowest_error = error
    bar.update(ncalls)

bar.finish()

# Ensure best_grid_values is not None before proceeding
if best_grid_values is not None:
    print("\nBest parameters from grid search:", best_grid_values)
    print("Minimum error from grid search =", lowest_error)
    print("Function calls during grid search:", ncalls)
else:
    print("No optimal values found during the grid search.")
    sys.exit(1)

# --- End of Grid Search ---

# Start optimizer with the SUBPLEX algorithm for the parameters
opt = nlopt.opt(nlopt.LN_SBPLX, len(parameters_to_optimize))
# Assign the objective function calculator
opt.set_min_objective(objfunc_calculator)
# Set lower and upper bounds
opt.set_lower_bounds(lower_bounds)
opt.set_upper_bounds(upper_bounds)
# Set initial step size
initial_step = [(ub - lb) * 0.1 for lb, ub in zip(lower_bounds, upper_bounds)]
opt.set_initial_step(initial_step)
# Maximum number of evaluations allowed
opt.set_maxeval(200)
# Relative tolerance for convergence
opt.set_ftol_rel(1e-4)

# Start the optimization with the best parameters from grid search
x0 = best_grid_values
try:
    x = opt.optimize(x0)
except Exception as e:
    print(f"Optimization failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Retrieve the best parameters found during optimization
best_parameters = objfunc_calculator.best_parameters

print(f"\nLowest error achieved at function call number: {objfunc_calculator.best_call}")

# Rerun with the best parameters found
error = objfunc_calculator(best_parameters)
df_simulations = objfunc_calculator.df_simulations

# Also run with the initial parameters (first guess)
error_initial = objfunc_calculator(default_values)
df_initial_simulation = objfunc_calculator.df_simulations

# Adjust pandas display options to print all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)  # Allow pandas to use full width

# Print a separator to make the output clearer
print("\n" + "="*80 + "\n")

# Print the final optimized model output
desired_columns = ['day', 'DVS', 'LAI', 'TAGP', 'TWSO', 'TWLV', 'TWST', 'TWRT', 'TRA', 'RD', 'SM', 'WWLOW']
available_columns = [col for col in desired_columns if col in df_simulations.columns]
print("Final Optimized Model Output:")
print(df_simulations[available_columns].to_string(index=False))

# Close any existing figures
plt.close('all')

# Downsample data for plotting if necessary
df_observations_sampled = df_observations[variables].resample('D').mean()
df_initial_simulation_sampled = df_initial_simulation[variables].resample('D').mean()
df_simulations_sampled = df_simulations[variables].resample('D').mean()

# Plot the model results against observed partitioning data
variables = ['LAI', 'TAGP', 'TWSO', 'TWLV', 'TWST', 'TWRT', 'SM']
colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y']  # Added 'y' for yellow
fig, axes = plt.subplots(len(variables), 1, figsize=(10, len(variables) * 2), sharex=True)

for i, (var, color) in enumerate(zip(variables, colors)):
    axes[i].plot_date(df_observations_sampled.index, df_observations_sampled[var], f'{color}o', label=f"Observed {var}")
    axes[i].plot_date(df_initial_simulation_sampled.index, df_initial_simulation_sampled[var], f'{color}--', label=f"First Guess Simulated {var}")
    axes[i].plot_date(df_simulations_sampled.index, df_simulations_sampled[var], f'{color}-', label=f"Simulated {var} (Optimized)")
    axes[i].set_title(f"Observed vs. Simulated {var}")
    axes[i].set_ylabel(var)
    axes[i].legend()

axes[-1].set_xlabel("Date")

# Save the plot to a file instead of displaying it
plt.tight_layout()

# Ensure the output directory exists
output_dir = data_dir  # Use the data directory or specify another path
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the plot to the specified directory
plot_path = os.path.join(output_dir, 'model_results.png')
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Plot saved to '{plot_path}'")

# Optional: Keep the script alive until user input
input("Press Enter to exit...")

# Clean up to free memory
del df_initial, df_merge, df_observations_sampled, df_initial_simulation_sampled, df_simulations_sampled
gc.collect()
