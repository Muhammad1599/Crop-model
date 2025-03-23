import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import numpy as np
import nlopt
from itertools import product
import progressbar

import pcse
from pcse.models import Wofost72_PP
from pcse.base import ParameterProvider
from pcse.db import NASAPowerWeatherDataProvider
from pcse.fileinput import CABOFileReader, YAMLAgroManagementReader
from pcse.util import WOFOST72SiteDataProvider

# Use matplotlib style if available
if hasattr(matplotlib, 'style'):
    matplotlib.style.use("ggplot")

print("This notebook was built with:")
print(f"python version: {sys.version}")
print(f"PCSE version: {pcse.__version__}")

# Define data directory
data_dir = r'C:\Users\user\Documents\ind set 2'

# Read crop data
cropfile = os.path.join(data_dir, 'practice_sugarbeet.crop')
cropdata = CABOFileReader(cropfile)
print(cropdata)

# Read soil data
soilfile = os.path.join(data_dir, 'Fine sandy loam.soil')
soildata = CABOFileReader(soilfile)

# Define site data
sitedata = WOFOST72SiteDataProvider(WAV=100, NAVAILI=120, PAVAILI=22, KAVAILI=120)
print(sitedata)

# Combine parameters
parameters = ParameterProvider(cropdata=cropdata, soildata=soildata, sitedata=sitedata)

# Read agromanagement data
agromanagement_file = os.path.join(data_dir, 'agrofile_npk.agro')
agromanagement = YAMLAgroManagementReader(agromanagement_file)
print(agromanagement)

# Define weather data
wdp = NASAPowerWeatherDataProvider(latitude=49.6180225626442, longitude=8.04017233970333)
print(wdp)

# Run initial WOFOST model
wofost = Wofost72_PP(parameters, wdp, agromanagement)
wofost.run_till_terminate()
df = pd.DataFrame(wofost.get_output())
df.index = pd.to_datetime(df.day)
df.tail()

# Load observed LAI data from Excel file
observed_lai_file = r'C:\Users\user\Documents\ind set 2\observed_lai.xlsx'
observed_lai_df = pd.read_excel(observed_lai_file, parse_dates=['Date'], index_col='Date')
print(observed_lai_df)

# Get observed LAI (assuming 'LAI' is the column name and the index is date)
df_observations = observed_lai_df

# Plot observed LAI
fig, axes = plt.subplots(figsize=(12, 8))
axes.plot_date(df_observations.index, df_observations['LAI'])
r = axes.set_title("Observed LAI")

# Define the ModelRerunner class
class ModelRerunner(object):
    """Reruns a given model with different values of parameters.
    
    Returns a pandas DataFrame with simulation results of the model with given
    parameter values.
    """
    
    def __init__(self, params, wdp, agro, parameters_to_optimize):
        self.params = params
        self.wdp = wdp
        self.agro = agro
        self.parameters_to_optimize = parameters_to_optimize
        
    def __call__(self, par_values):
        # Check if correct number of parameter values were provided
        if len(par_values) != len(self.parameters_to_optimize):
            msg = "Optimizing %i parameters, but only % values were provided!" % (len(self.parameters_to_optimize), len(par_values))
            raise RuntimeError(msg)
        # Clear any existing overrides
        self.params.clear_override()
        # Set overrides for the new parameter values
        for parname, value in zip(self.parameters_to_optimize, par_values):
            self.params.set_override(parname, value)
        # Run the model with given parameter values
        wofost = Wofost72_PP(self.params, self.wdp, self.agro)
        wofost.run_till_terminate()
        df = pd.DataFrame(wofost.get_output())
        df.index = pd.to_datetime(df.day)
        return df

# Define the ObjectiveFunctionCalculator class
class ObjectiveFunctionCalculator(object):
    """Computes the objective function.
    
    This class runs the simulation model with given parameter values and returns the objective
    function as the sum of squared difference between observed and simulated LAI.
    """
    
    def __init__(self, params, wdp, agro, observations, parameters_to_optimize):
        self.modelrerunner = ModelRerunner(params, wdp, agro, parameters_to_optimize)
        self.df_observations = observations
        self.n_calls = 0
       
    def __call__(self, par_values, grad=None):
        """Runs the model and computes the objective function for given par_values.
        
        The input parameter 'grad' must be defined in the function call, but is only
        required for optimization methods where analytical gradients can be computed.
        """
        self.n_calls += 1
        print(".", end="")
        # Run the model and collect output
        self.df_simulations = self.modelrerunner(par_values)
        # Compute the differences by subtracting the DataFrames
        # Note that the dataframes automatically join on the index (dates) and column names
        df_differences = self.df_simulations - self.df_observations
        # Compute the RMSE on the LAI column
        obj_func = np.sqrt(np.mean(df_differences.LAI**2))
        return obj_func

# Define the parameter switches
parameter_switches = {
    "TSUMEM": True,
    "TSUM1": True,
    "TSUM2": True,
    "TDWI": True,
    "LAIEM": True,
    "SPAN": True,
    "RGRLAI": True,
    "CFET": False,
    "DEPNR": False,
    "RDMCR": True
}

# Extract the parameters to optimize based on switches
parameters_to_optimize = [param for param, switch in parameter_switches.items() if switch]
print("Parameters to optimize:", parameters_to_optimize)

# Create an instance of the ObjectiveFunctionCalculator
objfunc_calculator = ObjectiveFunctionCalculator(parameters, wdp, agromanagement, df_observations, parameters_to_optimize)

# Define default values for all parameters
defaults = {param: cropdata[param] for param in parameter_switches.keys()}
default_values = [defaults[param] for param in parameters_to_optimize]

error = objfunc_calculator(default_values)
print("Objective function value with default parameters (%s): %s" % (default_values, error))

lowest_error = np.inf
best_values = None

# Define parameter ranges and steps for optimization
parameter_ranges = {
    "TSUMEM": (100, 150),
    "TSUM1": (1400, 1500),
    "TSUM2": (600, 1000),
    "TDWI": [200, 300],
    "LAIEM": (0.1, 0.20),
    "SPAN": [25, 40],
    "RGRLAI": (0, 0.009),
    "CFET": (0, 1),
    "DEPNR": (4.5, 5),
    "RDMCR": (100, 150)
}
steps = {
    "TSUMEM": 150,
    "TSUM1": 800,
    "TSUM2": 1200,
    "TDWI": 100,
    "LAIEM": 0.1,
    "SPAN": 50,
    "RGRLAI": 0.05,
    "CFET": 0.8,
    "DEPNR": 10,
    "RDMCR": 50
}

parameter_values = {key: np.arange(*parameter_ranges[key], step) for key, step in steps.items() if key in parameters_to_optimize}
for param, values in parameter_values.items():
    print(f"{param} values: {values}")

all_combinations = list(product(*parameter_values.values()))
nruns = len(all_combinations)

if nruns == 0:
    raise ValueError("No parameter combinations to run. Check your parameter ranges and steps.")

widgets = [progressbar.Percentage(), progressbar.Bar()]
bar = progressbar.ProgressBar(widgets=widgets, maxval=nruns).start()
ncalls = 0

for values in all_combinations:
    ncalls += 1
    error = objfunc_calculator(list(values))
    if error < lowest_error:
        best_values = list(values)
        lowest_error = error
    bar.update(ncalls)

bar.finish()

# Ensure best_values is not None before printing
if best_values is not None:
    print("\nOptimum at parameters: %s" % best_values)
    print("Minimum value =", lowest_error)
    print("With %i function calls" % ncalls)
else:
    print("No optimal values found during the parameter sweep.")

# Initialize the ObjectiveFunctionCalculator with new parameters if needed
objfunc_calculator = ObjectiveFunctionCalculator(parameters, wdp, agromanagement, df_observations, parameters_to_optimize)

# Start optimizer with the SUBPLEX algorithm for the parameters
opt = nlopt.opt(nlopt.LN_SBPLX, len(parameters_to_optimize))
# Assign the objective function calculator
opt.set_min_objective(objfunc_calculator)
# Lower bounds of parameters values
opt.set_lower_bounds([parameter_ranges[key][0] for key in parameters_to_optimize])
# Upper bounds of parameters values
opt.set_upper_bounds([parameter_ranges[key][1] for key in parameters_to_optimize])
# The initial step size to compute numerical gradients
opt.set_initial_step([steps[key] for key in parameters_to_optimize])
# Maximum number of evaluations allowed
opt.set_maxeval(200)
# Relative tolerance for convergence
opt.set_ftol_rel(0.1)

# Start the optimization with the first guess
firstguess = [defaults[param] for param in parameters_to_optimize]
x = opt.optimize(firstguess)
print("\nOptimum at parameters: %s" % x)
print("Minimum value = ",  opt.last_optimum_value())
print("Result code = ", opt.last_optimize_result())
print("With %i function calls" % objfunc_calculator.n_calls)

# Rerun with the best parameters found
error = objfunc_calculator(x)
fig, axes = plt.subplots(figsize=(12,8))
axes.plot_date(df_observations.index, df_observations.LAI, label="Observed LAI")
axes.plot_date(objfunc_calculator.df_simulations.index, objfunc_calculator.df_simulations.LAI, "k:", label="optimized")
# Rerun to show the first guess for the first guess
error = objfunc_calculator(firstguess)
axes.plot_date(objfunc_calculator.df_simulations.index, objfunc_calculator.df_simulations.LAI, "g:", label="first guess")
axes.set_title("Observed LAI with optimized model.")
r = fig.legend()

plt.show()

# Optionally save the data frame to a CSV file
df.to_csv(os.path.join(data_dir, 'model_output.csv'), index=False)
