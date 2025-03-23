import os
import pandas as pd
from datetime import datetime
import yaml
import logging
import time
from tqdm import tqdm
import numpy as np
from scipy.optimize import minimize
from pcse.fileinput import CABOFileReader, YAMLAgroManagementReader
from pcse.util import WOFOST71SiteDataProvider
from pcse.base import ParameterProvider
from pcse.db import NASAPowerWeatherDataProvider
from pcse.models import Wofost71_WLP_FD
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Disable logging
logging.disable(logging.CRITICAL)

# Custom representer to format dates correctly without time
def represent_date(self, data):
    value = data.strftime('%Y-%m-%d')
    return self.represent_scalar('tag:yaml.org,2002:str', value)

yaml.add_representer(datetime.date, represent_date)

def create_agro_yaml_file(index, id, sowing_date, harvest_date, directory):
    campaign_start_date = datetime(sowing_date.year, 1, 1).date()  # Ensure it's a date only
    data = {
        'AgroManagement': [{
            campaign_start_date: {
                'CropCalendar': {
                    'crop_name': 'relevant-crop',
                    'variety_name': 'relevant_crop_code',
                    'crop_start_date': sowing_date,
                    'crop_start_type': 'sowing',
                    'crop_end_date': harvest_date,
                    'crop_end_type': 'harvest',
                    'max_duration': 300
                },
                'TimedEvents': None,
                'StateEvents': None
            }
        }]
    }

    # Include index in the filename to ensure uniqueness
    filename = os.path.join(directory, f'agrorelevantcrop_ID_{id}_row_{index + 1}.agro')
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)
    print(f"File saved: {filename}")

# Setup paths and load data
crop_info_path = r'C:\data copy\WOFOST MODEL\Model input files\crop and agro\Crop info ID.xlsx'
output_directory = r'C:\data copy\WOFOST MODEL\Model input files\crop and agro\yaml agro files'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

crop_info = pd.read_excel(crop_info_path)
crop_info['Sowing date'] = pd.to_datetime(crop_info['Sowing date'], format='%d.%m.%Y').dt.date
crop_info['Harvest date'] = pd.to_datetime(crop_info['Harvest date'], format='%d.%m.%Y').dt.date

# Generate YAML files for each row entry
for index, row in crop_info.iterrows():
    create_agro_yaml_file(index, row['ID'], row['Sowing date'], row['Harvest date'], output_directory)

# Prepare for the model and optimization
soil_location_dir = r'C:\data copy\WOFOST MODEL\Model input files\Soil and Location'
agro_files_dir = r'C:\data copy\WOFOST MODEL\Model input files\crop and agro\yaml agro files'
id_soil_location_path = os.path.join(soil_location_dir, 'ID SOIL and Location.xlsx')
crop_info_path = os.path.join(soil_location_dir, 'Crop info ID.xlsx')
crop_parameters_file = os.path.join(agro_files_dir, '..', 'practice_relevantcrop.crop')

id_soil_location = pd.read_excel(id_soil_location_path)
crop_info = pd.read_excel(crop_info_path)
crop_parameters = CABOFileReader(crop_parameters_file)

crop_info['RRMSE TWSO %'] = np.nan

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
                value = params[0] if not np.isnan(params[0]) else 300
                file.write(f"TSUMEM   = {value:.1f}    ! temperature sum from sowing to emergence [cel d]\n")
            elif line.startswith("TSUM1"):
                value = params[1] if not np.isnan(params[1]) else 1100
                file.write(f"TSUM1    = {value:.1f}    ! temperature sum from emergence to anthesis [cel d]\n")
            elif line.startswith("TSUM2"):
                value = params[2] if not np.isnan(params[2]) else 680
                file.write(f"TSUM2    = {value:.1f}    ! temperature sum from anthesis to maturity [cel d]\n")
            elif line.startswith("TDWI"):
                value = params[3] if not np.isnan(params[3]) else 140
                file.write(f"TDWI     = {value:.1f}    ! initial total crop dry weight [kg ha-1]\n")
            elif line.startswith("LAIEM"):
                value = params[4] if not np.isnan(params[4]) else 0.1665
                file.write(f"LAIEM    = {value:.4f}    ! leaf area index at emergence [ha ha-1]\n")
            elif line.startswith("SPAN"):
                value = params[5] if not np.isnan(params[5]) else 26.3
                file.write(f"SPAN     = {value:.1f}    ! life span of leaves growing at 35 Celsius [d]\n")
            elif line.startswith("RGRLAI"):
                value = params[6] if not np.isnan(params[6]) else 0.0075
                file.write(f"RGRLAI   = {value:.4f}    ! maximum relative increase in LAI [ha ha-1 d-1]\n")
            elif line.startswith("SLATB"):
                value = params[7] if not np.isnan(params[7]) else 0.0020
                file.write(f"SLATB    = 0.00, {value:.4f},   ! specific leaf area\n")
            elif line.startswith("CFET"):
                value = params[8] if not np.isnan(params[8]) else 1.00
                file.write(f"CFET     = {value:.2f}   ! correction factor transpiration rate [-]\n")
            elif line.startswith("DEPNR"):
                value = params[9] if not np.isnan(params[9]) else 4.5
                file.write(f"DEPNR    = {value:.1f}    ! crop group number for soil water depletion [-]\n")
            elif line.startswith("RDMCR"):
                value = params[10] if not np.isnan(params[10]) else 125
                file.write(f"RDMCR    = {value:.1f}     ! maximum rooting depth [cm]\n")
            else:
                file.write(line)

# Wrapper function to include all necessary arguments
def objective_function_wrapper(params, id_soil_location, crop_info, crop_parameters_file, agro_files_dir):
    return run_model_and_get_rrmse(params, id_soil_location, crop_info, crop_parameters_file, agro_files_dir)

best_params = None
best_error = float('inf')

def run_model_and_get_rrmse(params, id_soil_location, crop_info, crop_parameters_file, agro_files_dir):
    global best_params, best_error
    # Update crop parameters
    update_crop_file(params, crop_parameters_file)
    
    errors_twso = []

    def simulate_crop(index, crop_row):
        id_number = crop_row['ID']
        row_number = index + 1
        agro_file_path = os.path.join(agro_files_dir, f'agrorelevantcrop_ID_{id_number}_row_{row_number}.agro')

        if not os.path.exists(agro_file_path):
            return None
        
        soil_info = id_soil_location[id_soil_location['ID'] == id_number].iloc[0]
        soil_file = os.path.join(soil_location_dir, f"{soil_info['Soil File']}")
        if not os.path.exists(soil_file):
            return None

        soildata = CABOFileReader(soil_file)
        sitedata = WOFOST71SiteDataProvider(WAV=100)
        parameters = ParameterProvider(soildata=soildata, cropdata=CABOFileReader(crop_parameters_file), sitedata=sitedata)
        wdp = NASAPowerWeatherDataProvider(latitude=soil_info['Latitude'], longitude=soil_info['Longitude'])
        agromanagement = YAMLAgroManagementReader(agro_file_path)
        wofsim = Wofost71_WLP_FD(parameters, wdp, agromanagement)

        try:
            wofsim.run_till_terminate()
            df = pd.DataFrame(wofsim.get_output())
        except Exception:
            return None

        estimated_yield = df['TWSO'].iloc[-1]

        observed_yield = crop_row['Dry matter yield (TWSO) kg/ha (observed)']
        rrms_twso = calculate_rrmse([observed_yield], [estimated_yield])

        return index, estimated_yield, rrms_twso

    results = Parallel(n_jobs=-1)(delayed(simulate_crop)(index, crop_row) for index, crop_row in tqdm(crop_info.iterrows(), total=crop_info.shape[0], desc="Running simulations"))

    for result in results:
        if result is None:
            continue
        index, estimated_yield, rrms_twso = result
        crop_info.at[index, 'Dry matter yield (TWSO) kg/ha (estimated)'] = estimated_yield
        crop_info.at[index, 'RRMSE TWSO %'] = rrms_twso
        errors_twso.append(np.abs(crop_info.at[index, 'Dry matter yield (TWSO) kg/ha (observed)'] - estimated_yield))

    mean_rrmse_twso = crop_info['RRMSE TWSO %'].mean()
    total_error = mean_rrmse_twso

    print(f"Mean RRMSE TWSO: {mean_rrmse_twso:.2f}")
    print(f"Parameters used: TSUMEM={params[0]:.1f}, TSUM1={params[1]:.1f}, TSUM2={params[2]:.1f}, TDWI={params[3]:.1f}, LAIEM={params[4]:.4f}, SPAN={params[5]:.1f}, RGRLAI={params[6]:.4f}, SLATB={params[7]:.4f}, CFET={params[8]:.2f}, DEPNR={params[9]:.1f}, RDMCR={params[10]:.1f}")

    if total_error < best_error:
        best_error = total_error
        best_params = params.copy()
        update_crop_file(best_params, crop_parameters_file)
        print(f"New best parameters found: {best_params} with total error: {best_error:.2f}")

    return total_error

calibrate_parameters = {
    'TSUMEM': True,
    'TSUM1': True,
    'TSUM2': True,
    'TDWI': True,
    'LAIEM': True,
    'SPAN': True,
    'RGRLAI': True,
    'SLATB': False,
    'CFET': False,
    'DEPNR': False,
    'RDMCR': True
}

default_params = {}
with open(crop_parameters_file, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if line.startswith("TSUMEM"):
            default_params['TSUMEM'] = float(line.split('=')[1].strip().split()[0])
        elif line.startswith("TSUM1"):
            default_params['TSUM1'] = float(line.split('=')[1].strip().split()[0])
        elif line.startswith("TSUM2"):
            default_params['TSUM2'] = float(line.split('=')[1].strip().split()[0])
        elif line.startswith("TDWI"):
            default_params['TDWI'] = float(line.split('=')[1].strip().split()[0])
        elif line.startswith("LAIEM"):
            default_params['LAIEM'] = float(line.split('=')[1].strip().split()[0])
        elif line.startswith("SPAN"):
            default_params['SPAN'] = float(line.split('=')[1].strip().split()[0])
        elif line.startswith("RGRLAI"):
            default_params['RGRLAI'] = float(line.split('=')[1].strip().split()[0])
        elif line.startswith("SLATB"):
            default_params['SLATB'] = float(line.split('=')[1].strip().split(',')[1].strip())
        elif line.startswith("CFET"):
            default_params['CFET'] = float(line.split('=')[1].strip().split()[0])
        elif line.startswith("DEPNR"):
            default_params['DEPNR'] = float(line.split('=')[1].strip().split()[0])
        elif line.startswith("RDMCR"):
            default_params['RDMCR'] = float(line.split('=')[1].strip().split()[0])

bounds = [
    (50, 150) if calibrate_parameters['TSUMEM'] else (default_params['TSUMEM'], default_params['TSUMEM']),
    (400, 1200) if calibrate_parameters['TSUM1'] else (default_params['TSUM1'], default_params['TSUM1']),
    (800, 1800) if calibrate_parameters['TSUM2'] else (default_params['TSUM2'], default_params['TSUM2']),
    (20, 40) if calibrate_parameters['TDWI'] else (default_params['TDWI'], default_params['TDWI']),
    (0.0120, 0.0380) if calibrate_parameters['LAIEM'] else (default_params['LAIEM'], default_params['LAIEM']),
    (15, 35) if calibrate_parameters['SPAN'] else (default_params['SPAN'], default_params['SPAN']),
    (0.00800, 0.0120) if calibrate_parameters['RGRLAI'] else (default_params['RGRLAI'], default_params['RGRLAI']),
    (0.0030, 0.0032) if calibrate_parameters['SLATB'] else (default_params['SLATB'], default_params['SLATB']),
    (0.9, 1.1) if calibrate_parameters['CFET'] else (default_params['CFET'], default_params['CFET']),
    (4.50, 4.51) if calibrate_parameters['DEPNR'] else (default_params['DEPNR'], default_params['DEPNR']),
    (80, 110) if calibrate_parameters['RDMCR'] else (default_params['RDMCR'], default_params['RDMCR'])
]

# DE with momentum initialization
def differential_evolution_with_momentum(objective_function, bounds, strategy='best1bin', maxiter=100, popsize=10, tol=0.01, momentum=0.9):
    population = np.random.rand(popsize, len(bounds))
    for i in range(len(bounds)):
        population[:, i] = bounds[i][0] + population[:, i] * (bounds[i][1] - bounds[i][0])
    velocities = np.zeros_like(population)
    best_position = None
    best_score = float('inf')
    scores = np.full(popsize, float('inf'))
    
    for iteration in range(maxiter):
        for i in range(popsize):
            trial = np.copy(population[i])
            if best_position is not None:
                velocities[i] = momentum * velocities[i] + np.random.rand() * (best_position - population[i])
            trial += velocities[i]
            for j in range(len(bounds)):
                if trial[j] < bounds[j][0] or trial[j] > bounds[j][1]:
                    trial[j] = population[i, j]
            score = objective_function(trial)
            if np.isfinite(score) and score < scores[i]:
                population[i] = trial
                scores[i] = score
                if score < best_score:
                    best_score = score
                    best_position = np.copy(trial)
                    best_params = best_position.copy()  # Update the global best parameters
                    best_error = best_score
                    update_crop_file(best_position, crop_parameters_file)  # Update crop file with new best parameters
                    print(f"New best parameters found: {best_params} with total error: {best_error:.2f}")
        print(f"Iteration {iteration+1}/{maxiter}, Best Score: {best_score}")
    
    return best_position, best_score

total_evaluations = 5

result = differential_evolution_with_momentum(
    lambda params: objective_function_wrapper(params, id_soil_location, crop_info, crop_parameters_file, agro_files_dir),
    bounds,
    maxiter=total_evaluations,
    popsize= 500,
    momentum=0.9
)

optimal_params, optimal_error = result

if optimal_params is None:
    raise ValueError("Differential Evolution did not converge to a solution.")

print(f"Optimal parameters found: {optimal_params}")

# Local refinement
local_result = minimize(
    lambda params: objective_function_wrapper(params, id_soil_location, crop_info, crop_parameters_file, agro_files_dir),
    optimal_params,
    method='L-BFGS-B',
    bounds=bounds
)
optimal_params_refined = local_result.x
print(f"Refined optimal parameters: {optimal_params_refined}")

# Save the optimal parameters to the crop file
update_crop_file(optimal_params_refined, crop_parameters_file)

# Print the best parameters and corresponding lowest error
print(f"Best parameters found during optimization: {best_params}")
print(f"Lowest total error: {best_error:.2f}")
