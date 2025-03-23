

# WOFOST Model and Remote Sensing Codebase (brief-unreviewed)

Welcome to the **DS_CROPS** repository! Here, you’ll find **two main sets of scripts**:

1. **WOFOST modeling and calibration** scripts (in `Scripts for WOFOST model/`)  
2. **Remote sensing (GEE) data acquisition and LAI/NDVI** scripts (in `Scripts for Remote Sensing data acquisition/`)

Additionally, outputs and intermediate `.agro` files or model results are organized under **`Output/`**. Below is a detailed explanation of the **folder structure** and how to get started.

---

## 1. Repository Structure

```
DS_CROPS/
│
├── Output/
│   ├── Optimization based on LAI/
│   │   └── model_output.csv
│   ├── Optimization script based on past yield/
│   │   ├── agrorelevantcrop_ID_1063_row_1.agro
│   │   ├── agrorelevantcrop_ID_1065_row_2.agro
│   │   ├── ...
│   └── Calibration of all crop parameters based on growth cycle1/
│       └── model_results.png
│
├── Scripts for Remote Sensing data acquisition/
│   ├── GEE COMBINED SCRIPT.txt
│   ├── NDVI animation script.txt
│   ├── READ ME.txt
│   ├── SG filter script.py
│   ├── Script for high resolution LAI data.py
│   ├── To create bounding box.txt
│   └── To locate lat long points script.txt
│
├── Scripts for WOFOST model/
│   ├── Calibration of all crop parameters based on growth cycle1.py
│   ├── Main script to run the model.py
│   ├── Optimization based on LAI.py
│   ├── Optimization script based on past yield.py
│   ├── Sensitivity analysis script based on past yield data.py
│   ├── Visuals script.py
│   ├── config.py
│   ├── sensitivity analysis based on growth cycle.py
│   └── __pycache__/
│       └── config.cpython-312.pyc
│
├── WOFOST MODEL/
│   ├── Test 2/
│   │   ├── Coarse.soil
│   │   ├── Observed partitioning data.xlsx
│   │   ├── agrofile_npk.agro
│   │   ├── literature findings.txt
│   │   ├── model_output.csv
│   │   ├── morris_sensitivity_analysis_results.csv
│   │   ├── practice_sugarbeet.crop
│   │   └── ss/
│   │       └── practice_sugarbeet.crop
│   └── ...
│
├── Model input files/
│   ├── Soil and Location/
│   │   ├── Coarse.soil
│   │   ├── Crop info ID.xlsx
│   │   ├── Fine sand.soil
│   │   ├── Fine sandy loam.soil
│   │   ├── Fine.soil
│   │   ├── ID SOIL and Location.xlsx
│   │   ├── Loamy fine sand.soil
│   │   ├── Medium fine.soil
│   │   ├── Medium.soil
│   │   ├── Very fine.soil
│   │   ├── Very loamy fine sand.soil
│   │   └── surface_texture.pdf
│   └── crop and agro/
│       ├── Crop info ID.xlsx
│       ├── Main script to run the model.py
│       ├── Optimization script based on past yield.py
│       ├── Output_data.xlsx
│       ├── Output_data_with_outliers.xlsx
│       ├── Sensitivity analysis script.py
│       ├── practice_relevantcrop.crop
│       ├── ~$Crop info BDF.xlsx
│       ├── ~$Output_data.xlsx
│       ├── visuals of results/
│       │   ├── Obs vs estimated.xlsx
│       │   └── Visuals script.py
│       ├── yaml agro files/
│       │   ├── agrorelevantcrop_ID_1063_row_1.agro
│       │   ├── agrorelevantcrop_ID_1065_row_2.agro
│       │   ├── ...
│       └── Crop parameters files/
│           └── Callibrated crops files from WOFOST/
│               ├── Read me.txt
│               ├── Barley crop/
│               ├── Faba bean/
│               ├── Potatoe/
│               ├── Grain maize/
│               ├── Rapeseed/
│               ├── Soybean/
│               ├── Sugarbeet/
│               ├── Sunflower/
│               ├── Sweet potato/
│               └── Winter wheat/
│
└── ...
```

### Key Folders

1. **`Scripts for Remote Sensing data acquisition/`**  
   - Contains scripts to get high-resolution LAI, NDVI animations, bounding boxes, etc.

2. **`Scripts for WOFOST model/`**  
   - **`calibration`** (yield-based, LAI-based, full growth cycle)
   - **`sensitivity analysis`** scripts
   - **`Main script to run the model.py`** for quick testing
   - **`config.py`** centralizing references to input paths, PCSE objects, etc.

3. **`Output/`**  
   - Subfolders for storing outputs from calibration or optimization runs (`model_results.png`, `.agro` files, CSV logs).

4. **`WOFOST MODEL/`**  
   - Contains sample soil files, example `.crop` files (`practice_sugarbeet.crop`), and `agrofile_npk.agro` for test runs. Also example “Test 2/” data.

5. **`Model input files/`**  
   - A more expanded set of soils, crop files, Excel references, and possibly older scripts or references.

---

## 2. Usage & Setup

1. **Install Dependencies**  
   - Python 3.9+  
   - [pcse](https://pypi.org/project/pcse/)  
   - [SALib](https://pypi.org/project/SALib/)  
   - standard scientific packages (`numpy`, `pandas`, `matplotlib`, etc.)
   - possibly `tqdm`, `PyYAML`, etc.

2. **Adjust `config.py`**  
   - Found under `Scripts for WOFOST model/`.
   - Points to default data directories (`DATA_DIR`, `SOIL_FILE`, etc.), sets up PCSE objects (cropdata, soildata, parameters).
   - Confirms references to `OBSERVED_PARTITIONING_FILE`, so you can seamlessly run scripts.

3. **Run a Script**  
   - **Yield-based**: `Optimization script based on past yield.py`  
   - **LAI-based**: `Optimization based on LAI.py` (requires remote-sensed LAI time series)  
   - **Growth-cycle**: `Calibration of all crop parameters based on growth cycle1.py`  
   - **Sensitivity**: `Sensitivity analysis script based on past yield data.py` or `sensitivity analysis based on growth cycle.py`

4. **Check Outputs**  
   - By default, scripts output `.agro` files, CSV logs, or figures under **`Output/`** subfolders.

---

## 3. Remote Sensing Integration

- **`Scripts for Remote Sensing data acquisition/`**  
  - **`NDVI animation script.txt`**, **`Script for high resolution LAI data.py`** (pull LAI from Sentinel-2).  
  - **`SG filter script.py`** for smoothing LAI data (Savitzky–Golay).
  - **`To create bounding box.txt`**, **`To locate lat long points script.txt`** for GEE-based region definitions.

- Once LAI or NDVI is downloaded, calibration scripts in the **WOFOST model** folder incorporate these data points for parameter optimization.

---

## 4. Important Scripts

- **`Main script to run the model.py`** – Quick demonstration or debugging of a single WOFOST scenario.  
- **`Optimization script based on past yield.py`** – Calibrate crop parameters using historical yields.  
- **`Optimization based on LAI.py`** – Minimizes LAI error between observed (Sentinel-2) and WOFOST-estimated LAI.  
- **`Calibration of all crop parameters based on growth cycle1.py`** – Comprehensive calibration (LAI + biomass partitioning, etc.).  
- **`Sensitivity analysis script based on past yield data.py`** – Uses SALib (Morris) for yield-based sensitivity.  
- **`sensitivity analysis based on growth cycle.py`** – Another SALib approach but focusing on multi-stage biomass or LAI.  
- **`Visuals script.py`** – For plotting or comparing observed vs. simulated data.  

---

## 5. Adding a New Crop

1. **Copy** a “nearest” `.crop` file from `Model input files/crop and agro/Crop parameters files/` if your new crop is physiologically similar.  
2. **Refine** or calibrate using:
   - Past yield data → `Optimization script based on past yield.py`.  
   - LAI time series → `Optimization based on LAI.py`.  
   - Multi-stage biomass → `Calibration of all crop parameters based on growth cycle1.py`.

---

## 6. Further Details

- **`READ ME.txt`** in `Scripts for Remote Sensing data acquisition/` for step-by-step instructions on GEE scripts to acquire NDVI, LAI, bounding boxes, etc.  
- WOFOST documentation:  
  - [PCSE official docs](https://pcse.readthedocs.io/en/stable/)  
  - [WOFOST notebooks](https://github.com/ajwdewit/pcse_notebooks)

---

