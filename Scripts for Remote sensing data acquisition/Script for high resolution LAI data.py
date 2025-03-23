# Step 0: Install necessary packages
#!pip install hda
#!pip install rasterio

# Step 1: Import necessary libraries
import time
import shutil
from hda import Client, Configuration  # WEkEO Hub Data Access API interaction
from pathlib import Path
import rasterio
from rasterio.windows import from_bounds
import numpy as np
from pyproj import Transformer
import re
import matplotlib.pyplot as plt
import pandas as pd

# Function to create bounding box around center lat/lon
def create_bounding_box(lat, lon, lat_offset=0.000244160692, lon_offset=0.001879995563507):
    return {
        'lon_min': lon - lon_offset,
        'lat_min': lat - lat_offset,
        'lon_max': lon + lon_offset,
        'lat_max': lat + lat_offset
    }

# Set your central latitude and longitude
center_lon = 12.0161479046633
center_lat = 50.1520599010391

# Create the bounding box using the center point
bounding_box = create_bounding_box(center_lat, center_lon)

# Step 1A: Clean up the downloaded data folders
downloaded_data_path = Path('/content/downloaded_data')
downloaded_quality_data_path = Path('/content/downloaded_quality_data')
downloaded_ppi_data_path = Path('/content/downloaded_ppi_data')

# Delete all files in the downloaded_data folder
if downloaded_data_path.exists() and downloaded_data_path.is_dir():
    shutil.rmtree(downloaded_data_path)
    print(f"Deleted all files in {downloaded_data_path}")

# Delete all files in the downloaded_quality_data folder
if downloaded_quality_data_path.exists() and downloaded_quality_data_path.is_dir():
    shutil.rmtree(downloaded_quality_data_path)
    print(f"Deleted all files in {downloaded_quality_data_path}")

# Delete all files in the downloaded_ppi_data folder
if downloaded_ppi_data_path.exists() and downloaded_ppi_data_path.is_dir():
    shutil.rmtree(downloaded_ppi_data_path)
    print(f"Deleted all files in {downloaded_ppi_data_path}")

# Recreate the folders to use them again
downloaded_data_path.mkdir(parents=True, exist_ok=True)
downloaded_quality_data_path.mkdir(parents=True, exist_ok=True)
downloaded_ppi_data_path.mkdir(parents=True, exist_ok=True)

# Step 2: Configure user's credentials without a .hdarc
conf = Configuration(user="muhammad1515", password="Honey15@%")
hda_client = Client(config=conf)

# Step 3: Create the request for QFLAG2 and download data
query_qflag = {
    "dataset_id": "EO:EEA:DAT:CLMS_HRVPP_VI",
    "productType": "QFLAG2",
    "platformSerialIdentifier": "S2A",
    "resolution": "10",
    "start": "2023-01-01T00:00:00.000Z",
    "end": "2023-10-07T00:00:00.000Z",
    "bbox": [
        bounding_box['lon_min'],  # Longitude min
        bounding_box['lat_min'],  # Latitude min
        bounding_box['lon_max'],  # Longitude max
        bounding_box['lat_max']   # Latitude max
    ],
    "itemsPerPage": 100,
    "startIndex": 0
}

# Search for QFLAG2 datasets
qflag_matches = hda_client.search(query_qflag)
print(f"Found {len(qflag_matches)} matching QFLAG2 items with a total volume of {qflag_matches.volume/1e9:.2f} GB")

# Step 4: Download QFLAG2 data and identify files with quality flag 1
if len(qflag_matches) > 0:
    QFLAG_OUTPUT_PATH = downloaded_quality_data_path

    for idx, match in enumerate(qflag_matches):
        print(f"Downloading QFLAG2 item {idx + 1}/{len(qflag_matches)}...")
        match.download(str(QFLAG_OUTPUT_PATH))
        print(f"Downloaded to {QFLAG_OUTPUT_PATH}")
        time.sleep(5)  # Delay between downloads

    # Transform the central coordinates to UTM
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    center_utm_x, center_utm_y = transformer.transform(center_lon, center_lat)
    half_size = 90
    west_utm = center_utm_x - half_size
    east_utm = center_utm_x + half_size
    south_utm = center_utm_y - half_size
    north_utm = center_utm_y + half_size

    selected_dates = []

    for file_path in QFLAG_OUTPUT_PATH.glob("*_QFLAG2.tif"):
        print(f"Processing file: {file_path.name}")

        with rasterio.open(file_path) as dataset:
            window = from_bounds(west_utm, south_utm, east_utm, north_utm, transform=dataset.transform)
            print(f"Window size: (rows, cols) = ({window.height}, {window.width})")
            print(f"Dataset CRS: {dataset.crs}, Dataset bounds: {dataset.bounds}")

            if window.height > 0 and window.width > 0:
                data = dataset.read(1, window=window)

                if data.size > 0 and not np.all(data == dataset.nodata):
                    if 1 in data:
                        date_pattern = re.compile(r'(\d{8}T\d{6})')
                        match = date_pattern.search(file_path.name)
                        if match:
                            date_str = match.group(1)
                            selected_dates.append(date_str)
                            print(f"Highest quality flag (1) found in file: {file_path.name}, Date: {date_str}")
                else:
                    print("The data array is empty or contains only NoData values.")
            else:
                print("The bounding box does not intersect with the raster data or is too small.")
else:
    print("No QFLAG2 matching items found to download.")
    selected_dates = []

# Step 5: Query and download LAI data for the selected dates with the highest quality
if selected_dates:
    for date_str in selected_dates:
        query_lai = {
            "dataset_id": "EO:EEA:DAT:CLMS_HRVPP_VI",
            "productType": "LAI",
            "platformSerialIdentifier": "S2A",
            "resolution": "10",
            "start": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T00:00:00.000Z",
            "end": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T23:59:59.999Z",
            "bbox": [
                bounding_box['lon_min'],  # Longitude min
                bounding_box['lat_min'],  # Latitude min
                bounding_box['lon_max'],  # Longitude max
                bounding_box['lat_max']   # Latitude max
            ],
            "itemsPerPage": 100,
            "startIndex": 0
        }

        lai_matches = hda_client.search(query_lai)
        print(f"Found {len(lai_matches)} matching LAI items for {date_str} with a total volume of {lai_matches.volume/1e9:.2f} GB")

        if len(lai_matches) > 0:
            LAI_OUTPUT_PATH = downloaded_data_path

            for idx, match in enumerate(lai_matches):
                print(f"Downloading LAI item {idx + 1}/{len(lai_matches)} for date {date_str}...")
                match.download(str(LAI_OUTPUT_PATH))
                print(f"Downloaded to {LAI_OUTPUT_PATH}")
                time.sleep(5)
        else:
            print(f"No matching LAI items found for date {date_str}.")

# Step 5A: Query and download PPI data for the selected dates with the highest quality
if selected_dates:
    for date_str in selected_dates:
        query_ppi = {
            "dataset_id": "EO:EEA:DAT:CLMS_HRVPP_VI",
            "productType": "PPI",
            "platformSerialIdentifier": "S2A",
            "resolution": "10",
            "start": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T00:00:00.000Z",
            "end": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T23:59:59.999Z",
            "bbox": [
                bounding_box['lon_min'],  # Longitude min
                bounding_box['lat_min'],  # Latitude min
                bounding_box['lon_max'],  # Longitude max
                bounding_box['lat_max']   # Latitude max
            ],
            "itemsPerPage": 100,
            "startIndex": 0
        }

        ppi_matches = hda_client.search(query_ppi)
        print(f"Found {len(ppi_matches)} matching PPI items for {date_str} with a total volume of {ppi_matches.volume/1e9:.2f} GB")

        if len(ppi_matches) > 0:
            PPI_OUTPUT_PATH = downloaded_ppi_data_path

            for idx, match in enumerate(ppi_matches):
                print(f"Downloading PPI item {idx + 1}/{len(ppi_matches)} for date {date_str}...")
                match.download(str(PPI_OUTPUT_PATH))
                print(f"Downloaded to {PPI_OUTPUT_PATH}")
                time.sleep(5)
        else:
            print(f"No matching PPI items found for date {date_str}.")

# Step 6: Extract LAI values from all relevant raster files
scaling_factor_lai = 0.0008
offset_lai = 0.0
dates_lai = []
max_lai_values = []
mean_lai_values = []

if len(selected_dates) > 0:
    for file_path in LAI_OUTPUT_PATH.glob("*_LAI.tif"):
        print(f"Processing file: {file_path.name}")

        with rasterio.open(file_path) as dataset:
            window = from_bounds(west_utm, south_utm, east_utm, north_utm, transform=dataset.transform)
            print(f"Window size: (rows, cols) = ({window.height}, {window.width})")
            print(f"Dataset CRS: {dataset.crs}, Dataset bounds: {dataset.bounds}")

            if window.height > 0 and window.width > 0:
                data = dataset.read(1, window=window)

                if data.size > 0 and not np.all(data == dataset.nodata):
                    lai_values = data * scaling_factor_lai + offset_lai
                    lai_values = np.ma.masked_equal(lai_values, dataset.nodata)
                    lai_min = np.min(lai_values)
                    lai_max = np.max(lai_values)
                    lai_mean = np.mean(lai_values)
                    lai_stddev = np.std(lai_values)

                    date_pattern = re.compile(r'(\d{8}T\d{6})')
                    match = date_pattern.search(file_path.name)
                    if match:
                        date_str = match.group(1)
                        date_obj = pd.to_datetime(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
                        dates_lai.append(date_obj)
                        max_lai_values.append(lai_max)
                        mean_lai_values.append(lai_mean)

                        print(f"Extracted LAI Statistics for {file_path.name}:")
                        print(f"Min LAI Value: {lai_min}")
                        print(f"Max LAI Value: {lai_max}")
                        print(f"Mean LAI Value: {lai_mean}")
                        print(f"Standard Deviation of LAI: {lai_stddev}")
                else:
                    print("The data array is empty or contains only NoData values.")
            else:
                print("The bounding box does not intersect with the raster data or is too small.")

# Step 6A: Extract PPI values from all relevant raster files
scaling_factor_ppi = 0.0001
offset_ppi = 0.0
dates_ppi = []
max_ppi_values = []
mean_ppi_values = []

if len(selected_dates) > 0:
    for file_path in PPI_OUTPUT_PATH.glob("*_PPI.tif"):
        print(f"Processing file: {file_path.name}")

        with rasterio.open(file_path) as dataset:
            window = from_bounds(west_utm, south_utm, east_utm, north_utm, transform=dataset.transform)
            print(f"Window size: (rows, cols) = ({window.height}, {window.width})")
            print(f"Dataset CRS: {dataset.crs}, Dataset bounds: {dataset.bounds}")

            if window.height > 0 and window.width > 0:
                data = dataset.read(1, window=window)

                if data.size > 0 and not np.all(data == dataset.nodata):
                    ppi_values = data * scaling_factor_ppi + offset_ppi
                    ppi_values = np.ma.masked_equal(ppi_values, dataset.nodata)
                    ppi_min = np.min(ppi_values)
                    ppi_max = np.max(ppi_values)
                    ppi_mean = np.mean(ppi_values)
                    ppi_stddev = np.std(ppi_values)

                    date_pattern = re.compile(r'(\d{8}T\d{6})')
                    match = date_pattern.search(file_path.name)
                    if match:
                        date_str = match.group(1)
                        date_obj = pd.to_datetime(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
                        dates_ppi.append(date_obj)
                        max_ppi_values.append(ppi_max)
                        mean_ppi_values.append(ppi_mean)

                        print(f"Extracted PPI Statistics for {file_path.name}:")
                        print(f"Min PPI Value: {ppi_min}")
                        print(f"Max PPI Value: {ppi_max}")
                        print(f"Mean PPI Value: {ppi_mean}")
                        print(f"Standard Deviation of PPI: {ppi_stddev}")
                else:
                    print("The data array is empty or contains only NoData values.")
            else:
                print("The bounding box does not intersect with the raster data or is too small.")

# Step 7: Prepare the DataFrame for plotting and saving to CSV for LAI
df_lai = pd.DataFrame({
    'Date': dates_lai,
    'Max LAI': max_lai_values,
    'Mean LAI': mean_lai_values
})

df_lai = df_lai.sort_values('Date')

# Save the Max LAI values to a CSV file
df_max_lai = df_lai[['Date', 'Max LAI']]
df_max_lai.to_csv('max_lai_values.csv', index=False)
print("Max LAI values saved to max_lai_values.csv")

# Save the Mean LAI values to a separate CSV file
df_mean_lai = df_lai[['Date', 'Mean LAI']]
df_mean_lai.to_csv('mean_lai_values.csv', index=False)
print("Mean LAI values saved to mean_lai_values.csv")

# Step 7A: Prepare the DataFrame for plotting and saving to CSV for PPI
df_ppi = pd.DataFrame({
    'Date': dates_ppi,
    'Max PPI': max_ppi_values,
    'Mean PPI': mean_ppi_values
})

df_ppi = df_ppi.sort_values('Date')

# Save the Max PPI values to a CSV file
df_max_ppi = df_ppi[['Date', 'Max PPI']]
df_max_ppi.to_csv('max_ppi_values.csv', index=False)
print("Max PPI values saved to max_ppi_values.csv")

# Save the Mean PPI values to a separate CSV file
df_mean_ppi = df_ppi[['Date', 'Mean PPI']]
df_mean_ppi.to_csv('mean_ppi_values.csv', index=False)
print("Mean PPI values saved to mean_ppi_values.csv")

# Step 8: Plotting the Max and Mean LAI values over time
plt.figure(figsize=(12, 6))
plt.plot(df_lai['Date'], df_lai['Max LAI'], marker='o', linestyle='-', color='blue')
plt.title('Max LAI Values Over Time (Sorted by Date)')
plt.xlabel('Date')
plt.ylabel('Max LAI Value')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df_lai['Date'], df_lai['Mean LAI'], marker='o', linestyle='-', color='green')
plt.title('Mean LAI Values Over Time (Sorted by Date)')
plt.xlabel('Date')
plt.ylabel('Mean LAI Value')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 8A: Plotting the Max and Mean PPI values over time
plt.figure(figsize=(12, 6))
plt.plot(df_ppi['Date'], df_ppi['Max PPI'], marker='o', linestyle='-', color='red')
plt.title('Max PPI Values Over Time (Sorted by Date)')
plt.xlabel('Date')
plt.ylabel('Max PPI Value')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df_ppi['Date'], df_ppi['Mean PPI'], marker='o', linestyle='-', color='purple')
plt.title('Mean PPI Values Over Time (Sorted by Date)')
plt.xlabel('Date')
plt.ylabel('Mean PPI Value')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
