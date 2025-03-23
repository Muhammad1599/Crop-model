import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# Step 1: Read the Excel file
file_path = r'C:\Users\user\Downloads\obslai.xlsx'
df = pd.read_excel(file_path)

# Ensure the Date column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Linear Interpolation for missing or noisy values
df['LAI_interpolated'] = df['LAI'].interpolate()

# Step 2: Initial Long-term Trend Estimation using a low-order S-G filter
window_length = 19  # Larger window length for smooth trend
polyorder = 10  # Lower-order polynomial for long-term trend
df['LAI_trend'] = savgol_filter(df['LAI_interpolated'], window_length, polyorder)

# Step 3: Assign weights based on proximity to the trend
df['weight'] = np.where(df['LAI_interpolated'] >= df['LAI_trend'], 1, 0.1)  # Higher weights to points above the trend

# Step 4: Iteratively replace noisy points and fit again with a higher-order S-G filter
iterations = 200
for i in range(iterations):
    df['LAI_adjusted'] = np.where(df['LAI_interpolated'] >= df['LAI_trend'], 
                                  df['LAI_interpolated'], df['LAI_trend'])
    
    # Step 5: Apply a higher-order Savitzky-Golay filter for final smoothing
    df['LAI_smoothed'] = savgol_filter(df['LAI_adjusted'], window_length, polyorder + 1)  # Use higher order

    # Update the trend for the next iteration
    df['LAI_trend'] = df['LAI_smoothed']

# Step 6: Print the smoothed LAI values with corresponding dates in MM/DD/YYYY format
print("Date\t\tSmoothed LAI")
for date, lai in zip(df['Date'], df['LAI_smoothed']):
    formatted_date = date.strftime('%m/%d/%Y')  # Format date as MM/DD/YYYY
    print(f"{formatted_date}\t{lai:.4f}")

# Step 7: Plot the results
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['LAI'], label='Original LAI', color='blue')
plt.plot(df['Date'], df['LAI_smoothed'], label='Smoothed LAI', color='red')
plt.xlabel('Date')
plt.ylabel('LAI')
plt.title('LAI Smoothing using Iterative Savitzky-Golay Filter')
plt.legend()
plt.show()
