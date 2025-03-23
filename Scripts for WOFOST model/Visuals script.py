import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
file_path = r"C:\data copy\WOFOST MODEL\Model input files\crop and agro\visuals of results\Obs vs estimated.xlsx"
df = pd.read_excel(file_path)

# Extract the necessary columns
observed = df['Observed']
estimated = df['Estimated']
mean_rrmse = df['Mean RRMSE %'].iloc[0]  # Assuming Mean RRMSE is the same across the column

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(observed, estimated, color='blue', s=2)  # s=20 for smaller points

# Plotting a line for perfect estimation (y = x)
min_val = min(min(observed), min(estimated))
max_val = max(max(observed), max(estimated))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x (Perfect Estimation)')

# Calculate and plot the regression line
slope, intercept = np.polyfit(observed, estimated, 1)
regression_line = slope * observed + intercept
plt.plot(observed, regression_line, color='green', linestyle='-', label=f'Regression Line: y = {slope:.2f}x + {intercept:.2f}')

# Calculate R-squared
correlation_matrix = np.corrcoef(observed, estimated)
correlation_xy = correlation_matrix[0, 1]
r_squared = correlation_xy**2

# Adjust the x and y axis limits to improve the visual scaling
padding = (max_val - min_val) * 0.1  # Add 10% padding on both sides
plt.xlim(min_val - padding, max_val + padding)
plt.ylim(min_val - padding, max_val + padding)

# Adding necessary statistics
plt.title('Faba bean regional validation result (estimated vs observed yield kg/ha)')
plt.xlabel('Observed Yield (kg/ha)')
plt.ylabel('Estimated Yield (kg/ha)')

# Add mean RRMSE % text to the plot
plt.text(0.05, 0.95, f'Mean RRMSE %: {mean_rrmse:.2f}', transform=plt.gca().transAxes, fontsize=12, color='black',
         bbox=dict(facecolor='yellow', alpha=0.5), verticalalignment='top')

# Add total number of observations
plt.text(0.05, 0.90, f'Total Observations: {len(observed)}', transform=plt.gca().transAxes, fontsize=12, color='black',
         bbox=dict(facecolor='lightblue', alpha=0.5), verticalalignment='top')

# Add R-squared value
plt.text(0.05, 0.85, f'R-squared: {r_squared:.2f}', transform=plt.gca().transAxes, fontsize=12, color='black',
         bbox=dict(facecolor='lightgreen', alpha=0.5), verticalalignment='top')

# Show legend with adjusted position to avoid overlap
plt.legend(loc='lower right', fontsize=10)

# Show the plot
plt.show()
