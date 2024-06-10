import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# Read data from the CSV file
image_names = []
actual_ages = []
estimated_ages = []
deltas = []

# Define paths to data and labels
current_script_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_script_path)
data_dir = os.path.join(root_path, "data")
validation_file = os.path.join(data_dir, "validation.csv")

with open(validation_file, 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        image_names.append(row['Image'])
        actual_ages.append(float(row['Actual Age']))
        estimated_ages.append(float(row['Estimated Age']))
        deltas.append(float(row['Delta']))

# Create a scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(actual_ages, estimated_ages, c=deltas, cmap='viridis', alpha=0.7)
plt.colorbar(label='Delta')

# Add labels and title
plt.xlabel('Actual Age')
plt.ylabel('Estimated Age')
plt.title('Actual Age vs. Estimated Age')

# Calculate and plot the line of best fit
coefficients = np.polyfit(actual_ages, estimated_ages, 1)
line = np.poly1d(coefficients)
plt.plot(actual_ages, line(actual_ages), color='red', linestyle='--', label='Line of Best Fit')

# Add legend
plt.legend()

# Display the plot
plt.tight_layout()

# plt.show()
plot_file = os.path.join(data_dir, "best-fit-scatter-plot.png")

# Save the plot
plt.savefig(plot_file)
print(f"Saved plot to {plot_file}")



# reset plt
plt.close('all')

# Create a histogram
plt.figure(figsize=(10, 8))
plt.hist(deltas, bins=20, edgecolor='black', alpha=0.7)

# Add labels and title
plt.xlabel('Delta (Months)')
plt.ylabel('Frequency')
plt.title('Histogram of Deltas')

# Add vertical line at delta = 0
plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Delta = 0')

# Add legend
plt.legend()

# Display the plot
plt.tight_layout()
plot_file = os.path.join(data_dir, "delta-histogram.png")

# Save the plot
plt.savefig(plot_file)
print(f"Saved plot to {plot_file}")


plt.close('all')
plt.figure(figsize=(12, 8))
plt.errorbar(image_names, actual_ages, yerr=deltas, fmt='o', capsize=4, ecolor='red', elinewidth=1.5, alpha=0.7)

# Add labels and title
plt.xlabel('Image')
plt.ylabel('Age (Months)')
plt.title('Error Bar Plot - Actual Age vs. Estimated Age')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust the spacing between subplots
plt.subplots_adjust(bottom=0.2)

# Display the plot
plt.tight_layout()
plot_file = os.path.join(data_dir, "error-bar-plot.png")

# Save the plot
plt.savefig(plot_file)
print(f"Saved plot to {plot_file}")