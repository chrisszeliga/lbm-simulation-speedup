import os
import pandas as pd
import matplotlib.pyplot as plt

#############################################################

# computation.png Plot

# Load the CSV file
df = pd.read_csv("fluid.csv")

# Extract relevant columns
x = df["Timestep"]
y = df["SimStepTime"]

# Create scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color="red", label="SimStepTime")

# Labels and title
plt.xlabel("Timestep")
plt.ylabel("SimStepTime (s)")
plt.title("Simulation Step Time per Timestep")
plt.legend()

# Save the plot as a file
plt.savefig("computation.png")

#############################################################

# speedup.png Plot

# Directory containing the CSV files
runtimes_dir = 'runtimes'

# Create an empty list to store DataFrames
dataframes = []

# Read all CSV files in the directory
for filename in os.listdir(runtimes_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(runtimes_dir, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)

# Concatenate all DataFrames if we have any
if dataframes:
    data = pd.concat(dataframes, ignore_index=True)
else:
    # Create an empty DataFrame with correct column types if no files found
    data = pd.DataFrame(columns=['Threads', 'TotalSimTime'], 
                       dtype={'Threads': int, 'TotalSimTime': float})

# Make sure Threads column is integer type
data['Threads'] = data['Threads'].astype(int)
# Make sure TotalSimTime column is float type
data['TotalSimTime'] = data['TotalSimTime'].astype(float)

# Sort by number of threads
data = data.sort_values('Threads')

# Calculate speedup (time with 1 thread / time with n threads)
baseline_time = data[data['Threads'] == 1]['TotalSimTime'].values[0]
data['Speedup'] = baseline_time / data['TotalSimTime']

# Create the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6))

# Set a single title for the entire figure
fig.suptitle('Island in the Stream - Speedup', fontsize=16, y=0.98)

# First plot: Thread vs. Time
ax1.plot(data['Threads'], data['TotalSimTime'], 'r-', marker='o')
ax1.set_title('Threads vs. Time', fontsize=14)
ax1.set_xlabel('Threads', fontsize=12)
ax1.set_ylabel('Time (s)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)

# Use log scale but with custom tick labels
ax1.set_xscale('log', base=2)
ax1.set_xticks(data['Threads'].tolist())
ax1.set_xticklabels([str(int(x)) for x in data['Threads']])

# Add annotations with time values
for _, row in data.iterrows():
    ax1.annotate(f"{int(row['TotalSimTime'])}", 
                xy=(row['Threads'], row['TotalSimTime']),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightgray', ec='gray', alpha=0.8))

# Second plot: Thread vs. Speedup
ax2.plot(data['Threads'], data['Speedup'], 'r-', marker='o')
ax2.set_title('Threads vs. Speedup', fontsize=14)
ax2.set_xlabel('Threads', fontsize=12)
ax2.set_ylabel('Speedup', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

# Use log scale but with custom tick labels
ax2.set_xscale('log', base=2)
ax2.set_xticks(data['Threads'].tolist())
ax2.set_xticklabels([str(int(x)) for x in data['Threads']])

# Format y-axis with 'x' suffix
from matplotlib.ticker import FuncFormatter
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}x"))

# Add annotations with speedup values
for _, row in data.iterrows():
    speedup_text = f"{row['Speedup']:.1f}"
    ax2.annotate(speedup_text, 
                xy=(row['Threads'], row['Speedup']),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightgray', ec='gray', alpha=0.8))

# Adjust layout with more space at the top for the main title
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the figure
plt.savefig('speedup.png')

# Show the figure
plt.show()