import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

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

# Load the single CSV file
data = pd.read_csv("runtimes.csv")

# Rename 'Nodes' to 'Threads' for reuse with existing plot logic (optional)
# or just update the rest of the code to use 'Nodes'
data['Nodes'] = data['Nodes'].astype(int)
data['TotalSimTime'] = data['TotalSimTime'].astype(float)

# Sort by number of nodes
data = data.sort_values('Nodes')

# Calculate speedup
baseline_time = data[data['Nodes'] == 1]['TotalSimTime'].values[0]
data['Speedup'] = baseline_time / data['TotalSimTime']

# Create the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6))

# Set a single title for the entire figure
fig.suptitle('Islands in the Stream on Multiple Nodes - Speedup', fontsize=16, y=0.98)

# First plot: Nodes vs. Time
ax1.plot(data['Nodes'], data['TotalSimTime'], 'r-', marker='o')
ax1.set_title('Nodes vs. Time', fontsize=14)
ax1.set_xlabel('Nodes', fontsize=12)
ax1.set_ylabel('Time (s)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)

# Use log scale with custom tick labels
ax1.set_xscale('log', base=2)
ax1.set_xticks(data['Nodes'].tolist())
ax1.set_xticklabels([str(n) for n in data['Nodes']])

# Annotations
for _, row in data.iterrows():
    ax1.annotate(f"{int(row['TotalSimTime'])}", 
                xy=(row['Nodes'], row['TotalSimTime']),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightgray', ec='gray', alpha=0.8))

# Second plot: Nodes vs. Speedup
ax2.plot(data['Nodes'], data['Speedup'], 'r-', marker='o')
ax2.set_title('Nodes vs. Speedup', fontsize=14)
ax2.set_xlabel('Nodes', fontsize=12)
ax2.set_ylabel('Speedup', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

ax2.set_xscale('log', base=2)
ax2.set_xticks(data['Nodes'].tolist())
ax2.set_xticklabels([str(n) for n in data['Nodes']])

# Format y-axis with 'x' suffix
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}x"))

# Annotations
for _, row in data.iterrows():
    ax2.annotate(f"{row['Speedup']:.1f}", 
                xy=(row['Nodes'], row['Speedup']),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightgray', ec='gray', alpha=0.8))

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("speedup.png")
plt.show()