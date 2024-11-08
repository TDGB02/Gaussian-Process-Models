import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
# Read the data
df = pd.read_csv('results_albin.csv')
sns.set_theme()

# Plot 1: RMSE vs Dataset Size
plt.figure(figsize=(12, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
for i, (model, group) in enumerate(df.groupby('Model')):
    plt.plot(group['Dataset Size'], group['RMSE'], 'o-', 
            label=model, color=colors[i % len(colors)])
plt.xlabel('Dataset Size')
plt.ylabel('RMSE')
plt.title('Model Accuracy vs Dataset Size')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('Figures/accuracy_analysis.pdf', bbox_inches='tight', dpi=300)
plt.close()

# Plot 2: Training Time vs Dataset Size (two separate plots)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot for GPR ADAM (separate scale)
gpr_adam_data = df[df['Model'] == 'GPR ADAM']
ax1.plot(gpr_adam_data['Dataset Size'], gpr_adam_data['Time'], 'o-', 
         label='GPR ADAM', color='#1f77b4')
ax1.set_xlabel('Dataset Size')
ax1.set_ylabel('Training Time (s)')
ax1.set_title('Training Time: GPR ADAM')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot for other models
other_models = df[df['Model'] != 'GPR ADAM']
for i, (model, group) in enumerate(other_models.groupby('Model')):
    ax2.plot(group['Dataset Size'], group['Time'], 'o-', 
             label=model, color=colors[(i+1) % len(colors)])
ax2.set_xlabel('Dataset Size')
ax2.set_ylabel('Training Time (s)')
ax2.set_title('Training Time: Other Models')
ax2.grid(True, alpha=0.3)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('Figures/time_analysis.pdf', bbox_inches='tight', dpi=300)
plt.close()

# Plot 3: Memory Usage vs Dataset Size
plt.figure(figsize=(12, 8))
for i, (model, group) in enumerate(df.groupby('Model')):
    plt.plot(group['Dataset Size'], group['Memory Usage'], 'o-', 
            label=model, color=colors[i % len(colors)])
plt.xlabel('Dataset Size')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage vs Dataset Size')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('Figures/memory_analysis.pdf', bbox_inches='tight', dpi=300)
plt.close()

# Plot 4: Improved Error vs Uncertainty Visualization
plt.figure(figsize=(12, 8))

# Create a scatter plot with size proportional to dataset size
for i, (model, group) in enumerate(df.groupby('Model')):
    sizes = group['Dataset Size'] * 2  # Scale factor for visibility
    plt.scatter(group['RMSE'], group['Avg Uncertainty'], 
               s=sizes, label=model, color=colors[i % len(colors)], alpha=0.6)
    
    # Add trend line
    z = np.polyfit(group['RMSE'], group['Avg Uncertainty'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(group['RMSE'].min(), group['RMSE'].max(), 100)
    plt.plot(x_trend, p(x_trend), '--', color=colors[i % len(colors)], alpha=0.3)

plt.xlabel('RMSE')
plt.ylabel('Average Uncertainty')
plt.title('Error vs Uncertainty Relationship\n(Point size indicates dataset size)')
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('Figures/uncertainty_analysis.pdf', bbox_inches='tight', dpi=300)
plt.close()

# Additional plot: Uncertainty distribution boxplot
plt.figure(figsize=(12, 6))
df.boxplot(column='Avg Uncertainty', by='Model', rot=45)
plt.title('Distribution of Uncertainty Estimates by Model')
plt.suptitle('')  # Remove automatic suptitle
plt.ylabel('Average Uncertainty')
plt.tight_layout()
plt.savefig('Figures/uncertainty_distribution.pdf', bbox_inches='tight', dpi=300)
plt.close()

print("Plots have been generated!")