import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
import seaborn as sns
sns.set_theme()

# Set font size
font_size = 20
plt.rcParams.update({
    'font.size': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'legend.fontsize': font_size,
    'figure.titlesize': font_size
})

# Create Figures directory if it doesn't exist
if not os.path.exists('Figures'):
    os.makedirs('Figures')

# Read the data
df = pd.read_csv('results_albin.csv')

# Calculate correlations between RMSE and Uncertainty for each model
correlations = {}
for model in df['Model'].unique():
    model_data = df[df['Model'] == model]
    corr, p_value = stats.spearmanr(model_data['RMSE'], model_data['Avg Uncertainty'])
    correlations[model] = {'correlation': corr, 'p_value': p_value}

# Create figure with two subplots: scatter plot and correlation summary
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), 
                               gridspec_kw={'width_ratios': [2, 1]})

# Plot 1: Scatter plot (similar to before but cleaner)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
for i, (model, group) in enumerate(df.groupby('Model')):
    sizes = group['Dataset Size'] * 2
    ax1.scatter(group['RMSE'], group['Avg Uncertainty'], 
                s=sizes, label=model, color=colors[i % len(colors)], alpha=0.6)

ax1.set_xlabel('RMSE (log scale)')
ax1.set_ylabel('Average Uncertainty')
ax1.set_title('Error vs Uncertainty Relationship\n(Circle size indicates dataset size)')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 2: Correlation summary
models = list(correlations.keys())
corr_values = [correlations[m]['correlation'] for m in models]
p_values = [correlations[m]['p_value'] for m in models]

# Create correlation bar plot
bars = ax2.barh(np.arange(len(models)), corr_values, 
                color=[colors[i % len(colors)] for i in range(len(models))])
ax2.set_yticks(np.arange(len(models)))
ax2.set_yticklabels(models)
ax2.set_xlabel('Spearman Correlation Coefficient')
ax2.set_title('Correlation between RMSE and Uncertainty')

# Add correlation values and significance stars
for i, (bar, p_val) in enumerate(zip(bars, p_values)):
    value = bar.get_width()
    ax2.text(value, i, 
             f' {value:.2f}{"*" if p_val < 0.05 else ""}', 
             va='center', fontsize=font_size)

ax2.grid(True, alpha=0.3)
ax2.set_xlim(-1, 1)  # Correlation ranges from -1 to 1

plt.tight_layout()
plt.savefig('Figures/Final/uncertainty_correlation_analysis.pdf', bbox_inches='tight', dpi=300)
plt.close()

# Print detailed correlation analysis
print("\nDetailed Correlation Analysis:")
print("-" * 50)
for model in correlations:
    print(f"\n{model}:")
    print(f"Spearman correlation: {correlations[model]['correlation']:.3f}")
    print(f"P-value: {correlations[model]['p_value']:.3f}")
