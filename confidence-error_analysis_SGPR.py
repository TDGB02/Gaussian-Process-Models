import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()
# Read the data
df = pd.read_csv('results_albin.csv')

# Create figure for SGPR analysis
plt.figure(figsize=(15, 6))

# Plot for SGPR ADAM
ax1 = plt.subplot(121)
sgpr_adam = df[df['Model'] == 'SGPR ADAM']

# Create scatter with colors based on dataset size
scatter = ax1.scatter(sgpr_adam['RMSE'], 
                     sgpr_adam['Avg Uncertainty'],
                     c=sgpr_adam['Dataset Size'],
                     cmap='viridis',
                     s=100,
                     alpha=0.7)

# Add dataset size annotations
for _, row in sgpr_adam.iterrows():
    ax1.annotate(f"n={int(row['Dataset Size'])}", 
                (row['RMSE'], row['Avg Uncertainty']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8)

ax1.set_xscale('log')
ax1.set_xlabel('RMSE (log scale)')
ax1.set_ylabel('Average Uncertainty')
ax1.set_title('SGPR ADAM: Error vs Uncertainty\n(color indicates dataset size)')
plt.colorbar(scatter, ax=ax1, label='Dataset Size')
ax1.grid(True, alpha=0.3)

# Plot for SGPR L-BFGS-B
ax2 = plt.subplot(122)
sgpr_lbfgs = df[df['Model'] == 'SGPR L-BFGS-B']

scatter = ax2.scatter(sgpr_lbfgs['RMSE'], 
                     sgpr_lbfgs['Avg Uncertainty'],
                     c=sgpr_lbfgs['Dataset Size'],
                     cmap='viridis',
                     s=100,
                     alpha=0.7)

# Add dataset size annotations
for _, row in sgpr_lbfgs.iterrows():
    ax2.annotate(f"n={int(row['Dataset Size'])}", 
                (row['RMSE'], row['Avg Uncertainty']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8)

ax2.set_xscale('log')
ax2.set_xlabel('RMSE (log scale)')
ax2.set_ylabel('Average Uncertainty')
ax2.set_title('SGPR L-BFGS-B: Error vs Uncertainty\n(color indicates dataset size)')
plt.colorbar(scatter, ax=ax2, label='Dataset Size')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Figures/sgpr_detailed_analysis.png', bbox_inches='tight', dpi=300)
plt.close()

# Print detailed analysis
for model in ['SGPR ADAM', 'SGPR L-BFGS-B']:
    model_data = df[df['Model'] == model]
    
    # Group by high/low error
    median_rmse = model_data['RMSE'].median()
    high_error = model_data[model_data['RMSE'] > median_rmse]
    low_error = model_data[model_data['RMSE'] <= median_rmse]
    
    print(f"\n{model} Analysis:")
    print("-" * 50)
    print(f"Average uncertainty for high errors (RMSE > {median_rmse:.2f}): {high_error['Avg Uncertainty'].mean():.3f}")
    print(f"Average uncertainty for low errors (RMSE <= {median_rmse:.2f}): {low_error['Avg Uncertainty'].mean():.3f}")
    
    # Analyze trend with dataset size
    print("\nUncertainty vs Dataset Size correlation:", 
          np.corrcoef(model_data['Dataset Size'], model_data['Avg Uncertainty'])[0,1].round(3))
    print("RMSE vs Dataset Size correlation:", 
          np.corrcoef(model_data['Dataset Size'], model_data['RMSE'])[0,1].round(3))