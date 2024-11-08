import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import os

def setup_plotting():
    """Setup plotting style and create Figures directory"""
    sns.set_theme()
    if not os.path.exists('Figures'):
        os.makedirs('Figures')
    return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

def plot_accuracy(df, colors):
    """Plot RMSE vs Dataset Size"""
    plt.figure(figsize=(12, 8))
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
    plt.savefig('Figures/Jonathan_results/accuracy_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_training_time(df, colors):
    """Plot Training Time vs Dataset Size (all models, log scale)"""
    plt.figure(figsize=(10, 6))
    
    # Plot all models
    for i, (model, group) in enumerate(df.groupby('Model')):
        plt.plot(group['Dataset Size'], group['Time'], 'o-', 
                label=model, color=colors[i % len(colors)])
    
    plt.xlabel('Dataset Size')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time vs Dataset Size')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    
    plt.tight_layout()
    plt.savefig('Figures/Jonathan_results/time_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_memory_usage(df, colors):
    """Plot Memory Usage vs Dataset Size"""
    plt.figure(figsize=(12, 8))
    for i, (model, group) in enumerate(df.groupby('Model')):
        plt.plot(group['Dataset Size'], group['Memory Usage'], 'o-', 
                label=model, color=colors[i % len(colors)])
    plt.xlabel('Dataset Size')
    plt.ylabel('Memory Usage (MB)')
    plt.yscale('log')
    plt.title('Memory Usage vs Dataset Size')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('Figures/Jonathan_results/memory_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_uncertainty_analysis(df, colors):
    """Plot Error vs Uncertainty Relationship"""
    plt.figure(figsize=(12, 8))
    for i, (model, group) in enumerate(df.groupby('Model')):
        sizes = group['Dataset Size'] * 2
        plt.scatter(group['RMSE'], group['Avg Uncertainty'], 
                   s=sizes, label=model, color=colors[i % len(colors)], alpha=0.6)
        
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
    plt.savefig('Figures/Jonathan_results/uncertainty_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_uncertainty_distribution(df):
    """Plot Distribution of Uncertainty Estimates"""
    plt.figure(figsize=(12, 6))
    df.boxplot(column='Avg Uncertainty', by='Model', rot=45)
    plt.title('Distribution of Uncertainty Estimates by Model')
    plt.suptitle('')
    plt.ylabel('Average Uncertainty')
    plt.tight_layout()
    plt.savefig('Figures/Jonathan_results/uncertainty_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_calibration_analysis(df):
    """Plot calibration analysis using 'In Interval' metric"""
    plt.figure(figsize=(12, 6))
    colors = setup_plotting()
    
    for i, (model, group) in enumerate(df.groupby('Model')):
        plt.scatter(group['Dataset Size'], group['In Interval'], 
                   label=model, color=colors[i % len(colors)], alpha=0.7)
    
    plt.axhline(y=0.95, color='r', linestyle='--', label='Expected (95%)')
    plt.xlabel('Dataset Size')
    plt.ylabel('Proportion within 1.96σ')
    plt.title('Model Calibration Analysis')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Figures/Jonathan_results/calibration_analysis.png', bbox_inches='tight')
    plt.close()

def plot_signal_variance(df):
    """Plot signal variance analysis"""
    plt.figure(figsize=(12, 6))
    colors = setup_plotting()
    
    for i, (model, group) in enumerate(df.groupby('Model')):
        plt.plot(group['Dataset Size'], group['Predictive Noise'], 
                'o-', label=model, color=colors[i % len(colors)])
    
    plt.xlabel('Dataset Size')
    plt.ylabel('Signal Variance')
    plt.title('Signal Variance vs Dataset Size')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Figures/Jonathan_results/signal_variance_analysis.png', bbox_inches='tight')
    plt.close()

def plot_performance_reliability(df):
    """
    Create comprehensive performance and reliability visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Top left: RMSE vs Dataset Size
    for i, (model, group) in enumerate(df.groupby('Model')):
        axes[0,0].plot(group['Dataset Size'], group['RMSE'], 'o-', 
                      label=model, color=colors[i % len(colors)])
    axes[0,0].set_yscale('log')
    axes[0,0].set_title('Prediction Error')
    axes[0,0].set_ylabel('RMSE')
    axes[0,0].set_xlabel('Dataset Size')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # Top right: Coverage vs Expected (95%)
    for i, (model, group) in enumerate(df.groupby('Model')):
        coverage_deviation = group['In Interval'] - 0.95  # Deviation from expected 95%
        axes[0,1].plot(group['Dataset Size'], coverage_deviation, 'o-', 
                      label=model, color=colors[i % len(colors)])
    axes[0,1].axhline(y=0, color='r', linestyle='--', label='Ideal Calibration')
    axes[0,1].set_title('Calibration Error')
    axes[0,1].set_ylabel('Coverage Deviation from 95%')
    axes[0,1].set_xlabel('Dataset Size')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # Bottom left: Relationship between Error and Uncertainty
    for i, (model, group) in enumerate(df.groupby('Model')):
        # Correlation between RMSE and Avg Uncertainty
        corr = np.corrcoef(group['RMSE'], group['Avg Uncertainty'])[0,1]
        axes[1,0].scatter(group['RMSE'], group['Avg Uncertainty'], 
                         label=f'{model} (corr={corr:.2f})', 
                         color=colors[i % len(colors)], alpha=0.6)
    axes[1,0].set_xscale('log')
    axes[1,0].set_title('Uncertainty vs Error')
    axes[1,0].set_xlabel('RMSE')
    axes[1,0].set_ylabel('Average Uncertainty')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Bottom right: Computational Efficiency (excluding GPR ADAM)
    gpr_adam_data = df[df['Model'] == 'GPR ADAM']
    other_models = df[df['Model'] != 'GPR ADAM']
    
    ax_time = axes[1,1]
    # Plot GPR ADAM on right y-axis
    ax_time2 = ax_time.twinx()
    
    # Plot other models on left y-axis
    for i, (model, group) in enumerate(other_models.groupby('Model')):
        ax_time.plot(group['Dataset Size'], group['Time'], 'o-', 
                    label=model, color=colors[(i+1) % len(colors)])
    
    # Plot GPR ADAM on right y-axis
    ax_time2.plot(gpr_adam_data['Dataset Size'], gpr_adam_data['Time'], 'o-',
                 label='GPR ADAM', color=colors[0], linestyle='--')
    
    ax_time.set_title('Computational Time')
    ax_time.set_xlabel('Dataset Size')
    ax_time.set_ylabel('Time (s) - Other Models')
    ax_time2.set_ylabel('Time (s) - GPR ADAM')
    
    # Combine legends
    lines1, labels1 = ax_time.get_legend_handles_labels()
    lines2, labels2 = ax_time2.get_legend_handles_labels()
    ax_time.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('Figures/Jonathan_results/performance_reliability.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_performance_summary(df):
    """Create comprehensive performance summary"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    colors = setup_plotting()
    
    for i, (model, group) in enumerate(df.groupby('Model')):
        color = colors[i % len(colors)]
        
        axes[0,0].plot(group['Dataset Size'], group['RMSE'], 'o-', label=model, color=color)
        axes[0,1].plot(group['Dataset Size'], group['In Interval'], 'o-', label=model, color=color)
        axes[1,0].plot(group['Dataset Size'], group['Time'], 'o-', label=model, color=color)
        axes[1,1].plot(group['Dataset Size'], group['Predictive Noise'], 'o-', label=model, color=color)
    
    # Customize subplots
    axes[0,0].set_ylabel('RMSE')
    axes[0,0].set_yscale('log')
    axes[0,0].set_title('Model Accuracy')
    axes[0,0].grid(True)
    
    axes[0,1].set_ylabel('Coverage')
    axes[0,1].axhline(y=0.95, color='r', linestyle='--', label='Expected')
    axes[0,1].set_title('Model Calibration')
    axes[0,1].grid(True)
    
    axes[1,0].set_ylabel('Time (s)')
    axes[1,0].set_yscale('log')
    axes[1,0].set_title('Computational Efficiency')
    axes[1,0].grid(True)
    
    axes[1,1].set_ylabel('Signal Variance')
    axes[1,1].set_title('Predictive Noise')
    axes[1,1].grid(True)
    
    for ax in axes.flat:
        ax.set_xlabel('Dataset Size')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('Figures/Jonathan_results/performance_summary.png', bbox_inches='tight')
    plt.close()

def main():
    """Main function to run all analyses"""
    # Read data
    df = pd.read_csv('results.csv')
    
    # Setup
    colors = setup_plotting()
    
    # Generate all plots
    plot_accuracy(df, colors)
    plot_training_time(df, colors)
    plot_memory_usage(df, colors)
    plot_uncertainty_analysis(df, colors)
    plot_uncertainty_distribution(df)
    plot_calibration_analysis(df)
    plot_signal_variance(df)
    plot_performance_summary(df)
    plot_performance_reliability(df)

    print("All plots have been generated in the Figures directory!")

if __name__ == "__main__":
    main()