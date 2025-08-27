# Gaussian Process Models: A Comparative Analysis

**Course Project - DSA4212: Optimization for Machine Learning**  
**National University of Singapore (Exchange Program)**  
**Author: Albin Liljefors**

## Project Overview

This project presents a comprehensive comparative analysis of **Gaussian Process Regression (GPR)** and **Sparse Gaussian Process Regression (SGPR)** models, evaluating their performance across different optimization algorithms and dataset sizes. The study focuses on computational efficiency, prediction accuracy, uncertainty quantification, and model calibration.

## Key Contributions

- **Custom Implementation**: Built from-scratch implementations of both GPR and SGPR models using JAX for automatic differentiation
- **Optimization Comparison**: Evaluated three optimization algorithms: L-BFGS-B, Conjugate Gradient (CG), and ADAM
- **Comprehensive Analysis**: Conducted systematic performance evaluation across multiple metrics including RMSE, computational time, memory usage, and uncertainty calibration
- **Statistical Validation**: Performed correlation analysis between prediction errors and uncertainty estimates

## Technical Implementation

### Core Models
- **Gaussian Process Regression (GPR)**: Full covariance matrix implementation with O(n³) complexity
- **Sparse Gaussian Process Regression (SGPR)**: Inducing point approximation reducing complexity to O(m²n) where m << n

### Optimization Algorithms
1. **L-BFGS-B**: Quasi-Newton method with box constraints
2. **Conjugate Gradient (CG)**: First-order optimization method
3. **ADAM**: Adaptive moment estimation with custom JAX implementation

### Key Features
- RBF (Radial Basis Function) kernel implementation
- Hyperparameter optimization for length scale, signal variance, and noise variance
- Uncertainty quantification and calibration analysis
- Memory-efficient sparse approximations

## Experimental Results

### Performance Highlights
- **SGPR** demonstrates superior computational efficiency with 10x faster training times
- **GPR** achieves better prediction accuracy on smaller datasets (< 50 samples)
- **L-BFGS-B** optimizer consistently outperforms CG and ADAM across both models
- Strong correlation between prediction errors and uncertainty estimates validates model reliability

### Key Findings
- Sparse approximation maintains competitive accuracy while dramatically reducing computational cost
- Model calibration analysis shows well-calibrated uncertainty estimates
- Memory usage scales quadratically with dataset size for GPR vs. linearly for SGPR

## Repository Structure

```
├── gaussian_process_models.py          # Core GPR and SGPR implementations
├── model_comparison_analysis.ipynb     # Interactive comparison notebook
├── overall_results_analysis.py         # Comprehensive analysis script
├── uncertainty_analysis.py             # Uncertainty correlation analysis
├── experimental_results.csv            # Complete experimental data
├── requirements.txt                     # Python dependencies
├── DSA4212_Gaussian_Processes-2.pdf   # Technical report
└── Figures/Final/                      # Publication-ready visualizations
    ├── accuracy_analysis.pdf
    ├── performance_summary.pdf
    ├── uncertainty_correlation_analysis.pdf
    └── ... (additional analysis plots)
```

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Analysis
```python
# Generate all performance plots
python overall_results_analysis.py

# Run uncertainty correlation analysis
python uncertainty_analysis.py

# Interactive model comparison
jupyter notebook model_comparison_analysis.ipynb
```

## Technical Skills Demonstrated

- **Machine Learning**: Gaussian Processes, Bayesian inference, uncertainty quantification
- **Optimization**: Multiple optimization algorithms, hyperparameter tuning
- **Programming**: Python, JAX, NumPy, SciPy, object-oriented design
- **Data Analysis**: Statistical analysis, visualization, experimental design
- **Research**: Systematic evaluation, technical writing, reproducible research

## Academic Context

This project was completed as part of the **DSA4212: Optimization for Machine Learning** course at the National University of Singapore during an exchange program. The work demonstrates advanced understanding of:

- Bayesian machine learning methods
- Computational optimization techniques
- Statistical model evaluation
- Scientific computing with modern frameworks

## Results Visualization

The project includes comprehensive visualizations covering:
- Model accuracy comparisons across dataset sizes
- Computational efficiency analysis
- Memory usage scaling behavior
- Uncertainty calibration assessment
- Error-uncertainty correlation analysis

---

*This project showcases proficiency in advanced machine learning techniques, optimization methods, and rigorous experimental evaluation - skills directly applicable to graduate-level research and industry applications.*
