import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import grad, jit
import time
import jax.numpy as jnp
from typing import Optional, Tuple
class ADAM:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        self.param_history = []  # Store parameter history for tracking

    def update(self, gradients, params):
        if self.m is None:
            self.m = jnp.zeros_like(params)
        if self.v is None:
            self.v = jnp.zeros_like(params)
        
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        
        # Bias-corrected moments
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        #self.param_history.append(params.copy())
        #print(f"Iteration {self.t}, Parameters: {params}")
        params -= self.lr * m_hat / (jnp.sqrt(v_hat) + self.epsilon)
        return params


  


class FastGPAdam:
    """
    Highly optimized ADAM implementation for GP hyperparameter optimization.
    Key optimizations:
    1. Pre-compiled update function using JAX
    2. Minimized memory allocations
    3. Simplified state management
    4. Removed unnecessary computations
    """
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate
        # Fixed values commonly used in ADAM - no need to make configurable for GP
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        
        # Single initialization of state
        self.t = 0
        self.m = None
        self.v = None
        
        # JIT compile the core update computation
        self._update_step = jit(self._compute_update)
    
    @staticmethod
    @jit
    def _compute_update(m: jnp.ndarray, v: jnp.ndarray, grad: jnp.ndarray, 
                       beta1: float, beta2: float, t: int, lr: float, 
                       eps: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Core update computation, JIT compiled for speed.
        Returns updated moments directly.
        """
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        
        # Compute bias corrections and update in one step
        correction1 = 1 - beta1 ** t
        correction2 = 1 - beta2 ** t
        step = lr * (m / correction1) / (jnp.sqrt(v / correction2) + eps)
        
        return m, v, step

    def update(self, grad: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """
        Single optimization step.
        Minimized operations and memory allocations.
        """
        if self.m is None:
            # Initialize state vectors just once
            self.m = jnp.zeros_like(grad)
            self.v = jnp.zeros_like(grad)
        
        self.t += 1
        # Compute update in a single JIT-compiled step
        self.m, self.v, step = self._update_step(
            self.m, self.v, grad, self.beta1, self.beta2, 
            self.t, self.lr, self.epsilon)
        
        # Update parameters ensuring positivity
        return jnp.maximum(params - step, 1e-8)
    

class RBF_Kernel:
    def __init__(self, X1, X2=None):
        self.X1 = X1
        self.X2 = X2 if X2 is not None else X1

    def make_kernel(self, l=1, sigma=1e-8):
        n1, n2 = self.X1.shape[0], self.X2.shape[0]
        kernel = jnp.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                result = jnp.exp(-jnp.linalg.norm(self.X1[i] - self.X2[j])**2 / (2 * l**2))
                kernel = kernel.at[i, j].set(result)
        return kernel + sigma * jnp.eye(n1) if n1 == n2 else kernel

class GaussianProcess:
    def __init__(self, X, y, optimizer=ADAM, l=1.0, sigma_n=1e-8, sigma_f=1.0):
        self.X = X
        self.y = y
        self.optimizer = optimizer
        self.l = l  # Initial length scale
        self.sigma_n = sigma_n  # Initial noise variance
        self.sigma_f = sigma_f  # Initial signal variance
        self.K = self.compute_kernel(self.X, self.l, self.sigma_n)

    def compute_kernel(self, X, l, sigma_n):
        rbf_kernel = RBF_Kernel(X)
        return rbf_kernel.make_kernel(l=l, sigma=sigma_n)

    def negative_log_likelihood(self, params):
        l, sigma_f, sigma_n = params
        K = self.compute_kernel(self.X, l, sigma_n) + sigma_f**2 * jnp.eye(len(self.X))
        L = jnp.linalg.cholesky(K + 1e-6 * jnp.eye(len(self.X)))  # Use JAX's Cholesky

        alpha = jax.scipy.linalg.solve_triangular(L.T, jax.scipy.linalg.solve_triangular(L, self.y, lower=True), lower=False)

        return 0.5 * jnp.dot(self.y.T, alpha) + jnp.sum(jnp.log(jnp.diagonal(L))) + 0.5 * len(self.X) * jnp.log(2 * jnp.pi)

    def fit(self):
        self.l = np.random.uniform(0.1, 2.0)         # Random length scale within [0.1, 2.0]
        self.sigma_f = np.random.uniform(0.1, 2.0)   # Random signal variance within [0.1, 2.0]
        self.sigma_n = np.random.uniform(1e-9, 1e-6) # Random noise variance within [1e-9, 1e-6]
        if self.optimizer == "L-BFGS-B":
            result = minimize(self.negative_log_likelihood, x0=[self.l, self.sigma_f, self.sigma_n], method=self.optimizer)
            self.l, self.sigma_f, self.sigma_n = result.x

        elif self.optimizer == "CG":
            result = minimize(self.negative_log_likelihood, x0=[self.l, self.sigma_f, self.sigma_n], method=self.optimizer)
            self.l, self.sigma_f, self.sigma_n = result.x
            
        elif self.optimizer == "ADAM":
            self.adam_optimizer = GPAdam(learning_rate=0.001)
            result = self.optimise()
            self.l, self.sigma_f, self.sigma_n = result
            
        

        self.K = self.compute_kernel(self.X, self.l, self.sigma_n) + self.sigma_f**2 * jnp.eye(len(self.X))
        print(f"Training complete. Optimal length scale: {self.l:.2f}, Signal variance: {self.sigma_f:.2f}, Noise variance: {self.sigma_n:.2f}")


    def predict(self, X_new):
        K_new = self.compute_kernel(np.vstack([self.X, X_new]), self.l, self.sigma_n)
        K = K_new[:len(self.X), :len(self.X)] 
        k = K_new[:len(self.X), -1]  
        # Cholesky decomposition of the kernel matrix K
        L = np.linalg.cholesky(K + 1e-6 * np.eye(len(self.X)))  # Add jitter for stability
        # Solve for alpha using forward and backward substitution
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
        # Solve for v = L^(-1) * k
        v = np.linalg.solve(L, k)
        # Compute the mean prediction for X_new
        mean = np.dot(k.T, alpha)    
        # Compute the variance of the prediction for X_new
        var = K_new[-1, -1] - np.dot(v.T, v)
        
        return mean, var
    
    def optimise(self, max_iter=10000, Plot_Hist=False):
        # Set initial parameters as a JAX array
        params = jnp.array([self.l, self.sigma_f, self.sigma_n])

        # JIT compile the value and gradient computation for efficiency
        value_and_grad = jax.jit(jax.value_and_grad(self.negative_log_likelihood))

        # Track loss history if needed for plotting
        loss_history = []
        time_history_ADAM = []
        start = time.time()

        # Set up time budget
        time_budget = 10.0
        for i in range(max_iter):
            # Compute loss and gradients
            val, gradients = value_and_grad(params)
            # Update parameters using ADAM
            params = self.adam_optimizer.update(gradients, params)
            
            # Track loss and time
            loss_history.append(val)
            time_history_ADAM.append(time.time() - start)

            if time.time() - start > time_budget:
                break

        # Plot the loss history if requested
        if Plot_Hist:
            plt.figure(figsize=(10, 6))
            plt.plot(time_history_ADAM, loss_history, lw=2)
            plt.yscale("log")
            plt.legend()
            plt.xlabel("Time (s)")
            plt.ylabel("Loss")
            plt.grid()
            plt.title("Adam optimizer with Random Restart")
            plt.show()

        # Update final parameters with optimized values
        self.l, self.sigma_f, self.sigma_n = params
        return params

    def plot(self, f=None, x_range=(0, 10), n_points=1000, xlabel="$x$", ylabel="$f(x)$", title="GP Regression", legend_labels=None):
        # Generate the test points (x) in the specified range
        x = np.linspace(x_range[0], x_range[1], n_points)
        
        # Obtain the corresponding mean and standard deviations for predictions
        y_pred = []
        y_std = []
        for i in range(len(x)):
            X_new = np.array([x[i]]).reshape(-1, 1)  # Ensure the test point is in the correct shape
            mean_pred, sigma_pred = self.predict(X_new)
            y_pred.append(mean_pred)
            y_std.append(sigma_pred)
            
        y_pred = np.array(y_pred)
        y_std = np.array(y_std)
        
        # Plot the true function (if provided), GP predictions, and confidence intervals
        plt.figure(figsize=(10, 6))
        
        # Plot the true function if available
        if f:
            plt.plot(x, f(x), "r", label=legend_labels[0] if legend_labels else "True Function")
        
        # Plot the training observations
        plt.plot(self.X, self.y, "ro", label=legend_labels[1] if legend_labels else "Observations")
        
        # Plot the GP mean prediction
        plt.plot(x, y_pred, "b-", label=legend_labels[2] if legend_labels else "GP Mean")
        
        # Plot the 95% confidence interval (1.96 * standard deviation for ~95%)
        plt.fill_between(x, y_pred.ravel() - 1.96 * y_std.ravel(), y_pred.ravel() + 1.96 * y_std.ravel(),
                         color="lightblue", alpha=0.5, label=legend_labels[3] if legend_labels else "95% Confidence Interval")
        
        # Set plot labels and title
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16)
        
        # Show the legend
        plt.legend(fontsize=12)
        
        # Display grid and plot
        plt.grid(True)
        plt.show()

class SparseGaussianProcess:
    def __init__(self, X, y, Z, optimizer=ADAM, l=1.0, sigma_n=1e-8, sigma_f=1.0):
        self.X = X
        self.y = y
        self.Z = Z
        self.optimizer = optimizer
        self.l = l  # Initial length scale
        self.sigma_n = sigma_n  # Initial noise variance
        self.sigma_f = sigma_f  # Initial signal variance
        self.K = self.compute_kernel(self.X, self.l, self.sigma_n)

    def compute_kernel(self, Z, l, sigma_n):
        rbf_kernel = RBF_Kernel(Z)
        return rbf_kernel.make_kernel(l=l, sigma=sigma_n)

    def negative_log_likelihood(self, params):
        l, sigma_f, sigma_n = params
        K = self.compute_kernel(self.Z, l, sigma_n) + sigma_f**2 * jnp.eye(len(self.Z))
        L = jnp.linalg.cholesky(K + 1e-6 * jnp.eye(len(self.Z)))  # Use JAX's Cholesky
        
        alpha = jax.scipy.linalg.solve_triangular(L.T, jax.scipy.linalg.solve_triangular(L, self.y[:len(self.Z)], lower=True), lower=False)

        return 0.5 * jnp.dot(self.y[:len(self.Z)].T, alpha) + jnp.sum(jnp.log(jnp.diagonal(L))) + 0.5 * len(self.Z) * jnp.log(2 * jnp.pi)

    def fit(self):
        self.l = np.random.uniform(0.1, 2.0)         # Random length scale within [0.1, 2.0]
        self.sigma_f = np.random.uniform(0.1, 2.0)   # Random signal variance within [0.1, 2.0]
        self.sigma_n = np.random.uniform(1e-9, 1e-6) # Random noise variance within [1e-9, 1e-6]

        if self.optimizer == "L-BFGS-B":
            result = minimize(self.negative_log_likelihood, x0=[self.l, self.sigma_f, self.sigma_n], method=self.optimizer)
            self.l, self.sigma_f, self.sigma_n = result.x

        elif self.optimizer == "CG":
            result = minimize(self.negative_log_likelihood, x0=[self.l, self.sigma_f, self.sigma_n], method=self.optimizer)
            self.l, self.sigma_f, self.sigma_n = result.x
            
        elif self.optimizer == "ADAM":
            self.adam_optimizer = FastGPAdam(learning_rate=0.001)
            result = self.optimise()
            self.l, self.sigma_f, self.sigma_n = result
            
        elif self.optimizer == "ADAM-SCIPY":
            from scipy.optimize import minimize
            result = minimize(self.negative_log_likelihood, 
                            x0=[self.l, self.sigma_f, self.sigma_n],
                            method='adam',
                            options={'learning_rate': 0.001,
                                    'maxiter': 1000,
                                    'beta1': 0.9,
                                    'beta2': 0.999,
                                    'epsilon': 1e-8})
            self.l, self.sigma_f, self.sigma_n = result.x

        self.K = self.compute_kernel(self.Z, self.l, self.sigma_n) + self.sigma_f**2 * jnp.eye(len(self.Z))
        print(f"Training complete. Optimal length scale: {self.l:.2f}, Signal variance: {self.sigma_f:.2f}, Noise variance: {self.sigma_n:.2f}")


    def predict(self, X_new):
        K_new = self.compute_kernel(np.vstack([self.Z, X_new]), self.l, self.sigma_n)
        K = K_new[:len(self.Z), :len(self.Z)] 
        k = K_new[:len(self.Z), -1]  
        # Cholesky decomposition of the kernel matrix K
        L = np.linalg.cholesky(K + 1e-6 * np.eye(len(self.Z)))  # Add jitter for stability
        # Solve for alpha using forward and backward substitution
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y[:len(self.Z)]))
        # Solve for v = L^(-1) * k
        v = np.linalg.solve(L, k)
        # Compute the mean prediction for X_new
        mean = np.dot(k.T, alpha)    
        # Compute the variance of the prediction for X_new
        var = K_new[-1, -1] - np.dot(v.T, v)
        
        return mean, var
    
    def optimise(self, max_iter=10000, Plot_Hist=False):
        # Set initial parameters as a JAX array
        params = jnp.array([self.l, self.sigma_f, self.sigma_n])

        # JIT compile the value and gradient computation for efficiency
        value_and_grad = jax.jit(jax.value_and_grad(self.negative_log_likelihood))

        # Track loss history if needed for plotting
        loss_history = []
        time_history_ADAM = []
        start = time.time()

        # Set up time budget
        time_budget = 10.0
        for i in range(max_iter):
            # Compute loss and gradients
            val, gradients = value_and_grad(params)
            # Update parameters using ADAM
            params = self.adam_optimizer.update(gradients, params)
            
            # Track loss and time
            loss_history.append(val)
            time_history_ADAM.append(time.time() - start)

            if time.time() - start > time_budget:
                break

        # Plot the loss history if requested
        if Plot_Hist:
            plt.figure(figsize=(10, 6))
            plt.plot(time_history_ADAM, loss_history, lw=2)
            plt.yscale("log")
            plt.legend()
            plt.xlabel("Time (s)")
            plt.ylabel("Loss")
            plt.grid()
            plt.title("Adam optimizer with Random Restart")
            plt.show()

        # Update final parameters with optimized values
        self.l, self.sigma_f, self.sigma_n = params
        return params
    
    def plot(self, f=None, x_range=(0, 10), n_points=1000, xlabel="$x$", ylabel="$f(x)$", title="GP Regression", legend_labels=None):
        # Generate the test points (x) in the specified range
        x = np.linspace(x_range[0], x_range[1], n_points)
        
        # Obtain the corresponding mean and standard deviations for predictions
        y_pred = []
        y_std = []
        for i in range(len(x)):
            X_new = np.array([x[i]]).reshape(-1, 1)  # Ensure the test point is in the correct shape
            mean_pred, sigma_pred = self.predict(X_new)
            y_pred.append(mean_pred)
            y_std.append(sigma_pred)
            
        y_pred = np.array(y_pred)
        y_std = np.array(y_std)
        
        # Plot the true function (if provided), GP predictions, and confidence intervals
        plt.figure(figsize=(10, 6))
        
        # Plot the true function if available
        if f:
            plt.plot(x, f(x), "r", label=legend_labels[0] if legend_labels else "True Function")
        
        # Plot the training observations
        plt.plot(self.X, self.y, "ro", label=legend_labels[1] if legend_labels else "Observations")
        
        # Plot the GP mean prediction
        plt.plot(x, y_pred, "b-", label=legend_labels[2] if legend_labels else "GP Mean")
        
        # Plot the 95% confidence interval (1.96 * standard deviation for ~95%)
        plt.fill_between(x, y_pred.ravel() - 1.96 * y_std.ravel(), y_pred.ravel() + 1.96 * y_std.ravel(),
                         color="lightblue", alpha=0.5, label=legend_labels[3] if legend_labels else "95% Confidence Interval")
        
        # Set plot labels and title
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16)
        
        # Show the legend
        plt.legend(fontsize=12)
        
        # Display grid and plot
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Sample data: a simple sinusoidal function with noise
    np.random.seed(42)
    # Select a subset of X_train as inducing points


    n_points = 20
    X_train = np.random.uniform(-5, 5, n_points).reshape(-1, 1)  # 20 points between -5 and 5
    y_train = np.sin(X_train) + 0.2 * np.random.randn(n_points, 1)  # Sinusoidal with noise

    # Points to predict (for visualization)
    X_test = np.linspace(-6, 6, 100).reshape(-1, 1)
    n_inducing_points = 10  # or any other number less than n_points
    Z = X_train[np.random.choice(len(X_train), n_inducing_points, replace=False)]
    # Initialize and fit the Gaussian Process model
    gp = SparseGaussianProcess(X_train, y_train,Z, optimizer="CG")
    gp.fit()

    # Predict the mean and standard deviation for the test points
    means = []
    stds = []
    for x in X_test:
        mean, std = gp.predict(x.reshape(-1, 1))
        means.append(mean)
        stds.append(std)

    means = np.array(means)
    stds = np.array(stds)

    print("Means:", means)
    print("Standard Deviations:", stds)

    # Plot the GP regression results
    '''gp.plot(f=true_function, x_range=(-5, 5), n_points=1000, 
            xlabel="X", ylabel="y", 
            title="GP Regression on Sinusoidal Data", 
            legend_labels=["True Function (sin)", "Training Data", "GP Mean", "95% Confidence Interval"])'''