import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import grad, jit
import time

class ADAM:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.param_history = []  # Track parameter history

    def init_state(self, params):
        """Initialize the optimizer state with zeros for m, v, and set timestep t=0."""
        m = jnp.zeros_like(params)
        v = jnp.zeros_like(params)
        t = 0
        return m, v, t

    def update(self, params, grads, state):
        """Compute the ADAM parameter update and record parameter history."""
        m, v, t = state
        t += 1
        m = self.beta1 * m + (1 - self.beta1) * grads
        v = self.beta2 * v + (1 - self.beta2) * grads ** 2
        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)
        params = params - self.lr * m_hat / (jnp.sqrt(v_hat) + self.epsilon)
        
        # Append updated parameters to history
        self.param_history.append(params)
        
        return params, (m, v, t)

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
    def __init__(self, X, y, optimizer="ADAM", l=1.0, sigma_n=1e-8, sigma_f=1.0):
        self.X = X
        self.y = y
        self.optimizer = optimizer
        self.l = l  # Initial length scale
        self.sigma_n = sigma_n  # Initial noise variance
        self.sigma_f = sigma_f  # Initial signal variance
        self.K = self.compute_kernel(self.X, self.l, self.sigma_n)

        # Initialize ADAM optimizer if chosen
        if self.optimizer == "ADAM":
            self.adam_optimizer = ADAM(learning_rate=0.001)

    def compute_kernel(self, X, l, sigma_n):
        rbf_kernel = RBF_Kernel(X)
        return rbf_kernel.make_kernel(l=l, sigma=sigma_n)

    def negative_log_likelihood(self, params):
        l, sigma_f, sigma_n = params
        K = self.compute_kernel(self.X, l, sigma_n) + sigma_f**2 * jnp.eye(len(self.X))
        L = jnp.linalg.cholesky(K + 1e-6 * jnp.eye(len(self.X)))
        alpha = jax.scipy.linalg.solve_triangular(L.T, jax.scipy.linalg.solve_triangular(L, self.y, lower=True), lower=False)
        return 0.5 * jnp.dot(self.y.T, alpha) + jnp.sum(jnp.log(jnp.diagonal(L))) + 0.5 * len(self.X) * jnp.log(2 * jnp.pi)

    def optimise(self, max_iter=10000, tol=1e-6, patience=5):
        # Set initial parameters as a JAX array
        params = jnp.array([self.l, self.sigma_f, self.sigma_n])
        state = self.adam_optimizer.init_state(params)

        # JIT compile the value and gradient computation
        value_and_grad = jax.jit(jax.value_and_grad(self.negative_log_likelihood))

        # Initialize loss tracking for tolerance-based stopping
        prev_loss = float('inf')
        no_improve_steps = 0

        for i in range(max_iter):
            val, grads = value_and_grad(params)
            params, state = self.adam_optimizer.update(params, grads, state)
            
            # Track parameter history
            self.adam_optimizer.param_history.append(params)
            
            # Check if loss change is below tolerance
            if abs(prev_loss - val) < tol:
                no_improve_steps += 1
                if no_improve_steps >= patience:
                    print(f"Convergence reached at iteration {i} with minimal loss change.")
                    break
            else:
                no_improve_steps = 0  # Reset if loss change exceeds tolerance
            prev_loss = val

        # Update final model parameters with optimized values
        self.l, self.sigma_f, self.sigma_n = params
        return self.l, self.sigma_f, self.sigma_n

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
            self.l, self.sigma_f, self.sigma_n = self.optimise()

        self.K = self.compute_kernel(self.X, self.l, self.sigma_n) + self.sigma_f**2 * jnp.eye(len(self.X))
        print(f"Training complete. Optimal length scale: {self.l:.2f}, Signal variance: {self.sigma_f:.2f}, Noise variance: {self.sigma_n:.2f}")

    def predict(self, X_new):
        K_new = self.compute_kernel(np.vstack([self.X, X_new]), self.l, self.sigma_n)
        K = K_new[:len(self.X), :len(self.X)] 
        k = K_new[:len(self.X), -1]  
        L = np.linalg.cholesky(K + 1e-6 * np.eye(len(self.X)))  # Add jitter for stability
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
        v = np.linalg.solve(L, k)
        mean = np.dot(k.T, alpha)    
        var = K_new[-1, -1] - np.dot(v.T, v)
        
        return mean, var

    def plot(self, f=None, x_range=(0, 10), n_points=1000, xlabel="$x$", ylabel="$f(x)$", title="GP Regression", legend_labels=None):
        # Generate test points (x) in the specified range
        x = np.linspace(x_range[0], x_range[1], n_points).reshape(-1, 1)

        # Batch predictions for all points
        means, stds = zip(*(self.predict(xi.reshape(1, -1)) for xi in x))
        means = np.array(means).flatten()
        stds = np.array(stds).flatten()

        # Plotting
        plt.figure(figsize=(10, 6))

        # Plot the true function if available
        if f:
            plt.plot(x, f(x), "r", label=legend_labels[0] if legend_labels else "True Function")

        # Plot the training observations
        plt.plot(self.X, self.y, "ro", label=legend_labels[1] if legend_labels else "Observations")

        # Plot the GP mean prediction
        plt.plot(x, means, "b-", label=legend_labels[2] if legend_labels else "GP Mean")

        # Plot the 95% confidence interval (1.96 * std for ~95%)
        plt.fill_between(x.flatten(), means - 1.96 * stds, means + 1.96 * stds,
                         color="lightblue", alpha=0.5, label=legend_labels[3] if legend_labels else "95% Confidence Interval")

        # Set plot labels and title
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16)
        plt.legend(fontsize=12)
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
        # Initialize parameters with random values
        self.l = np.random.uniform(0.1, 2.0)
        self.sigma_f = np.random.uniform(0.1, 2.0)
        self.sigma_n = np.random.uniform(1e-9, 1e-6)

        if self.optimizer == "L-BFGS-B":
            result = minimize(self.negative_log_likelihood, x0=[self.l, self.sigma_f, self.sigma_n], method=self.optimizer)
            self.l, self.sigma_f, self.sigma_n = result.x

        elif self.optimizer == "CG":
            result = minimize(self.negative_log_likelihood, x0=[self.l, self.sigma_f, self.sigma_n], method=self.optimizer)
            self.l, self.sigma_f, self.sigma_n = result.x

        elif self.optimizer == "ADAM":
            self.adam_optimizer = ADAM(learning_rate=0.001)
            # Directly assign the consistent 3-value output from `optimise`
            self.l, self.sigma_f, self.sigma_n = self.optimise()

        # Recompute kernel after fitting
        self.K = self.compute_kernel(self.X, self.l, self.sigma_n) + self.sigma_f**2 * jnp.eye(len(self.X))
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
    
    
    def optimise(self, max_iter=10000, track_history=False):
        # Initialize parameters and optimizer state
        params = jnp.array([self.l, self.sigma_f, self.sigma_n])
        adam_optimizer = ADAM(learning_rate=0.001)
        state = adam_optimizer.init_state(params)

        # JIT compile the value and gradient computation
        value_and_grad = jax.jit(jax.value_and_grad(self.negative_log_likelihood))

        # Initialize param_history if tracking is enabled
        param_history = [] if track_history else None

        # Define the optimization step
        @jax.jit
        def optimization_step(carry, _):
            params, state = carry
            val, grads = value_and_grad(params)
            params, state = adam_optimizer.update(params, grads, state)
            return (params, state), val  # Return updated params and loss

        # Run the optimization loop with lax.scan
        (params, _), losses = jax.lax.scan(
            optimization_step,
            (params, state),
            jnp.arange(max_iter)
        )

        # Update param_history if tracking enabled
        if track_history:
            param_history.append(params)

        # Unpack and assign to object attributes for consistency
        self.l, self.sigma_f, self.sigma_n = params

        # Return a consistent output
        return (self.l, self.sigma_f, self.sigma_n)
        
        
    
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