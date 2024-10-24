import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import grad, jit


class ADAM:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

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
        params -= self.lr * m_hat / (jnp.sqrt(v_hat) + self.epsilon)
        
        return params

        
class RBF_Kernel:
    def __init__(self, X):
        self.X = X  
        self.kernel = jnp.zeros((X.shape[0], X.shape[0]))

    def make_kernel(self, l=1, sigma=0):
        for i in range(self.kernel.shape[0]):
            for j in range(self.kernel.shape[1]):
                self.kernel = self.kernel.at[i, j].set(self.Kernel(self.X[i], self.X[j])(l))

        return self.kernel + sigma * jnp.eye(len(self.kernel))

    class Kernel:
        def __init__(self, x_n, x_m):
            self.x_n = x_n
            self.x_m = x_m

        def __call__(self, l=1):
            # RBF Kernel using JAX
            K = jnp.exp(-jnp.linalg.norm(self.x_n - self.x_m)**2 / (2 * l**2))
            return K



class GaussianProcess:
    def __init__(self, X, y, optimizer, l=1.0, sigma_n=1e-8, sigma_f=1.0):
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
        if self.optimizer == "L-BFGS-B":
            result = minimize(self.negative_log_likelihood, x0=[self.l, self.sigma_f, self.sigma_n], 
                            bounds=[(1e-5, None), (1e-5, None), (1e-5, None)], method=self.optimizer)
            self.l, self.sigma_f, self.sigma_n = result.x
            
        elif self.optimizer == "ADAM":
            self.adam_optimizer = ADAM(learning_rate=0.01)
            result = self.optimise()
            self.l, self.sigma_f, self.sigma_n = result
            
        self.K = self.compute_kernel(self.X, self.l, self.sigma_n) + self.sigma_f**2 * jnp.eye(len(self.X))
        print(f"Training complete. Optimal length scale: {self.l:.2f}, Signal variance: {self.sigma_f:.2f}, Noise variance: {self.sigma_n:.2f}")


    def fit(self):
        # Optimize hyperparameters l, sigma_f, and sigma_n
        if self.optimizer == "L-BFGS-B":
            result = minimize(self.negative_log_likelihood, x0=[self.l, self.sigma_f, self.sigma_n], 
                            bounds=[(1e-5, None), (1e-5, None), (1e-5, None)], method = self.optimizer)
            self.l, self.sigma_f, self.sigma_n = result.x
        elif self.optimizer == "ADAM":
            self.adam_optimizer = ADAM(learning_rate=0.01)
            result = self.optimise()
            self.l, self.sigma_f, self.sigma_n = result
            
        self.K = self.compute_kernel(self.X, self.l, self.sigma_n) + self.sigma_f**2 * np.eye(len(self.X))
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
    
    def optimise(self, max_iter=1000):
        # Ensure the parameters are JAX arrays
        params = jnp.array([self.l, self.sigma_f, self.sigma_n])
        
        # JIT compile the value and gradient computation
        value_and_grad = jax.jit(jax.value_and_grad(self.negative_log_likelihood))

        for i in range(max_iter):
            val, gradients = value_and_grad(params)
            params = self.adam_optimizer.update(gradients, params)
            
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
    n_points = 20
    X_train = np.random.uniform(-5, 5, n_points).reshape(-1, 1)  # 20 points between -5 and 5
    y_train = np.sin(X_train) + 0.2 * np.random.randn(n_points, 1)  # Sinusoidal with noise

    # Points to predict (for visualization)
    X_test = np.linspace(-6, 6, 100).reshape(-1, 1)

    # Initialize and fit the Gaussian Process model
    gp = GaussianProcess(X_train, y_train)
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

    true_function = np.sin

    # Plot the GP regression results
    '''gp.plot(f=true_function, x_range=(-5, 5), n_points=1000, 
            xlabel="X", ylabel="y", 
            title="GP Regression on Sinusoidal Data", 
            legend_labels=["True Function (sin)", "Training Data", "GP Mean", "95% Confidence Interval"])'''