import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import jax
import jax.numpy as jnp
from jax import grad, jit
from scipy.optimize import minimize
import time

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
    def __init__(self, X1, X2=None):
        self.X1 = jnp.array(X1)  # Ensure X1 is a JAX array
        self.X2 = jnp.array(X2) if X2 is not None else self.X1  # Ensure X2 is a JAX array

    def make_kernel(self, l=1, sigma=1e-8):
        n1, n2 = self.X1.shape[0], self.X2.shape[0]
        kernel = jnp.zeros((n1, n2))  # Use jnp.zeros instead of np.zeros
        for i in range(n1):
            for j in range(n2):
                # Use jnp.linalg.norm and ensure the difference is a JAX array
                kernel = kernel.at[i, j].set(
                    jnp.exp(-jnp.linalg.norm(self.X1[i] - self.X2[j]) ** 2 / (2 * l ** 2))
                )  # Use jnp.norm instead of np.linalg.norm
        return kernel + sigma * jnp.eye(n1) if n1 == n2 else kernel



class SparseGaussianProcess:
    def __init__(self, X, y, Z):
        self.X = jnp.array(X)  # Ensure X is a JAX array
        self.y = jnp.array(y)  # Ensure y is a JAX array
        self.Z = jnp.array(Z)  # Ensure Z is a JAX array

    def predict_mean(self, X_new):
        K_zz = RBF_Kernel(self.Z).make_kernel()
        K_xz = RBF_Kernel(self.X, self.Z).make_kernel()
        K_zx = K_xz.T
        K_xx = RBF_Kernel(self.X).make_kernel()
        
        alpha = jnp.linalg.solve(K_zz + K_zx @ jnp.linalg.solve(K_xx, K_xz), K_zx @ jnp.linalg.solve(K_xx, self.y))
        k_new_z = RBF_Kernel(X_new, self.Z).make_kernel()
        return jnp.dot(k_new_z, alpha)

    def predict_std(self, X_new):
        K_zz = RBF_Kernel(self.Z).make_kernel()
        K_xz = RBF_Kernel(self.X, self.Z).make_kernel()
        K_zx = K_xz.T
        K_xx = RBF_Kernel(self.X).make_kernel()
        
        k_new_z = RBF_Kernel(X_new, self.Z).make_kernel()
        k_new_new = RBF_Kernel(X_new).make_kernel()
        
        K_inv = jnp.linalg.inv(K_zz + jnp.dot(K_zx, jnp.linalg.solve(K_xx, K_xz)))
        return k_new_new - jnp.dot(k_new_z, jnp.dot(K_inv, k_new_z.T))

    def negative_log_likelihood(self, params):
        
        l = params[0]
        sigma_n = params[1]

        K_xx = RBF_Kernel(self.X).make_kernel(l=l, sigma=sigma_n)  # Shape (5, 5)
        K_zx = RBF_Kernel(self.Z, self.X).make_kernel(l=l, sigma=sigma_n)  # Shape (3, 5)
        K_zz = RBF_Kernel(self.Z).make_kernel(l=l, sigma=sigma_n)  # Shape (3, 3)

        # Check shapes
        print(f"K_xx shape: {K_xx.shape}")
        print(f"K_zx shape: {K_zx.shape}")
        print(f"K_zz shape: {K_zz.shape}")

        # Compute the necessary matrices
        K_zx_inv_K_xx = K_zx @ jnp.linalg.inv(K_xx)  # Shape (3, 5)
        K_term = K_zx_inv_K_xx @ K_zx.T  # Should yield shape (5, 5)

        # Make sure this addition is valid
        K_combined = K_zz + K_term  # Shape (3, 3)

        # Check if K_combined is singular using jax.lax.cond
        det_value = jnp.linalg.det(K_combined)
        
        def singularity_warning():
            print("Warning: K_combined is singular!")
            return jnp.eye(K_combined.shape[0]) * 1e-8  # Return a small perturbation to avoid singularity

        K_inv = jax.lax.cond(
            jnp.isclose(det_value, 0),
            singularity_warning,  # Call this function if true
            lambda: jnp.linalg.inv(K_combined)  # Otherwise, compute the inverse normally
        )

        # Return a scalar value for the negative log likelihood
        log_likelihood = -0.5 * jnp.log(det_value) - 0.5 * (self.y.T @ K_inv @ self.y)
        return log_likelihood



    def fit(self, max_iter=10000):
        # Initialize parameters
        params = jnp.array([1.0, 1.0, 1e-8])  # length scale, signal variance, noise variance

        # JIT compile the value and gradient computation
        value_and_grad = jax.jit(jax.value_and_grad(self.negative_log_likelihood))
        for _ in range(max_iter):
            val, gradients = value_and_grad(params)
            params = self.adam_optimizer.update(gradients, params)

        self.l, self.sigma_f, self.sigma_n = params
        
    def predict_mean(self, X_new):
        K_zz = RBF_Kernel(self.Z).make_kernel(l=self.l)
        K_xz = RBF_Kernel(self.X, self.Z).make_kernel(l=self.l)
        K_zx = K_xz.T
        K_xx = RBF_Kernel(self.X).make_kernel(l=self.l)

        alpha = np.linalg.solve(K_zz + K_zx @ np.linalg.inv(K_xx) @ K_xz, K_zx @ np.linalg.solve(K_xx, self.y))
        k_new_z = RBF_Kernel(X_new, self.Z).make_kernel(l=self.l)
        return np.dot(k_new_z, alpha)
    
    def predict_std(self, X_new):
        K_zz = RBF_Kernel(self.Z).make_kernel(l=self.l)
        K_xz = RBF_Kernel(self.X, self.Z).make_kernel(l=self.l)
        K_zx = K_xz.T
        K_xx = RBF_Kernel(self.X).make_kernel(l=self.l)
        
        k_new_z = RBF_Kernel(X_new, self.Z).make_kernel(l=self.l)
        k_new_new = RBF_Kernel(X_new).make_kernel(l=self.l)
        
        K_inv = np.linalg.inv(K_zz + np.dot(K_zx, np.linalg.solve(K_xx, K_xz)))
        return k_new_new - np.dot(k_new_z, np.dot(K_inv, k_new_z.T))

    def plot(self, f):
        x = np.linspace(0, 10, 1000).reshape(-1, 1)
        y_pred, y_std = [], []
        for X_new in x:
            y_pred.append(self.predict_mean(X_new.reshape(1, -1)))
            y_std.append(np.sqrt(self.predict_std(X_new.reshape(1, -1))))
        
        y_pred, y_std = np.array(y_pred).flatten(), np.array(y_std).flatten()
        plt.figure(figsize=(15, 5))
        plt.plot(x, f(x), "r")
        plt.plot(self.X, self.y, "ro")
        plt.plot(x, y_pred, "b-")
        plt.fill(np.hstack([x.flatten(), x.flatten()[::-1]]),
                 np.hstack([y_pred - 1.9600 * y_std, (y_pred + 1.9600 * y_std)[::-1]]),
                 alpha=0.5, fc="b")
        plt.xlabel("$x$", fontsize=14)
        plt.ylabel("$f(x)$", fontsize=14)
        plt.legend(["$y = f(x)$", "Observations", "Predictions", "95% Confidence Interval"], fontsize=14)
        plt.grid(True)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

def f(x):
    return (x - 5) ** 2

if __name__ == "__main__":
    X = np.array([1.0, 3.0, 5.0, 7.0, 9.0]).reshape(-1, 1)
    y = f(X)
    X_new = np.array([[5.5]])
    
    # Sparse Gaussian Process with inducing points
    n_inducing_points = 3
    kmeans = KMeans(n_clusters=n_inducing_points, random_state=0).fit(X)
    Z = kmeans.cluster_centers_
    
    sgp = SparseGaussianProcess(X, y, Z)
    sgp.fit()
    mean_pred_sparse = sgp.predict_mean(X_new)
    sigma_pred_sparse = np.sqrt(sgp.predict_std(X_new))
    print("Sparse GP Mean Prediction:", mean_pred_sparse)
    print("Sparse GP Std Prediction:", sigma_pred_sparse)
    sgp.plot(f)

    # Print kernel values (inducing points for sparse GP)
    print("Kernel values (inducing points):")
    kernel_matrix = RBF_Kernel(Z).make_kernel()
    for row in kernel_matrix:
        print(", ".join("{:.3f}".format(val) for val in row))

