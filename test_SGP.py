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

    def make_kernel(self, X1, X2, l=1, sigma=0):
        K = jnp.exp(-jnp.linalg.norm(X1[:, None] - X2[None, :], axis=-1)**2 / (2 * l**2))
        if len(X1) == len(X2):  # Check if X1 and X2 are the same
            K += sigma * jnp.eye(len(X1))  # Add noise only if they are the same
        return K

class SparseGaussianProcess:
    def __init__(self, X, y, inducing_points, optimizer="L-BFGS-B", l=1.0, sigma_n=1e-8, sigma_f=1.0):
        self.X = X
        self.y = y
        self.inducing_points = inducing_points
        self.optimizer = optimizer
        self.l = l  # Initial length scale
        self.sigma_n = sigma_n  # Initial noise variance
        self.sigma_f = sigma_f  # Initial signal variance

    def kernel(self, X1, X2):
        rbf_kernel = RBF_Kernel(X1)
        return rbf_kernel.make_kernel(X1, X2, l=self.l, sigma=self.sigma_n)

    def negative_log_likelihood(self, params):
        # Unpack parameters
        self.l, self.sigma_n, self.sigma_f = params

        # Kernel for inducing points (Kuu) and training points (Kff)
        Kuu = self.kernel(self.inducing_points, self.inducing_points) + jnp.eye(len(self.inducing_points)) * self.sigma_n
        Kuf = self.kernel(self.inducing_points, self.X)  # Cross kernel
        Kff = self.kernel(self.X, self.X) + jnp.eye(len(self.X)) * self.sigma_n  # Noise in the training points kernel

        # Solve for alpha using the correct kernel matrices
        L = jnp.linalg.cholesky(Kuu)  # Cholesky decomposition of Kuu
        alpha = jax.scipy.linalg.solve_triangular(L, jax.scipy.linalg.solve_triangular(L.T, self.y, lower=False), lower=True)

        # Compute the negative log likelihood
        log_likelihood = -0.5 * jnp.dot(self.y.T, alpha) - jnp.sum(jnp.log(jnp.diag(L))) - (len(self.y) / 2) * jnp.log(2 * jnp.pi)

        return -log_likelihood

    def fit(self):
        if self.optimizer in ["L-BFGS-B", "CG"]:
            result = minimize(self.negative_log_likelihood, x0=[self.l, self.sigma_f, self.sigma_n], method=self.optimizer)
            self.l, self.sigma_f, self.sigma_n = result.x

        elif self.optimizer == "ADAM":
            self.adam_optimizer = ADAM(learning_rate=0.01)
            result = self.optimise()
            self.l, self.sigma_f, self.sigma_n = result

        print(f"Training complete. Optimal length scale: {self.l:.2f}, Signal variance: {self.sigma_f:.2f}, Noise variance: {self.sigma_n:.2f}")

    def predict(self, X_new):
        Kuu = self.kernel(self.inducing_points, self.inducing_points) + self.sigma_n**2 * jnp.eye(len(self.inducing_points))
        Kuf = self.kernel(self.inducing_points, X_new)
        Kuu_inv = jnp.linalg.inv(Kuu)

        mean = jnp.dot(Kuf.T, jnp.dot(Kuu_inv, self.y))
        var = self.kernel(X_new, X_new) - jnp.dot(Kuf.T, jnp.dot(Kuu_inv, Kuf))

        return mean, var

    def optimise(self, max_iter=10000, Plot_Hist=False):
        params = jnp.array([self.l, self.sigma_f, self.sigma_n])
        value_and_grad = jax.jit(jax.value_and_grad(self.negative_log_likelihood))
        loss_history, time_history_ADAM = [], []
        start = time.time()
        time_budget = 10.

        for i in range(max_iter):
            val, gradients = value_and_grad(params)
            params = self.adam_optimizer.update(gradients, params)
            loss_history.append(val)
            time_history_ADAM.append(time.time() - start)
            if time.time() - start > time_budget:
                break

        if Plot_Hist:
            plt.figure(figsize=(10, 6))
            plt.plot(time_history_ADAM, loss_history, lw=2)
            plt.yscale("log")
            plt.xlabel("Time (s)")
            plt.ylabel("Loss")
            plt.grid()
            plt.title("Adam optimizer")
            plt.show()
        return params

    def plot(self, f=None, x_range=(-5, 5), n_points=1000, xlabel="$x$", ylabel="$f(x)$", title="Sparse GP Regression", legend_labels=None):
        x = np.linspace(x_range[0], x_range[1], n_points)
        y_pred, y_std = [], []

        for i in range(len(x)):
            X_new = np.array([x[i]]).reshape(-1, 1)
            mean_pred, sigma_pred = self.predict(X_new)
            y_pred.append(mean_pred)
            y_std.append(sigma_pred)

        y_pred = np.array(y_pred)
        y_std = np.array(y_std)

        plt.figure(figsize=(10, 6))
        if f:
            plt.plot(x, f(x), "r", label=legend_labels[0] if legend_labels else "True Function")
        plt.plot(self.X, self.y, "ro", label=legend_labels[1] if legend_labels else "Observations")
        plt.plot(x, y_pred, "b-", label=legend_labels[2] if legend_labels else "GP Mean")
        plt.fill_between(x, y_pred.ravel() - 1.96 * y_std.ravel(), y_pred.ravel() + 1.96 * y_std.ravel(),
                         color="lightblue", alpha=0.5, label=legend_labels[3] if legend_labels else "95% Confidence Interval")

        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.show()

# Sample usage
if __name__ == "__main__":
    np.random.seed(42)
    n_points = 20
    X_train = np.random.uniform(-5, 5, n_points).reshape(-1, 1)
    y_train = np.sin(X_train) + 0.2 * np.random.randn(n_points, 1)

    # Define inducing points (choose a subset of training points or randomly)
    inducing_points = X_train[np.random.choice(X_train.shape[0], size=5, replace=False)].reshape(-1, 1)  # 5 inducing points

    gp = SparseGaussianProcess(X_train, y_train, inducing_points, optimizer="ADAM")
    gp.fit()

    # Predict and plot
    gp.plot(f=np.sin, x_range=(-5, 5), n_points=1000,
            xlabel="X", ylabel="y",
            title="Sparse GP Regression on Sinusoidal Data",
            legend_labels=["True Function (sin)", "Training Data", "GP Mean", "95% Confidence Interval"])
