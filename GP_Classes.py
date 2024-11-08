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
        self.lr_max = 1e-2
        self.lr_min = 1e-6
        self.prev_f_val = float('inf') 
    
    def update(self, gradients, params, f_val=None, tol_f=1e-6, tol_g=1e-6, tol_x=1e-6):
        
        if self.m is None:
            self.m = jnp.zeros_like(params)
        if self.v is None:
            self.v = jnp.zeros_like(params)
        
        # Update timestep and apply decay to learning rate
        self.t += 1
        # Use exponential decay for learning rate, with max and min values
        self.lr = self.lr_max - (self.lr_max - self.lr_min) * jnp.exp(-self.t)
         
        # Update biased first and second moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        
        # Compute bias-corrected first and second moments
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Parameter update step
        step = self.lr * m_hat / (jnp.sqrt(v_hat) + self.epsilon)
        params -= step
        
        # Check termination criteria if f_val is provided
        if f_val is not None:
            f_change = abs(f_val - self.prev_f_val)
            grad_norm = jnp.linalg.norm(gradients)
            step_norm = jnp.linalg.norm(step)

            # Update previous function value for next comparison
            self.prev_f_val = f_val

            # Check if any termination criterion is met
            if f_change < tol_f:
                return params, True
            if grad_norm < tol_g:
                return params, True
            if step_norm < tol_x:
                return params, True
        
        return params, False 


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
        L = jnp.linalg.cholesky(K + 1e-9 * jnp.eye(len(self.X)))  # Use JAX's Cholesky

        alpha = jax.scipy.linalg.solve_triangular(L.T, jax.scipy.linalg.solve_triangular(L, self.y, lower=True), lower=False)

        return jnp.squeeze(0.5 * jnp.dot(self.y.T, alpha) + jnp.sum(jnp.log(jnp.diagonal(L))) + 0.5 * len(self.X) * jnp.log(2 * jnp.pi))

    def fit(self):
        self.l = np.random.uniform(0.1, 2.0)         # Random length scale 
        self.sigma_f = np.random.uniform(0.1, 2.0)   # Random signal variance within
        self.sigma_n = np.random.uniform(1e-9, 1e-6) # Random noise variance within
        if self.optimizer == "L-BFGS-B":
            result = minimize(self.negative_log_likelihood, x0=[self.l, self.sigma_f, self.sigma_n], method=self.optimizer)
            print("NLL after optimization L-BFGS-B:", self.negative_log_likelihood(result.x))
            self.l, self.sigma_f, self.sigma_n = result.x

        elif self.optimizer == "CG":
            result = minimize(self.negative_log_likelihood, x0=[self.l, self.sigma_f, self.sigma_n], method=self.optimizer)
            print("NLL after optimization CG:", self.negative_log_likelihood(result.x))
            self.l, self.sigma_f, self.sigma_n = result.x
            
        elif self.optimizer == "ADAM":
            self.adam_optimizer = ADAM(learning_rate=0.001)
            result = self.optimise()
            print("NLL after optimization ADAM:", self.negative_log_likelihood(result))
            self.l, self.sigma_f, self.sigma_n = result

        self.K = self.compute_kernel(self.X, self.l, self.sigma_n) + self.sigma_f**2 * jnp.eye(len(self.X))
        return self.sigma_f


    def predict(self, X_new):
        K_new = self.compute_kernel(np.vstack([self.X, X_new]), self.l, self.sigma_n)
        K = K_new[:len(self.X), :len(self.X)] 
        k = K_new[:len(self.X), -1]  
        
        # Cholesky decomposition of the kernel matrix K
        L = np.linalg.cholesky(K + 1e-3 * np.eye(len(self.X)))  # Add jitter for stabilityn
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
        v = np.linalg.solve(L, k)
        
        # Compute the mean prediction for X_new
        mean = np.dot(k.T, alpha)    
        
        # Compute the variance of the prediction for X_new
        var = K_new[-1, -1] - np.dot(v.T, v)
        
        return mean, var
    
    def optimise(self, max_iter=10000, Plot_Hist=False):
        # Set initial parameters as a JAX array
        params = jnp.array([self.l, self.sigma_f, self.sigma_n])        
        value_and_grad = jax.jit(jax.value_and_grad(self.negative_log_likelihood))

        start = time.time()
        # Set up time budget and tolerance values
        time_budget = 10.0
        tol_f = 1e-1
        tol_g = 1e-1
        tol_x = 1e-1

        for _ in range(max_iter):
            # Compute loss and gradients
            val, gradients = value_and_grad(params)
            
            # Update parameters using ADAM with termination checks
            params, terminate = self.adam_optimizer.update(gradients, params, f_val=val, tol_f=tol_f, tol_g=tol_g, tol_x=tol_x)
            # Check termination flag and time budget
            if terminate:
                #Must use clip otherwise negative parameters will cause -> not a positive definite matrix
                params = jnp.clip(params, 1e-6, None)
                break
            if time.time() - start > time_budget:
                break

        # Update final parameters with optimized values
        self.l, self.sigma_f, self.sigma_n = params
        return params

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
        self.l = np.random.uniform(0.1, 2.0)         # Random length scale
        self.sigma_f = np.random.uniform(0.1, 2.0)   # Random signal variance
        self.sigma_n = np.random.uniform(1e-9, 1e-6) # Random noise variance

        if self.optimizer == "L-BFGS-B":
            result = minimize(self.negative_log_likelihood, x0=[self.l, self.sigma_f, self.sigma_n], method=self.optimizer)
            params = result.x
            params = jnp.clip(params, 1e-6, None)
            self.l, self.sigma_f, self.sigma_n = params

        elif self.optimizer == "CG":
            result = minimize(self.negative_log_likelihood, x0=[self.l, self.sigma_f, self.sigma_n], method=self.optimizer)
            params = result.x
            params = jnp.clip(params, 1e-6, None)
            self.l, self.sigma_f, self.sigma_n = params
            
        elif self.optimizer == "ADAM":
            self.adam_optimizer = ADAM(learning_rate=0.001)
            result = self.optimise()
            self.l, self.sigma_f, self.sigma_n = result
            
        self.K = self.compute_kernel(self.Z, self.l, self.sigma_n) + self.sigma_f**2 * jnp.eye(len(self.Z))
        return self.sigma_f

    def predict(self, X_new):
        K_new = self.compute_kernel(np.vstack([self.Z, X_new]), self.l, self.sigma_n)
        K = K_new[:len(self.Z), :len(self.Z)] 
        k = K_new[:len(self.Z), -1]  
        # Cholesky decomposition of the kernel matrix K
        L = np.linalg.cholesky(K + 1e-6 * np.eye(len(self.Z)))  # Add jitter for stability
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y[:len(self.Z)]))
        v = np.linalg.solve(L, k)
        
        # Compute the mean prediction for X_new
        mean = np.dot(k.T, alpha)    
        
        # Compute the variance of the prediction for X_new
        var = K_new[-1, -1] - np.dot(v.T, v)
        
        return mean, var
    
    def optimise(self, max_iter=10000, Plot_Hist=False):

        params = jnp.array([self.l, self.sigma_f, self.sigma_n])
        value_and_grad = jax.value_and_grad(self.negative_log_likelihood)

        start = time.time()

        time_budget = 10.0
        tol_f = 1e-6
        tol_g = 1e-6
        tol_x = 1e-6

        for _ in range(max_iter):
            # Compute loss and gradients
            val, gradients = value_and_grad(params)
            
            # Update parameters using ADAM with termination checks
            params, terminate = self.adam_optimizer.update(gradients, params, f_val=val, tol_f=tol_f, tol_g=tol_g, tol_x=tol_x)

            # Check termination flag and time budget
            if terminate:
                break
            if time.time() - start > time_budget:
                break

        # Update final parameters with optimized values
        self.l, self.sigma_f, self.sigma_n = params
        return params