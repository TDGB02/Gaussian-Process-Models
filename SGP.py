import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class RBF_Kernel:
    def __init__(self, X1, X2=None):
        self.X1 = X1
        self.X2 = X2 if X2 is not None else X1

    def make_kernel(self, l=1, sigma=1e-8):
        n1, n2 = self.X1.shape[0], self.X2.shape[0]
        kernel = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                kernel[i, j] = np.exp(-np.linalg.norm(self.X1[i] - self.X2[j])**2 / (2 * l**2))
        return kernel + sigma * np.eye(n1) if n1 == n2 else kernel

class GaussianProcess:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def predict_mean(self, X, y, X_new):
        rbf_kernel = RBF_Kernel(np.vstack([X, X_new])).make_kernel()
        K = rbf_kernel[:len(X), :len(X)]
        k = rbf_kernel[:len(X), -1]
        return np.dot(np.dot(k, np.linalg.inv(K)), y)        
    
    def predict_std(self, X, X_new):
        rbf_kernel = RBF_Kernel(np.vstack([X, X_new])).make_kernel()
        K = rbf_kernel[:len(X), :len(X)]
        k = rbf_kernel[:len(X), -1]
        return rbf_kernel[-1, -1] - np.dot(np.dot(k, np.linalg.inv(K)), k)
    
    def plot(self, f):
        x = np.linspace(0, 10, 1000).reshape(-1, 1)
        y_pred, y_std = [], []
        for X_new in x:
            y_pred.append(self.predict_mean(self.X, self.y, X_new.reshape(1, -1)))
            y_std.append(np.sqrt(self.predict_std(self.X, X_new.reshape(1, -1))))
        
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
        plt.legend(["$y = x^2$", "Observations", "Predictions", "95% Confidence Interval"], fontsize=14)
        plt.grid(True)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

class SparseGaussianProcess:
    def __init__(self, X, y, Z):
        self.X = X
        self.y = y
        self.Z = Z

    def predict_mean(self, X, y, X_new):
        K_zz = RBF_Kernel(self.Z).make_kernel()
        K_xz = RBF_Kernel(X, self.Z).make_kernel()
        K_zx = K_xz.T
        K_xx = RBF_Kernel(X).make_kernel()
        
        alpha = np.linalg.solve(K_zz + K_zx @ np.linalg.solve(K_xx, K_xz), K_zx @ np.linalg.solve(K_xx, y))
        k_new_z = RBF_Kernel(X_new, self.Z).make_kernel()
        return np.dot(k_new_z, alpha)
    
    def predict_std(self, X_new):
        K_zz = RBF_Kernel(self.Z).make_kernel()
        K_xz = RBF_Kernel(self.X, self.Z).make_kernel()
        K_zx = K_xz.T
        K_xx = RBF_Kernel(self.X).make_kernel()
        
        k_new_z = RBF_Kernel(X_new, self.Z).make_kernel()
        k_new_new = RBF_Kernel(X_new).make_kernel()
        
        K_inv = np.linalg.inv(K_zz + np.dot(K_zx, np.linalg.solve(K_xx, K_xz)))
        return k_new_new - np.dot(k_new_z, np.dot(K_inv, k_new_z.T))

    def plot(self, f):
        x = np.linspace(0, 10, 1000).reshape(-1, 1)
        y_pred, y_std = [], []
        for X_new in x:
            y_pred.append(self.predict_mean(self.X, self.y, X_new.reshape(1, -1)))
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
        plt.legend(["$y = x^2$", "Observations", "Predictions", "95% Confidence Interval"], fontsize=14)
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
    
    # Full Gaussian Process
    gp = GaussianProcess(X, y)
    mean_pred = gp.predict_mean(X, y, X_new)
    sigma_pred = np.sqrt(gp.predict_std(X, X_new))
    print("Full GP Mean Prediction:", mean_pred)
    print("Full GP Std Prediction:", sigma_pred)
    gp.plot(f)

    # Sparse Gaussian Process with inducing points
    n_inducing_points = 3
    kmeans = KMeans(n_clusters=n_inducing_points, random_state=0).fit(X)
    Z = kmeans.cluster_centers_
    sgp = SparseGaussianProcess(X, y, Z)
    mean_pred_sparse = sgp.predict_mean(X, y, X_new)
    sigma_pred_sparse = np.sqrt(sgp.predict_std(X_new))
    print("Sparse GP Mean Prediction:", mean_pred_sparse)
    print("Sparse GP Std Prediction:", sigma_pred_sparse)
    sgp.plot(f)
    
    # Print kernel values (inducing points for sparse GP)
    print("Kernel values (inducing points):")
    kernel_matrix = RBF_Kernel(Z).make_kernel()
    for row in kernel_matrix:
        print(", ".join("{:.3f}".format(val) for val in row))
