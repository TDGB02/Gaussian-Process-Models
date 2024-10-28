
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

class RBF_Kernel:
    def __init__(self, X):
        self.X = X  
        self.kernel = np.zeros((X.shape[0], X.shape[0]))


    def make_kernel(self, l=1, sigma=0):
        for i in range(self.kernel.shape[0]):
            for j in range(self.kernel.shape[1]):
                self.kernel[i, j] = self.Kernel(self.X[i], self.X[j])(l)

        return self.kernel + sigma * np.eye(len(self.kernel))

    class Kernel:
        def __init__(self, x_n, x_m):
            self.x_n = x_n
            self.x_m = x_m

        def __call__(self, l=1):
            ## RBF Kernel
            K = np.exp(-np.linalg.norm(self.x_n - self.x_m)**2 / (2 * l**2))
            return K

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
        return rbf_kernel[-1,-1] - np.dot(np.dot(k,np.linalg.inv(K)),k)
    
    def plot(self, f):
        # Range of x to obtain the confidence intervals.
        x = np.linspace(0, 10, 1000)
        # Obtain the corresponding mean and standard deviations.
        y_pred = []
        y_std = []
        for i in range(len(x)):
            X_new = np.array([x[i]])
            y_pred.append(self.predict_mean(self.X, self.y, X_new))
            y_std.append(np.sqrt(self.predict_std(self.X, X_new)))
            
        y_pred = np.array(y_pred)
        y_std = np.array(y_std)
        plt.figure(figsize = (15, 5))
        plt.plot(x, f(x), "r")
        plt.plot(self.X, self.y, "ro")
        plt.plot(x, y_pred, "b-")
        plt.fill(np.hstack([x, x[::-1]]),
                np.hstack([y_pred - 1.9600 * y_std, 
                        (y_pred + 1.9600 * y_std)[::-1]]),
                alpha = 0.5, fc = "b")
        plt.xlabel("$x$", fontsize = 14)
        plt.ylabel("$f(x)$", fontsize = 14)
        plt.legend(["$y = x^2$", "Observations", "Predictions", "95% Confidence Interval"], fontsize = 14)
        plt.grid(True)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.show()


import numpy as np
import matplotlib.pyplot as plt

class SparseGaussianProcess:
    def __init__(self, X, y, Z):
        self.X = X
        self.y = y
        self.Z = Z  # Inducing points

    def predict_mean(self, X, y, X_new):
        # Covariance matrices
        K_zz = RBF_Kernel(self.Z).make_kernel()
        K_xz = RBF_Kernel(X, self.Z).make_kernel()
        K_zx = K_xz.T
        K_xx = RBF_Kernel(X).make_kernel()

        # Compute alpha (inverted term for mean prediction)
        alpha = np.dot(np.linalg.inv(K_zz + K_zx @ np.linalg.inv(K_xx) @ K_xz), K_zx @ np.linalg.inv(K_xx) @ y)

        # Covariance between new point and inducing points
        k_new_z = RBF_Kernel(X_new, self.Z).make_kernel()
        
        # Mean prediction
        return np.dot(k_new_z, alpha)
    
    def predict_std(self, X_new):
        K_zz = RBF_Kernel(self.Z).make_kernel()
        K_xz = RBF_Kernel(self.X, self.Z).make_kernel()
        K_zx = K_xz.T
        K_xx = RBF_Kernel(self.X).make_kernel()
        
        k_new_z = RBF_Kernel(X_new, self.Z).make_kernel()
        k_new_new = RBF_Kernel(X_new).make_kernel()

        # Variance prediction (posterior variance)
        var_pred = k_new_new - np.dot(k_new_z, np.dot(np.linalg.inv(K_zz + np.dot(K_zx, np.linalg.inv(K_xx) @ K_xz)), k_new_z.T))
        return var_pred

    def plot(self, f):
        # Range of x to obtain the confidence intervals.
        x = np.linspace(0, 10, 1000)
        # Obtain the corresponding mean and standard deviations.
        y_pred = []
        y_std = []
        for i in range(len(x)):
            X_new = np.array([x[i]])
            y_pred.append(self.predict_mean(self.X, self.y, X_new))
            y_std.append(np.sqrt(self.predict_std(X_new)))
            
        y_pred = np.array(y_pred)
        y_std = np.array(y_std)
        plt.figure(figsize=(15, 5))
        plt.plot(x, f(x), "r")
        plt.plot(self.X, self.y, "ro")
        plt.plot(x, y_pred, "b-")
        plt.fill(np.hstack([x, x[::-1]]),
                 np.hstack([y_pred - 1.9600 * y_std, 
                            (y_pred + 1.9600 * y_std)[::-1]]),
                 alpha=0.5, fc="b")
        plt.xlabel("$x$", fontsize=14)
        plt.ylabel("$f(x)$", fontsize=14)
        plt.legend(["$y = x^2$", "Observations", "Predictions", "95% Confidence Interval"], fontsize=14)
        plt.grid(True)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()


if __name__ == "__main__":
    def f(x):
        return (x-5) ** 2
    # Training data x and y:
    X = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    y = f(X)
    X = X.reshape(-1, 1)
    # New input to predict:
    X_new = np.array([5.5])
    # Calculate and print the new predicted value of y:
    gp = GaussianProcess(X, y)
    mean_pred = gp.predict_mean(X, y, X_new)
    sigma_pred = np.sqrt(gp.predict_std(X, X_new))
    print("mean predict :{}".format(mean_pred))
    # Calculate and print the corresponding standard deviation:
    print("std predict :{}".format(sigma_pred))
    gp.plot(f)


    for i in RBF_Kernel(X).make_kernel():
        for j in i:
            print("{:.3f}".format(j), end = ", ")
        print()