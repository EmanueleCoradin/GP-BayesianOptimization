import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias
from typing import Callable

# Type aliases
d_Vector: TypeAlias = NDArray[np.float64]
n_Vector: TypeAlias = NDArray[np.float64]
n_by_d_Matrix: TypeAlias = NDArray[np.float64]
n_by_n_Matrix: TypeAlias = NDArray[np.float64]
d_VectorToReal: TypeAlias = Callable[[d_Vector], float]

# ===============================
#      FUNCTION DEFINITIONS
# ===============================

# -------------------------------
#        Bayesian Optimization
# -------------------------------


def BayesianOptimization(X, y, alpha):
    # TODO: Implement Bayesian Optimization loop
    #select x_new from optimizing alpha
    #x_new = argmax(alpha)
    #y_new = f(x_new)
    #update X and y
    #X = np.append(X, x_new)
    #y = np.append(y, y_new)
    #Update statistical model

    return (X, y, alpha)


def MaternKernel(x: d_Vector, x_p: d_Vector, theta_0: float,
                 theta: d_Vector) -> float:
    """
    Computes the Matérn 5/2 kernel between two d-dimensional input vectors.
    """
    assert x.ndim == 1 and x.shape == x_p.shape == theta.shape, \
        "x, x_p, and theta must be 1D arrays of the same shape"
    Lambda = np.diag(np.power(theta, 2))
    delta = x - x_p
    r = np.sqrt(delta.T @ Lambda @ delta)
    k = theta_0**2 * (1 + np.sqrt(5) * r +
                      (5 * r**2) / 3) * np.exp(-np.sqrt(5) * r)
    return float(k)


def MatrixKernel(X: n_by_d_Matrix, theta_0: float,
                 theta: d_Vector) -> n_by_n_Matrix:
    """
    Computes the kernel matrix K for a set of input vectors using the Matérn 5/2 kernel.
    """
    n = X.shape[0]
    K = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            val = MaternKernel(X[i], X[j], theta_0, theta)
            K[i, j] = val
            K[j, i] = val
    return K


def CovarianceVector(X: n_by_d_Matrix, x: d_Vector, theta_0: float,
                     theta: d_Vector) -> n_Vector:
    k = np.array([MaternKernel(x, x_p, theta_0, theta) for x_p in X])
    return k


def Posterior_Functions(x: d_Vector, X: n_by_d_Matrix, y: n_Vector,
                        theta_0: float, theta: d_Vector, mu_0: d_VectorToReal,
                        sigma_squared: float) -> tuple[float, float]:
    """
    Computes the posterior mean and variance of the GP at a new point x.
    """
    k = CovarianceVector(X, x, theta_0, theta)
    K = MatrixKernel(X, theta_0, theta)
    K += np.eye(K.shape[0]) * sigma_squared
    K_inv = np.linalg.inv(K)

    m = np.array([mu_0(xi) for xi in X])
    mu_n = mu_0(x) + k.T @ K_inv @ (y - m)
    sigma_squared_n = MaternKernel(x, x, theta_0, theta) - k.T @ K_inv @ k

    return float(mu_n), float(sigma_squared_n)


# -------------------------------
#       PLOTTING FUNCTIONS
# -------------------------------

# ===============================
#    CONSTANTS AND SAMPLE DATA
# ===============================

# Sample data (n=5, d=1)
X = np.array([[1], [2], [3], [4], [5]], dtype=np.float64)
y = np.power(X, 2).flatten()  # Fix: call flatten()
theta = np.array([0.1], dtype=np.float64)  # length scale vector (d=1)
theta_0 = 1.0
alpha = np.zeros(X.shape[0])
beta = np.zeros(X.shape[0])
sigma_squared = 1.0  # observation noise variance


def mu_0(x: d_Vector) -> float:
    prior = 1.0
    return prior  # prior mean


# ===============================
#           MAIN CODE
# ===============================

print(
    Posterior_Functions(np.array([6], dtype=np.float64), X, y, theta_0, theta,
                        mu_0, sigma_squared))
