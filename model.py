import jax.numpy as jnp
from jax import vmap, jit
from jax.scipy.linalg import solve
from config import EPSILON, d_vector, n_vector, n_by_d_matrix, n_by_n_matrix

@jit
def matern_kernel(x: d_vector, x_p: d_vector, theta_0: float, theta: d_vector):
    """
    Computes the Matérn 5/2 kernel between two d-dimensional input vectors.
    """
    assert x.ndim == 1 and x.shape == x_p.shape == theta.shape, (
        "x, x_p, and theta must be 1D arrays of the same shape")
    delta = x - x_p
    scaled_delta = delta * theta
    r = jnp.sqrt(jnp.sum(scaled_delta**2) + EPSILON)

    sqrt5_r = jnp.sqrt(5.0) * r
    k = (theta_0**2 * (1 + sqrt5_r + (5.0 / 3.0) * r**2) * jnp.exp(-sqrt5_r))
    return k

@jit
def compute_kernel_matrix(X: n_by_d_matrix, theta_0: float,
                          theta: d_vector) -> n_by_n_matrix:
    """-
    Computes the kernel matrix K for a set of input vectors using the Matérn 5/2 kernel.
    """

    def kernel_row(x):
        return vmap(lambda x_p: matern_kernel(x, x_p, theta_0, theta))(X)

    return vmap(kernel_row)(X)

@jit
def compute_covariance_vector(X: n_by_d_matrix, x: d_vector, theta_0: float,
                              theta: d_vector) -> n_vector:
    """Computes covariance vector k(x, X) between new point x and dataset X."""
    return vmap(lambda x_p: matern_kernel(x, x_p, theta_0, theta))(X)

def compute_posterior(
    x: d_vector,
    X: n_by_d_matrix,
    y: n_vector,
    theta_0: float,
    theta: d_vector,
    mu_0: float,
    sigma_squared: float
):
    """
    Computes the posterior mean and variance of the GP at a new point x.
    """
    k = compute_covariance_vector(X, x, theta_0, theta)
    K = compute_kernel_matrix(X, theta_0, theta)
    K += jnp.eye(X.shape[0]) * sigma_squared
    m = mu_0*jnp.ones(X.shape[0])
    mu_n = mu_0 + k.T @ solve(K, y - m)
    sigma_squared_n = matern_kernel(x, x, theta_0, theta) - k.T @ solve(K, k)

    return mu_n, sigma_squared_n