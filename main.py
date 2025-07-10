import jax.numpy as jnp
from jax import grad, jit, vmap
from typing import Callable
from jax.scipy.linalg import solve
from jax.scipy.optimize import minimize
import matplotlib.pyplot as plt
import jax.random as random


# Type aliases
d_vector = jnp.ndarray
n_vector = jnp.ndarray
n_by_d_matrix = jnp.ndarray
n_by_n_matrix = jnp.ndarray
d_vector_to_real = Callable[[d_vector], float]


# ===============================
# CONSTANTS AND SAMPLE DATA
# ===============================

EPSILON = 1e-12  # small constant to regularize operations
TOL = 1e-5
KEY = random.PRNGKey(0)
NOISE_STD = 1e-1  # small noise standard deviation


def f_true(x):
    """True function to be approximated."""
    return jnp.sin(x)


X = jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = f_true(X.flatten())
noise = NOISE_STD * random.normal(KEY, shape=y.shape)
y += noise
X_new = jnp.linspace(-5, 10, 200).reshape(-1, 1)


# ===============================
# FUNCTION DEFINITIONS
# ===============================


def bayesian_optimization(X, y, alpha):
    """
    Placeholder for Bayesian Optimization loop.
    Select new points to sample based on acquisition function alpha.
    """
    # TODO: Implement Bayesian Optimization loop:
    # x_new = argmax(alpha)
    # y_new = f(x_new)
    # Update X and y with new sample
    # X = np.append(X, x_new)
    # y = np.append(y, y_new)
    # Update statistical model

    return X, y, alpha


def matern_kernel(x: d_vector, x_p: d_vector, theta_0: float, theta: d_vector):
    """
    Computes the Matérn 5/2 kernel between two d-dimensional input vectors.
    """
    assert x.ndim == 1 and x.shape == x_p.shape == theta.shape, (
        "x, x_p, and theta must be 1D arrays of the same shape"
    )
    delta = x - x_p
    scaled_delta = delta * theta
    r = jnp.sqrt(jnp.sum(scaled_delta ** 2) + EPSILON)

    sqrt5_r = jnp.sqrt(5.0) * r
    k = (
        theta_0 ** 2
        * (1 + sqrt5_r + (5.0 / 3.0) * r ** 2)
        * jnp.exp(-sqrt5_r)
    )
    return k


@jit
def compute_kernel_matrix(
    X: n_by_d_matrix, theta_0: float, theta: d_vector
) -> n_by_n_matrix:
    """
    Computes the kernel matrix K for a set of input vectors using the Matérn 5/2 kernel.
    """
    def kernel_row(x):
        return vmap(lambda x_p: matern_kernel(x, x_p, theta_0, theta))(X)

    return vmap(kernel_row)(X)


def compute_covariance_vector(
    X: n_by_d_matrix, x: d_vector, theta_0: float, theta: d_vector
) -> n_vector:
    """Computes covariance vector k(x, X) between new point x and dataset X."""
    return vmap(lambda x_p: matern_kernel(x, x_p, theta_0, theta))(X)


def mu_0(x: d_vector) -> float:
    """Prior mean function, here constant 1.0."""
    return 1.0


def compute_posterior(
    x: d_vector,
    X: n_by_d_matrix,
    y: n_vector,
    theta_0: float,
    theta: d_vector,
    mu_0: d_vector_to_real,
    sigma_squared: float,
):
    """
    Computes the posterior mean and variance of the GP at a new point x.
    """
    k = compute_covariance_vector(X, x, theta_0, theta)
    K = compute_kernel_matrix(X, theta_0, theta)
    K += jnp.eye(X.shape[0]) * sigma_squared
    m = vmap(mu_0)(X)
    mu_n = mu_0(x) + k.T @ solve(K, y - m)
    sigma_squared_n = matern_kernel(x, x, theta_0, theta) - k.T @ solve(K, k)

    return mu_n, sigma_squared_n


def marginal_likelihood(params: jnp.ndarray, X: n_by_d_matrix, y: n_vector):
    """
    Computes the negative log marginal likelihood of the GP model.
    """
    theta_0 = params[0]
    theta = params[1:-1]
    sigma_squared = params[-1]
    K = compute_kernel_matrix(X, theta_0, theta) + jnp.eye(X.shape[0]) * sigma_squared
    m = vmap(mu_0)(X)
    y_c = y - m

    term1 = -0.5 * y_c.T @ solve(K, y_c)
    term2 = -0.5 * jnp.linalg.slogdet(K)[1]
    term3 = -0.5 * X.shape[0] * jnp.log(2 * jnp.pi)

    return -(term1 + term2 + term3)  # negative log-likelihood


def marginal_likelihood_log(params_log, X, y):
    """
    Wrapper to optimize marginal likelihood in log parameter space.
    """
    params = jnp.exp(params_log)
    return marginal_likelihood(params, X, y)


# ===============================
# PLOTTING FUNCTIONS
# ===============================


def plot_gp_posterior(
    X_train: n_by_d_matrix,
    y_train: n_vector,
    X_new: jnp.ndarray,
    posterior_fn: Callable,
    theta_0: float,
    theta: d_vector,
    mu_0: d_vector_to_real,
    sigma_squared: float,
    noise_std: float,
    true_fn: Callable = None,
    xlabel="x",
    ylabel="f(x)",
    title="Gaussian Process reconstruction of unknown function",
):
    """
    Plots GP posterior mean and confidence interval along with training data,
    and optionally the true underlying function.
    """
    posterior_means, posterior_vars = vmap(
        lambda x: posterior_fn(x, X_train, y_train, theta_0, theta, mu_0, sigma_squared)
    )(X_new)

    posterior_means = posterior_means.flatten()
    posterior_stds = jnp.sqrt(posterior_vars.flatten())

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        X_train.flatten(),
        y_train,
        yerr=noise_std,
        fmt="o",
        color="red",
        label="Training data",
    )
    plt.plot(X_new.flatten(), posterior_means, "b-", label="GP mean prediction")
    plt.fill_between(
        X_new.flatten(),
        posterior_means - 2 * posterior_stds,
        posterior_means + 2 * posterior_stds,
        color="blue",
        alpha=0.2,
        label="95% confidence interval",
    )

    if true_fn is not None:
        y_true = true_fn(X_new.flatten())
        plt.plot(X_new.flatten(), y_true, "g--", label="True function")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.show()


# ===============================
# MAIN CODE
# ===============================

params_init = jnp.array([1.0, 1.0, 1.0])
params_init_log = jnp.log(params_init)

result = minimize(
    marginal_likelihood_log, params_init_log, args=(X, y), method="BFGS", tol=TOL
)

params_opt = jnp.exp(result.x)
theta_0 = params_opt[0]
theta = params_opt[1:-1]
sigma_squared = params_opt[-1]

print("Optimization success:", result.success)
print("Final params (original scale):", params_opt)
print("Final negative log-likelihood:", result.fun)

plot_gp_posterior(
    X,
    y,
    X_new,
    compute_posterior,
    theta_0,
    theta,
    mu_0,
    sigma_squared,
    NOISE_STD,
    true_fn=f_true,
)
