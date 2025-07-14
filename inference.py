import numpyro
from jax import vmap, random
import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax.scipy.optimize import minimize
from jax.scipy.stats.norm import cdf, pdf
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
from model import compute_kernel_matrix, mu_0
from config import EPSILON, TOL,  d_vector, n_vector, n_by_d_matrix, n_by_n_matrix, d_vector_to_real
from typing import Callable, Dict
from jax import jit


@jit
def marginal_likelihood(params: jnp.ndarray, X: n_by_d_matrix, y: n_vector):
    """
    Computes the negative log marginal likelihood of the GP model.
    """
    theta_0 = params[0]
    theta = params[1:-1]
    sigma_squared = params[-1]
    K = compute_kernel_matrix(X, theta_0,
                              theta) + jnp.eye(X.shape[0]) * sigma_squared
    m = vmap(mu_0)(X)
    y_c = y - m

    term1 = -0.5 * y_c.T @ solve(K, y_c)
    term2 = -0.5 * jnp.linalg.slogdet(K)[1]
    term3 = -0.5 * X.shape[0] * jnp.log(2 * jnp.pi)

    return -(term1 + term2 + term3)  # negative log-likelihood
@jit
def marginal_likelihood_log(params_log, X, y):
    """
    Wrapper to optimize marginal likelihood in log parameter space.
    """
    params = jnp.exp(params_log)
    return marginal_likelihood(params, X, y)

def exclude_existing_points(acquisition, X_new, X_train, threshold=1e-5):
    # Compute pairwise distances between X_new and X_train
    dists = jnp.linalg.norm(X_new[:, None, :] - X_train[None, :, :], axis=-1)  # shape (num_new, num_train)
    min_dists = jnp.min(dists, axis=1)  # min distance from each candidate to any train point

    mask = min_dists > threshold
    safe_acquisition = jnp.where(mask, acquisition, 0)
    return safe_acquisition
@jit
def expected_improvement_acquisition_function(mu_n, sigma_n, tau):
    z = (mu_n - tau) / (sigma_n + EPSILON)
    return (mu_n - tau) * cdf(z) + sigma_n * pdf(z)

def compute_posterior_and_ei(
    X_train: n_by_d_matrix,
    y_train: n_vector,
    X_new: jnp.ndarray,
    posterior_fn: Callable,
    theta_0: float,
    theta: d_vector,
    mu_0: d_vector_to_real,
    sigma_squared: float,
):
    posterior_means, posterior_vars = vmap(lambda x: posterior_fn(
        x, X_train, y_train, theta_0, theta, mu_0, sigma_squared))(X_new)

    posterior_means = posterior_means.flatten()
    posterior_stds = jnp.sqrt(posterior_vars.flatten())

    alpha_EI = expected_improvement_acquisition_function(
        posterior_means, posterior_stds, jnp.max(y_train))
    safe_alpha = exclude_existing_points(alpha_EI, X_new, X_train)
    idx_next = jnp.argmax(safe_alpha)
    x_next = X_new[idx_next]

    return posterior_means, posterior_stds, alpha_EI, x_next, idx_next

def optimize_hyperparameters(X, y):
    init_log_params = jnp.log(jnp.array([1.0, 1.0, 1.0]))
    result = minimize(marginal_likelihood_log,
                      init_log_params,
                      args=(X, y),
                      method="BFGS",
                      tol=TOL)
    return jnp.exp(result.x), result

def numpyro_model(X, y=None):
    d = X.shape[1]
    theta_0 = numpyro.sample("theta_0", dist.LogNormal(jnp.log(2.5), 0.3))
    theta = numpyro.sample("theta", dist.LogNormal(jnp.log(0.5), 0.3).expand([d]))
    sigma = numpyro.sample("sigma", dist.LogNormal(-5.0, 1.0))
    K = compute_kernel_matrix(X, theta_0, theta)  # <-- pass theta vector directly
    K += jnp.eye(X.shape[0]) * sigma**2
    mu = jnp.ones(X.shape[0])
    numpyro.sample("obs", dist.MultivariateNormal(mu, covariance_matrix=K), obs=y)


'''
def run_mcmc(X, y, key, num_warmup=500, num_samples=1000):
    nuts_kernel = NUTS(numpyro_model)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)
    mcmc.run(key, X=X, y=y)
    return mcmc.get_samples()
'''

def run_mcmc(X, y, key, num_chains=100, warmup=1000) -> Dict[str, jnp.ndarray]:
    """
    Run N MCMC chains in parallel using NumPyro and return the final samples.
    """

    mcmc = MCMC(
        NUTS(numpyro_model),
        num_warmup=warmup,
        num_samples=1,  # one sample per chain
        num_chains=num_chains,
        chain_method="parallel",  # truly parallel execution
        progress_bar=True,
    )

    mcmc.run(key, X=X, y=y)  # single call for all chains
    samples = mcmc.get_samples()

    return samples  # samples already shaped (num_chains, ...)


def compute_posterior_from_mcmc(
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_new: jnp.ndarray,
    posterior_fn: Callable,
    samples: dict,
    mu_0: Callable,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes GP posterior mean and std using MCMC samples.
    """
    def single_sample_prediction(theta_0, theta, sigma):
        theta_vec = jnp.atleast_1d(theta)  # ensures vector shape (d,) even if scalar
        return vmap(lambda x: posterior_fn(
            x, X_train, y_train, theta_0, theta_vec, mu_0, sigma**2
        ))(X_new)


    means, variances = vmap(single_sample_prediction)(
        samples["theta_0"], samples["theta"], samples["sigma"]
    )

    mean_gp = jnp.mean(means, axis=0)
    var_gp = jnp.mean(variances, axis=0) + jnp.var(means, axis=0)
    std_gp = jnp.sqrt(var_gp)

    return mean_gp, std_gp

def average_acquisition_over_posterior_samples(
    X_train, y_train, X_new, posterior_fn, mu_0, samples
):
    # samples["theta"] has shape (num_samples, d)
    all_alpha = vmap(lambda theta_0, theta, sigma: compute_posterior_and_ei(
        X_train, y_train, X_new, posterior_fn, theta_0, theta, mu_0, sigma**2)[2]
    )(samples["theta_0"], samples["theta"], samples["sigma"])  # shape (num_samples, num_candidates)

    avg_alpha = jnp.mean(all_alpha, axis=0)
    safe_alpha = exclude_existing_points(avg_alpha, X_new, X_train, threshold=1e-5)
    idx_next = jnp.argmax(safe_alpha)
    print(safe_alpha[::50])
    x_next = X_new[idx_next]

    return avg_alpha, x_next, idx_next

