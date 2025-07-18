
import numpyro
numpyro.set_host_device_count(100)

import jax.numpy as jnp
import jax.random as random

from config import load_initial_data, NOISE_STD, KEY 
from config import f_rosenbrock, f_hosaki, f_bird
from model import compute_posterior
from inference import (
    run_mcmc,
    expected_improvement_acquisition_function,
    compute_posterior_from_mcmc, 
    average_acquisition_over_posterior_samples,
)
from plotting import plot_gp_results

# ===============================
# MAIN MCMC-BASED BAYESIAN OPTIMIZATION LOOP
# ===============================

function_name = "f_rosenbrock"
function_map = {
    "f_rosenbrock": f_rosenbrock,
    "f_hosaki": f_hosaki,
    "f_bird": f_bird
}

f_true = function_map[function_name]
X, y, X_new = load_initial_data(function_name)
MAX_ITER = 10
for iteration in range(MAX_ITER):
    print(f"\n--- Iteration {iteration + 1} ---")

    # Step 1: Run MCMC to sample hyperparameters
    key = random.PRNGKey(iteration)
    samples = run_mcmc(X, y, key)

    avg_alpha, x_next, idx_next = average_acquisition_over_posterior_samples(
        X_train=X,
        y_train=y,
        X_new=X_new,
        posterior_fn=compute_posterior,
        samples=samples,
    )

    # Step 4: To visualize, compute posterior mean and std from averaged posterior (optional)
    posterior_means, posterior_stds = compute_posterior_from_mcmc(
        X, y, X_new, compute_posterior, samples
    )

    # Step 5: Plot results
    plot_gp_results(
        X,
        y,
        X_new,
        posterior_means,
        posterior_stds,
        avg_alpha,
        idx_next,
        noise_std=NOISE_STD,
        true_fn=f_true,
        title=f"Iteration {iteration + 1} of Bayesian Optimization (MCMC)",
    )

    # Step 6: Evaluate function at selected next point with noise
    y_next = f_true(x_next.flatten()) + NOISE_STD * random.normal(KEY, shape=())

    # Step 7: Append new point to training data
    X = jnp.vstack([X, x_next.reshape(1, -1)])
    y = jnp.append(y, y_next)