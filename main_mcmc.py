
import numpyro
numpyro.set_host_device_count(100)

import jax.numpy as jnp
import jax.random as random

from config import NOISE_STD, KEY, load_initial_data 
from model import compute_posterior, mu_0_factory
from inference import (
    run_mcmc,
    compute_posterior_from_mcmc, 
    average_acquisition_over_posterior_samples,
)
from plotting import plot_gp_results

# ===============================
# MAIN MCMC-BASED BAYESIAN OPTIMIZATION LOOP
# ===============================

function_name = "f_svm"  # or any other: "f_rosenbrock", ...
f_true, X, y, X_new = load_initial_data(function_name)

MAX_ITER = 20
patience = 5
improvement_threshold  = 1e-4
no_improvement_counter = 0
best_y = jnp.max(y)
best_x = jnp.argmax(y)
plot   = False

for iteration in range(MAX_ITER):
    print(f"\n--- Iteration {iteration + 1} ---")

    # Step 1: Run MCMC to sample hyperparameters
    key = random.PRNGKey(iteration)
    samples = run_mcmc(X, y, key)
    mu_0_fn = mu_0_factory(y)
    avg_alpha, x_next, idx_next = average_acquisition_over_posterior_samples(
        X_train=X,
        y_train=y,
        X_new=X_new,
        posterior_fn=compute_posterior,
        mu_0=mu_0_fn,
        samples=samples,
    )

    # Step 4: To visualize, compute posterior mean and std from averaged posterior (optional)
    posterior_means, posterior_stds = compute_posterior_from_mcmc(
        X, y, X_new, compute_posterior, samples, mu_0_fn
    )

    # Step 5: Plot results
    if(plot):
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

    y_next = f_true(x_next.flatten()) + NOISE_STD * random.normal(key, shape=())

    # Early stopping check
    if y_next > best_y + improvement_threshold:
        best_x = jnp.argmax(y)
        best_y = y_next
        no_improvement_counter = 0
    else:
        no_improvement_counter += 1
        if no_improvement_counter >= patience:
            print(f"Early stopping triggered at iteration {iteration+1}.")
            break

    # Step 7: Append new point to training data
    X = jnp.vstack([X, x_next.reshape(1, -1)])
    y = jnp.append(y, y_next)

best_id = jnp.argmax(y)
best_x  = X[best_id]
best_y  = y[best_id]

print('Maximum Found: ', best_x, best_y)
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