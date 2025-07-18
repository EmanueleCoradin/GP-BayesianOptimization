import jax.numpy as jnp
import jax.random as random

from config import load_initial_data, NOISE_STD, TOL, KEY, f_hosaki
from model import compute_posterior
from inference import compute_posterior_and_ei, marginal_likelihood_log
from plotting import plot_gp_results
from jax.scipy.optimize import minimize

# Load data
X, y, X_new = load_initial_data()

MAX_ITER = 10
X_train = X
y_train = y
params_init_log = jnp.log(jnp.array([1.0, 1.0, 1.0, 1.0]))

for iteration in range(MAX_ITER):
    print(f"\n=== Iteration {iteration + 1} ===")

    # Step 1: Optimize GP hyperparameters on current data
    
    result = minimize(marginal_likelihood_log,
                      params_init_log,
                      args=(X_train, y_train),
                      method="BFGS",
                      tol=TOL)
    params_opt = jnp.exp(result.x)
    mu_0  = params_opt[0]
    theta = params_opt[1:-1]
    sigma_squared = params_opt[-1]

    params_init_log = result.x
    print(f"Optimization success: {result.success}")
    print(f"Params: {params_opt}")
    print(f"Neg log-likelihood: {result.fun}")

    # Step 2: Compute GP posterior and EI acquisition
    posterior_means, posterior_stds, alpha_EI, x_next, idx_next = compute_posterior_and_ei(
        X_train,
        y_train,
        X_new,
        compute_posterior,
        theta,
        mu_0,
        sigma_squared
    )

    print(f"Next sampling point: {x_next}")

    # Step 3: Plot results
    plot_gp_results(
        X_train,
        y_train,
        X_new,
        posterior_means,
        posterior_stds,
        alpha_EI,
        idx_next,
        noise_std=NOISE_STD,
        true_fn=f_hosaki,
    )

    # Step 4: Evaluate function at x_next (with noise)
    y_next = f_hosaki(x_next.flatten()) + NOISE_STD * random.normal(KEY, shape=())

    # Step 5: Update training data
    X_train = jnp.vstack([X_train, x_next.reshape(1, -1)])
    y_train = jnp.append(y_train, y_next)
