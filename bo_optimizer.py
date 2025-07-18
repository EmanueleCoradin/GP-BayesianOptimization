import jax.numpy as jnp
import jax.random as random
from jax.scipy.optimize import minimize

from model import compute_posterior, mu_0
from inference import compute_posterior_and_ei, marginal_likelihood_log
from plotting import plot_gp_results

def run_bo_optimization_MLE(
    initial_X,
    initial_y,
    X_candidates,
    f_evaluate,
    max_iter=10,
    tol=1e-4,
    noise_std=0.0,
    seed=42,
    true_fn = None
):
    key = random.PRNGKey(seed)
    X_train = initial_X
    y_train = initial_y
    params_init_log = jnp.array([0., 0., -2.])

    for iteration in range(max_iter):
        print(f"\n=== Iteration {iteration + 1} ===")

        # Optimize GP hyperparameters on current data
        result = minimize(
            marginal_likelihood_log,
            params_init_log,
            args=(X_train, y_train),
            method="BFGS",
            tol=tol,
            options={"maxiter": 300}
        )
        params_opt = jnp.exp(result.x)
        params_init_log = result.x
        print(f"Optimization success: {result.success}")
        print(f"Params: {params_opt}")
        print(f"Neg log-likelihood: {result.fun}")

        theta = params_opt[:-1]
        sigma_squared = params_opt[-1]

        # Compute GP posterior and EI acquisition
        posterior_means, posterior_stds, alpha_EI, x_next, idx_next = compute_posterior_and_ei(
            X_train,
            y_train,
            X_candidates,
            compute_posterior,
            theta,
            mu_0,
            sigma_squared,
        )

        print(f"Next sampling point: {x_next}")

        # Optional plotting
        plot_gp_results(
            X_train,
            y_train,
            X_candidates,
            posterior_means,
            posterior_stds,
            alpha_EI,
            idx_next,
            noise_std=noise_std,
            true_fn=true_fn,
        )

        # Evaluate function at x_next (with noise)
        f_val = f_evaluate(jnp.array(x_next).flatten())
        noise = noise_std * random.normal(key, shape=())
        y_next = f_val + noise

        # Update training data
        X_train = jnp.vstack([X_train, x_next.reshape(1, -1)])
        y_train = jnp.append(y_train, y_next)

    return X_train, y_train

def run_bo_optimization(
    initial_X, initial_y, X_candidates,
    f_evaluate, posterior_fn, run_mcmc_fn,
    acquisition_fn, max_iter, tol, noise_std, base_key
):
    X, y = initial_X, initial_y
    for it in range(max_iter):
        key = random.fold_in(base_key, it)
        samples = run_mcmc_fn(X, y, key)

        avg_acq, idx_next = average_acquisition_over_posterior_samples(
            X, y, X_candidates, posterior_fn, samples, acquisition_fn
        )
        x_next = X_candidates[idx_next]
        y0 = f_evaluate(x_next)
        y_next = y0 + noise_std * random.normal(key, ())
        X = jnp.vstack([X, x_next])
        y = jnp.append(y, y_next)
        if tol and avg_acq.max() < tol:
            break
    return X, y
