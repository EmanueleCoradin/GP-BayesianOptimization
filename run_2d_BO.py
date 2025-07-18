import jax.numpy as jnp
import numpy as np
import jax.random as random

from config import load_initial_data, NOISE_STD, TOL, KEY, f_hosaki
from bo_optimizer import run_bo_optimization  # assuming this is saved in run_bo_optimization.py


def run_2d_bo(max_iter=10, noise_std=NOISE_STD, tol=TOL, seed=42):
    # Load initial data
    X_init, y_init, X_candidates = load_initial_data()

    # Define function to evaluate: f_hosaki without noise (noise added inside BO)
    def objective(x):
        return f_hosaki(x)

    # Run the BO loop with your generic run_bo_optimization
    X_final, y_final = run_bo_optimization(
        initial_X=X_init,
        initial_y=y_init,
        X_candidates=X_candidates,
        f_evaluate=objective,
        max_iter=max_iter,
        tol=tol,
        noise_std=noise_std,
        seed=seed,
        true_fn=objective 
    )
    return X_final, y_final


if __name__ == "__main__":
    run_2d_bo()
