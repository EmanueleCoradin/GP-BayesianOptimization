import jax.numpy as jnp
import numpy as np

from svm_explorer import SVMExplorer
from bo_optimizer import run_bo_optimization

def main():
    svm_explorer = SVMExplorer()
    svm_explorer.load_data('./DATA/x_XGB_24.dat', './DATA/y_XGB_24.dat')

    # Objective function returns accuracy (positive)
    def svm_objective(params):
        log_C, log_gamma = params[0], params[1]
        return svm_explorer.evaluate_hyperparams(log_C, log_gamma)

    initial_points = np.array([[-4, -5], [7, 7], [2, 4], [4, -2.5], [-1, 3]])
    initial_values = np.array([
        svm_explorer.evaluate_hyperparams(*initial_points[0]),
        svm_explorer.evaluate_hyperparams(*initial_points[1]),
        svm_explorer.evaluate_hyperparams(*initial_points[2]),
        svm_explorer.evaluate_hyperparams(*initial_points[3]),
        svm_explorer.evaluate_hyperparams(*initial_points[4]),
    ])

    initial_X = jnp.array(initial_points)
    initial_y = jnp.array(initial_values)

    print(initial_X, initial_y)

    log_C_grid = jnp.linspace(-4.5, 8.5, 30)
    log_gamma_grid = jnp.linspace(-6, 8.5, 30)
    C_mesh, gamma_mesh = jnp.meshgrid(log_C_grid, log_gamma_grid)
    X_candidates = jnp.vstack([C_mesh.ravel(), gamma_mesh.ravel()]).T

    # Your BO code needs to **maximize** the objective.
    # If your BO routine is a minimizer, pass -accuracy inside it (see next note)
    X_final, y_final = run_bo_optimization(
        initial_X=initial_X,
        initial_y=initial_y,
        X_candidates=X_candidates,
        f_evaluate=svm_objective,
        max_iter=15,
        noise_std=0.0,
        seed=123
    )

    # Best accuracy: use argmax, since accuracy is positive and BO maximizes
    best_idx = jnp.argmax(y_final)
    best_params = X_final[best_idx]
    best_acc = y_final[best_idx]

    print(f"Best parameters found (log scale): {best_params}")
    print(f"Best accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
