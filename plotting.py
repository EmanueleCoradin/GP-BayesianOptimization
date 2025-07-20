from typing import Optional, Callable
import jax.numpy as jnp
import matplotlib.pyplot as plt
from config import EPSILON, d_vector, n_vector, n_by_d_matrix, n_by_n_matrix, d_vector_to_real
#from statsmodels.graphics.tsaplots import plot_acf


def plot_gp_results(
    X_train: n_by_d_matrix,
    y_train: n_vector,
    X_new: n_by_d_matrix,
    posterior_means: n_vector,
    posterior_stds: n_vector,
    alpha_EI: n_vector,
    idx_next: int,
    noise_std: float,
    true_fn: Optional[Callable[[d_vector], float]] = None,
    title="GP posterior vs True function",
):

    x1_vals = jnp.unique(X_new[:, 0])
    x2_vals = jnp.unique(X_new[:, 1])
    n1, n2 = len(x1_vals), len(x2_vals)

    Z_mean = posterior_means.reshape(n2, n1)
    Z_ei = alpha_EI.reshape(n2, n1)
    Z_std = posterior_stds.reshape(n2, n1)

    x1_grid, x2_grid = jnp.meshgrid(x1_vals, x2_vals)

    fig, axs = plt.subplots(2, 2, figsize=(8, 6))

    # ===== Top-Left: GP posterior mean + EI contour
    c0 = axs[0, 0].contourf(x1_grid,
                            x2_grid,
                            Z_mean,
                            cmap="plasma",
                            levels=30)
    axs[0, 0].contour(x1_grid,
                      x2_grid,
                      Z_ei,
                      colors="magenta",
                      linewidths=0.8,
                      linestyles="dashed")
    axs[0, 0].scatter(X_train[:, 0],
                      X_train[:, 1],
                      c="red",
                      edgecolor="k",
                      label="Training points")
    axs[0, 0].scatter(X_new[idx_next, 0],
                      X_new[idx_next, 1],
                      c="dodgerblue",
                      marker="*",
                      s=100,
                      label="Next sample")
    axs[0, 0].set_title("GP posterior mean + EI contours")
    axs[0, 0].set_xlabel("x₁")
    axs[0, 0].set_ylabel("x₂")
    axs[0, 0].legend()
    fig.colorbar(c0, ax=axs[0, 0])

    # ===== Top-Right: True function
    if true_fn is not None:
        Z_true = jnp.array([true_fn(x) for x in X_new])
        Z_true = Z_true.reshape(n2, n1)

        c1 = axs[0, 1].contourf(x1_grid,
                                x2_grid,
                                Z_true,
                                cmap="plasma",
                                levels=30)
        max_idx = jnp.argmax(Z_true)
        max_pos = X_new[max_idx]
        axs[0, 1].scatter(max_pos[0],
                          max_pos[1],
                          c="deeppink",
                          marker="*",
                          s=120,
                          label="True max")

        axs[0, 1].set_title("True function")
        axs[0, 1].set_xlabel("x₁")
        axs[0, 1].set_ylabel("x₂")
        axs[0, 1].legend()
        fig.colorbar(c1, ax=axs[0, 1])
    else:
        axs[0, 1].axis("off")

    # ===== Bottom-Left: Posterior std dev
    c2 = axs[1, 0].contourf(x1_grid, x2_grid, Z_std, cmap="viridis", levels=30)
    axs[1, 0].scatter(X_train[:, 0],
                      X_train[:, 1],
                      c="red",
                      edgecolor="k",
                      s=30)
    axs[1, 0].set_title("Posterior standard deviation")
    axs[1, 0].set_xlabel("x₁")
    axs[1, 0].set_ylabel("x₂")
    fig.colorbar(c2, ax=axs[1, 0])

    # ===== Bottom-Right: Expected Improvement (dedicated)
    c3 = axs[1, 1].contourf(x1_grid, x2_grid, Z_ei, cmap="inferno", levels=30)
    axs[1, 1].scatter(X_new[idx_next, 0],
                      X_new[idx_next, 1],
                      c="dodgerblue",
                      marker="*",
                      s=100,
                      label="Next sample")
    axs[1, 1].set_title("Expected Improvement")
    axs[1, 1].set_xlabel("x₁")
    axs[1, 1].set_ylabel("x₂")
    axs[1, 1].legend()
    fig.colorbar(c3, ax=axs[1, 1])

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_gp_results_1D(
    X_train,
    y_train,
    X_new,
    posterior_means,
    posterior_stds,
    alpha_EI,
    idx_next,
    noise_std,
    true_fn=None,
    xlabel="x",
    ylabel="f(x)",
    title="Gaussian Process reconstruction",
):
    x_next = X_new[idx_next]
    if jnp.max(alpha_EI) > 0:
        alpha_EI_normalized = alpha_EI / jnp.max(alpha_EI)
    else:
        alpha_EI_normalized = alpha_EI

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot GP posterior
    ax1.plot(X_new.flatten(), posterior_means, "b-", label="GP mean")
    ax1.fill_between(
        X_new.flatten(),
        posterior_means - 2 * posterior_stds,
        posterior_means + 2 * posterior_stds,
        color="blue",
        alpha=0.2,
        label="95% CI",
    )
    ax1.errorbar(
        X_train.flatten(),
        y_train,
        yerr=noise_std,
        fmt="o",
        color="red",
        label="Training data",
    )

    if true_fn is not None:
        y_true = true_fn(X_new.flatten())
        ax1.plot(X_new.flatten(), y_true, "g--", label="True function")

    ax1.axvline(x_next, color="black", linestyle="--", label="x_next")
    ax1.plot(x_next, posterior_means[idx_next], "ko", label="Next point")

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(X_new.flatten(),
             alpha_EI_normalized,
             "m-",
             label="Expected Improvement (normalized)",
             alpha=0.6)
    ax2.set_ylabel("Normalized EI")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.show()

'''
def plot_traces(samples_dict, param_name):
    chains = samples_dict[param_name]
    for i in range(chains.shape[0]):
        plt.plot(chains[i], label=f"chain {i}")
    plt.title(f"Trace for {param_name}")
    plt.xlabel("Sample")
    plt.ylabel(param_name)
    plt.legend()
    plt.show()


def plot_acf_chain(chain, param_name, chain_idx=0):
    plt.title(f"Autocorrelation for {param_name} (chain {chain_idx})")
    plot_acf(chain[chain_idx], lags=50)
    plt.show()
'''