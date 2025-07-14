from typing import Optional, Callable
import jax.numpy as jnp
import matplotlib.pyplot as plt
from config import EPSILON, d_vector, n_vector, n_by_d_matrix, n_by_n_matrix, d_vector_to_real
from statsmodels.graphics.tsaplots import plot_acf


def plot_gp_results(
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
    alpha_EI_normalized = alpha_EI / (jnp.max(alpha_EI) + EPSILON)

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

