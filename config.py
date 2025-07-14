import jax.numpy as jnp
import jax.random as random
import jax.numpy as jnp
from typing import Callable

d_vector = jnp.ndarray
n_vector = jnp.ndarray
n_by_d_matrix = jnp.ndarray
n_by_n_matrix = jnp.ndarray
d_vector_to_real = Callable[[d_vector], float]

EPSILON = 1e-12  # small constant to regularize operations
TOL = 1e-5
KEY = random.PRNGKey(0)
NOISE_STD = 1e-3  # small noise standard deviation

def f_true(x):
    """True function to be approximated."""
    return 2 * jnp.sin(1.5 * x) - 0.5 * x

def load_initial_data():
    X = jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = f_true(X.flatten())
    noise = NOISE_STD * random.normal(KEY, shape=y.shape)
    y += noise
    X_new = jnp.linspace(-5, 10, 200).reshape(-1, 1)
    return X, y, X_new

