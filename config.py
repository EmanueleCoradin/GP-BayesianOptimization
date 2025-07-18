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
NOISE_STD = 0  # small noise standard deviation


def f_hosaki(x: d_vector) -> float:
    """Hosaki function."""
    x1, x2 = x[0], x[1]
    return -(1 - 8 * x1 + 7 * x1**2 - 7. / 3. * x1**3 +
             1. / 4. * x1**4) * x2**2 * jnp.exp(-x2)

def f_rosenbrock(x: d_vector) -> float:
    """Rosenbrock function."""
    x1, x2 = x[0], x[1]
    return -(74 + 100 * (x2 - x1**2)**2 + (1 - x1)**2 - 400 * jnp.exp(-((x1+1)**2 + (x2+1)**2)/0.1))

def f_bird(x: d_vector) -> float:
    """Bird function."""
    x1, x2 = x[0], x[1]
    return -jnp.sin(x1) * jnp.exp(1-jnp.cos(x2))**2 + jnp.cos(x2) *jnp.exp (1-jnp.sin(x1))**2 + (x1 - x2)**2 

def f_sin(x):
    """True function to be approximated."""
    return 2 * jnp.sin(1.5 * x) - 0.5 * x


def load_initial_data(function: str = "f_hosaki"):
    key1, key2 = random.split(KEY)

    function_map = {
        "f_hosaki": f_hosaki,
        "f_rosenbrock": f_rosenbrock,
        "f_bird": f_bird,
    }

    bounds_map = {
        "f_hosaki": (jnp.array([0.0, 0.0]), jnp.array([5.0, 6.0])),
        "f_rosenbrock": (jnp.array([-1.0, -1.0]), jnp.array([0.5, 1.0])),
        "f_bird": (jnp.array([-6.0, -6.0]), jnp.array([6.0, 6.0])),
    }

    f = function_map.get(function.lower(), f_hosaki)

    minval, maxval = bounds_map.get(function.lower(), (jnp.array([0.0, 0.0]), jnp.array([5.0, 6.0])))

    X = random.uniform(
        key1,
        shape=(7, 2),
        minval=minval,
        maxval=maxval
    )
    y = jnp.array([f(x) for x in X])
    noise = NOISE_STD * random.normal(key2, shape=y.shape)
    y += noise

    # grid of candidate evaluation points
    x1_vals = jnp.linspace(minval[0], maxval[0], 50)
    x2_vals = jnp.linspace(minval[1], maxval[1], 50)
    X_new = jnp.stack(jnp.meshgrid(x1_vals, x2_vals), axis=-1).reshape(-1, 2)

    return X, y, X_new



def load_initial_data_1D():
    X = jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = f_sin(X.flatten())
    noise = NOISE_STD * random.normal(KEY, shape=y.shape)
    y += noise
    X_new = jnp.linspace(-5, 10, 200).reshape(-1, 1)
    return X, y, X_new
