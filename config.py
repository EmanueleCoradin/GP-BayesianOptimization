import jax.numpy as jnp
import jax.random as random
import jax.numpy as jnp
from typing import Callable
from svm_explorer import SVMExplorer

# TEMP
import time

d_vector = jnp.ndarray
n_vector = jnp.ndarray
n_by_d_matrix = jnp.ndarray
n_by_n_matrix = jnp.ndarray
d_vector_to_real = Callable[[d_vector], float]

EPSILON = 1e-12  # small constant to regularize operations
TOL = 1e-5
KEY = random.PRNGKey(0)
NOISE_STD = 0  # small noise standard deviation

svm_explorer = SVMExplorer()

# Call this early in your main before using the objective
def prepare_svm_data():
    svm_explorer.load_data()

def f_svm(x: jnp.ndarray) -> float:
    log_C, log_gamma = x[0], x[1]
    acc = svm_explorer.evaluate_hyperparams(log_C, log_gamma)
    return acc  # We negate because Bayesian optimization assumes a maximization problem

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
    return -(jnp.sin(x1) * jnp.exp(1-jnp.cos(x2))**2 + jnp.cos(x2) *jnp.exp (1-jnp.sin(x1))**2 + (x1 - x2)**2 )

def f_ursem(x: d_vector) -> float:
    x1,x2=x[0],x[1]
    return -(-0.9* x1**2 + (x2**2-4.5*x2**2)*x1*x2+4.7*jnp.cos(3*x1-x2**2*(2+x1))*jnp.sin(2.5*x1))

def f_1D(x):
    """True function to be approximated."""
    return 2 * jnp.sin(1.5 * x) - 0.5 * x

def f_goldstein_price(x: d_vector):
    """
    Goldstein–Price function.
    Global minimum: f(0, -1) = 3
    Domain: x, y in [-2, 2]
    """
    x, y = x[0], x[1]
    term1 = (1 + (x + y + 1)**2 *
             (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2))
    
    term2 = (30 + (2*x - 3*y)**2 *
             (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    
    return - jnp.log(term1) - jnp.log(term2)

#def f_1D(x):
#    return (6*x - 2)**2 * jnp.sin(12*x - 4)

def load_initial_data(function: str = "f_hosaki", initial_points: int = 7):
    key1, key2 = random.split(KEY)

    function_map = {
        "f_hosaki": f_hosaki,
        "f_rosenbrock": f_rosenbrock,
        "f_bird": f_bird,
        "f_ursem": f_ursem,
        "f_svm": f_svm,
        "f_goldstein_price": f_goldstein_price,
    }

    bounds_map = {
        "f_hosaki": (jnp.array([0.0, 0.0]), jnp.array([5.0, 6.0])),
        "f_rosenbrock": (jnp.array([-1.0, -1.0]), jnp.array([0.5, 1.0])),
        "f_bird": (jnp.array([-6.0, -6.0]), jnp.array([6.0, 6.0])),
        "f_ursem":(jnp.array([-1.0,-1.0]),jnp.array([1.5,1.5])),
        "f_svm": (jnp.array([-4.5, -6.5]), jnp.array([8.5, 8.5])),
        "f_goldstein_price": (jnp.array([-2., -2.]), jnp.array([2., 2.])),
    }
    # TEMP
    # key = random.PRNGKey(int(time.time()))
    # key1, key2 = random.split(key)

    f = function_map.get(function.lower(), f_hosaki)
    
    minval, maxval = bounds_map.get(function.lower(), (jnp.array([0.0, 0.0]), jnp.array([5.0, 6.0])))
    if function.lower() == "f_svm":
        prepare_svm_data()
    # TEMP: less points initial 
    X = random.uniform(
        key1,
        shape=(initial_points, 2),
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

    return f, X, y, X_new



def load_initial_data_1D():
    X = jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = f_1D(X.flatten())
    noise = NOISE_STD * random.normal(KEY, shape=y.shape)
    y += noise
    X_new = jnp.linspace(-5, 10, 200).reshape(-1, 1)
    return X, y, X_new
