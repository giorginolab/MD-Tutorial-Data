import numpy as np
import sys
import jax.numpy as jnp
from jax import grad, jit, vmap

print(sys.path)
z=np.zeros(1)

def zero(x):
    N=len(x)
    zg=np.zeros((N,3))
    return z,zg


@jit
def jitzero(x):
    N=len(x)
    zg=jnp.zeros((N,3))
    return z,zg
