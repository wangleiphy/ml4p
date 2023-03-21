import jax
import jax.numpy as jnp
from jax.config import config   
config.update("jax_enable_x64", True)
from jax.flatten_util import ravel_pytree
from functools import partial
import numpy as np 
import haiku as hk

from egnn import EGNN

@partial(jax.vmap, in_axes=(0, None))
def rotate(x, angle):
    """
        Euler rotation of x.
        Input:
            x.shape: (3,)
            angle.shape: (3,)
    """
    # Create rotation matrices
    Rx = jnp.array([[1, 0, 0], [0, jnp.cos(angle[0]), -jnp.sin(angle[0])], [0, jnp.sin(angle[0]), jnp.cos(angle[0])]])
    Ry = jnp.array([[jnp.cos(angle[1]), 0, jnp.sin(angle[1])], [0, 1, 0], [-jnp.sin(angle[1]), 0, jnp.cos(angle[1])]])
    Rz = jnp.array([[jnp.cos(angle[2]), -jnp.sin(angle[2]), 0], [jnp.sin(angle[2]), jnp.cos(angle[2]), 0], [0, 0, 1]])
    
    # Apply rotations z -> y -> x
    R = jnp.dot(Rz, jnp.dot(Ry, Rx))
    return jnp.dot(R, x)

def test_egnn():
    depth = 4
    H = 24
    L = 1.234

    n, dim = 16, 3
    
    @hk.without_apply_rng
    @hk.transform
    def egnn(x):
        net = EGNN(depth, H, L)
        return net(x)
    

    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (n, dim), minval=0, maxval=L)
    print(hk.experimental.tabulate(egnn)(x))

    params = egnn.init(key, x)

    raveled_params, _ = ravel_pytree(params)
    print ('# of params', raveled_params.size)

    
    z = egnn.apply(params, x)
    
    # Test that flow results of two "equivalent"
    # Test the translation equivariance.
    print("---- Test translation equivariance ----")
    shift = jnp.array( np.random.randn(dim) )
    shiftz = egnn.apply(params, x + shift)
    assert jnp.allclose(shiftz, z + shift)
   
    # Test the rotation equivariance
    print("---- Test rotation equivariance ----")
    angle = jax.random.uniform(key, (dim,), minval=0, maxval=2*jnp.pi)
    rotatex = rotate(x, angle)
    rotatez = egnn.apply(params, rotatex)
    assert jnp.allclose(rotatez, rotate(z, angle))

    # Test of permutation equivariance.
    print("---- Test permutation equivariance ----")
    P = np.random.permutation(n)
    Pz = egnn.apply(params, x[P, :])
    assert jnp.allclose(Pz, z[P, :])

test_egnn()
