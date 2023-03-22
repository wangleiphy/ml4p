import jax
import jax.numpy as jnp
from jax.config import config   
config.update("jax_enable_x64", True)
from jax.flatten_util import ravel_pytree
import numpy as np 
import haiku as hk

from egnn import EGNN

def test_egnn():
    depth = 4
    H = 24
    L = 1.234

    n, dim = 16, 3
    
    @hk.without_apply_rng
    @hk.transform
    def egnn(x, h):
        net = EGNN(depth, H)
        return net(x, h)
    

    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (n, dim), minval=0, maxval=L)
    h = jnp.ones((n, H))
    print(hk.experimental.tabulate(egnn)(x, h))

    params = egnn.init(key, x, h)

    raveled_params, _ = ravel_pytree(params)
    print ('# of params', raveled_params.size)

    
    z, hz = egnn.apply(params, x, h)
    
    # Test that flow results of two "equivalent"
    # Test the translation equivariance.
    print("---- Test translation equivariance ----")
    shift = jnp.array( np.random.randn(dim) )
    shiftz, shifth = egnn.apply(params, x + shift, h)
    assert jnp.allclose(shiftz, z + shift)
    assert jnp.allclose(shifth, hz)
   
    # Test the rotation equivariance
    print("---- Test rotation equivariance ----")
    rotate = jax.random.orthogonal(key, dim)
    rotatez, rotateh = egnn.apply(params, jnp.dot(x, rotate), h)
    assert jnp.allclose(rotatez, jnp.dot(z, rotate))
    assert jnp.allclose(rotateh, hz)

    # Test of permutation equivariance.
    print("---- Test permutation equivariance ----")
    P = np.random.permutation(n)
    Pz, Ph = egnn.apply(params, x[P, :], h[P, :])
    assert jnp.allclose(Pz, z[P, :])
    assert jnp.allclose(Ph, hz[P, :])


test_egnn()
