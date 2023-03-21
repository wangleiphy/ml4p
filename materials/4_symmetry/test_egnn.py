import jax
import jax.numpy as jnp
from jax.config import config   
config.update("jax_enable_x64", True)
from scipy.stats import ortho_group
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
    rotate = ortho_group.rvs(dim)
    rotatez = egnn.apply(params, jnp.dot(x, rotate))
    assert jnp.allclose(rotatez, jnp.dot(z, rotate))

    # Test of permutation equivariance.
    print("---- Test permutation equivariance ----")
    P = np.random.permutation(n)
    Pz = egnn.apply(params, x[P, :])
    assert jnp.allclose(Pz, z[P, :])

test_egnn()
