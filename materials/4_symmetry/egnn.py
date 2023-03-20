'''
EGNN adapted to periodic system
https://arxiv.org/abs/2102.09844
'''
import jax
import jax.numpy as jnp
import numpy as np 
import haiku as hk
from typing import Optional
from functools import partial

def switch(r, rc):
    '''
    Eq(9) of 1805.09003
    #or smooth envelop following PhysRevE.74.066701 
    ((rc -rij)/rc)**3 * jnp.heaviside(rc-r, 0.0) 
    '''
    rcs = 0.5*rc
    _s = lambda r: (0.5*jnp.cos(np.pi*(r-rcs)/(rc-rcs))+0.5)
    s = jnp.where(r<rc, jnp.where(r<rcs, 1.0, _s(r)), 0.0)
    return s

class EGNN(hk.Module):

    def __init__(self, 
                 depth :int,
                 F:int, 
                 L:float,
                 remat: bool = False, 
                 init_stddev:float = 0.01,
                 name: Optional[str] = None
                 ):
        super().__init__(name=name)
        assert (depth >= 2)
        self.depth = depth
        self.F = F
        self.L = L
        self.rc = 0.5*L
        self.remat = remat
        self.init_stddev = init_stddev
  
    def phie(self, x, h, d):
        n, dim = x.shape 
        rij = (jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim)))
        #|r| calculated with periodic consideration
        rij = jnp.linalg.norm(jnp.sin(np.pi*rij/self.L)+jnp.eye(n)[..., None], axis=-1)*(1.0-jnp.eye(n))
        rij = rij.reshape(n, n, 1)

        mlp = hk.nets.MLP([self.F, self.F], w_init=hk.initializers.TruncatedNormal(self.init_stddev), activation=jax.nn.silu, name=f"edge_mlp_{d}") 
        @partial(hk.vmap, in_axes=(0, None, 0), out_axes=0, split_rng=False)
        @partial(hk.vmap, in_axes=(None, 0, 0), out_axes=0, split_rng=False)
        def phi(hi, hj, r):
            hhr = jnp.concatenate([hi, hj, r], axis=0)
            return mlp(hhr)
        return phi(h, h, rij)

    def phix(self, mij, d):
        mlp = hk.nets.MLP([self.F, 1], w_init=hk.initializers.TruncatedNormal(self.init_stddev), activation=jax.nn.silu, name=f"coord_mlp_{d}")
        return mlp(mij)

    def phih(self, h, m, d):
        hm = jnp.concatenate([h, m], axis=-1)
        mlp = hk.nets.MLP([self.F, self.F], w_init=hk.initializers.TruncatedNormal(self.init_stddev), activation=jax.nn.silu, name=f"node_mlp_{d}")
        return mlp(hm) + h 

    def __call__(self, x):
        assert x.ndim == 2
        n, dim = x.shape
        h = jnp.ones((n, self.F))

        def block(x, h, d):
            mij = self.phie(x, h, d)

            xij = jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim))
            xij = xij - self.L*jnp.rint(xij/self.L)

            mask = ~np.eye(n, dtype=bool) # maskout diagonal
            mij = mij[mask].reshape(n, n-1, self.F)
            xij = xij[mask].reshape(n, n-1, dim)
            rij = jnp.linalg.norm(xij, axis=-1)

            weight = self.phix(mij, d).reshape(n, n-1)/(n-1)
            weight *= switch(rij, self.rc)

            x = x + jnp.einsum('ijd,ij->id', xij, weight)

            m = jnp.sum(mij, axis=1)

            h = self.phih(h, m, d) 
            return x, h 
        
        if self.remat:
            block = hk.remat(block, static_argnums=2)

        for d in range(self.depth):
            x, h = block(x, h, d)
       
        final = hk.Linear(dim, w_init=hk.initializers.TruncatedNormal(self.init_stddev), with_bias=False)
        return final(h) + x
