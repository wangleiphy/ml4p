'''
https://arxiv.org/abs/2102.09844
'''
import jax
import jax.numpy as jnp
import numpy as np 
import haiku as hk
from typing import Optional
from functools import partial

class EGNN(hk.Module):

    def __init__(self, 
                 depth :int,
                 F:int, 
                 remat: bool = False, 
                 init_stddev:float = 0.01,
                 name: Optional[str] = None
                 ):
        super().__init__(name=name)
        assert (depth >= 2)
        self.depth = depth
        self.F = F
        self.remat = remat
        self.init_stddev = init_stddev
  
    def phie(self, x, h, d):
        n, dim = x.shape 
        rij = (jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim)))
        rij = jnp.sum(jnp.square(rij), axis=-1).reshape(n, n, 1)

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

    def __call__(self, x, h):
        assert x.ndim == 2
        n, dim = x.shape

        def block(x, h, d):
            mij = self.phie(x, h, d)

            xij = jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim))

            mask = ~np.eye(n, dtype=bool) # maskout diagonal
            mij = mij[mask].reshape(n, n-1, self.F)
            xij = xij[mask].reshape(n, n-1, dim)

            weight = self.phix(mij, d).reshape(n, n-1)/(n-1)

            x = x + jnp.einsum('ijd,ij->id', xij, weight)

            m = jnp.sum(mij, axis=1)

            h = self.phih(h, m, d) 
            return x, h 
        
        if self.remat:
            block = hk.remat(block, static_argnums=2)

        for d in range(self.depth):
            x, h = block(x, h, d)
       
        return x, h
