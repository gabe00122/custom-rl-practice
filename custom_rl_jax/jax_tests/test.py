import flax.core
import jax
from jax import numpy as jnp, Array, random
from jax.typing import ArrayLike
from flax import linen as nn, struct
from ..networks.mlp import Mlp


key = random.PRNGKey(234)

key, init_key, action_key = random.split(key, 3)

network_input = jnp.ones((4,), dtype=jnp.float32)

actor_model = Mlp(features=[64, 64, 4])
params = actor_model.init(init_key, network_input)


class TestAlgo(struct.PyTreeNode):
    x: float = struct.field()
    y: float = struct.field(default=10)

    def double(self) -> 'TestAlgo':
        return TestAlgo(x=self.x * 2, y=self.y * 2)

    @classmethod
    def create(cls, x: float) -> 'TestAlgo':
        return cls(x=x)


test_algo = TestAlgo.create(x=1)


jit_double = jax.jit(test_algo.double)
test_algo = jit_double()
test2 = test_algo.replace(y=0)

print(f"{test2.x}, {test2.y}")
