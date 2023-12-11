import flax.linen as nn
from .activation import mish
from typing import Sequence


class Mlp(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, feat in enumerate(self.features):
            not_last_feat = i != len(self.features) - 1
            initializer_scale = 1.0 if not_last_feat else 0.01

            x = nn.Dense(
                feat,
                name=f'Layer {i}',
                kernel_init=nn.initializers.variance_scaling(initializer_scale, 'fan_avg', 'uniform')
            )(x)
            if not_last_feat:
                x = mish(x)
        return x
