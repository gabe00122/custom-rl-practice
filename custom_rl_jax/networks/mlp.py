import flax.linen as nn
from .activation import mish
from typing import Sequence


class Mlp(nn.Module):
    features: Sequence[int]
    last_layer_scale: float = 1.0

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, feat in enumerate(self.features):
            not_last_feat = i != len(self.features) - 1

            if not_last_feat:
                kernel_init = nn.initializers.he_normal()
            else:
                kernel_init = nn.initializers.variance_scaling(self.last_layer_scale, 'fan_avg', 'truncated_normal')

            x = nn.Dense(
                feat,
                name=f'Layer {i}',
                kernel_init=kernel_init
            )(x)
            if not_last_feat:
                x = mish(x)
        return x
