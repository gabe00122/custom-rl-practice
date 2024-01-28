from typing import TypedDict
from jax import Array

Metrics = TypedDict(
    "Metrics",
    {
        "state_value": Array,
        "td_error": Array,
        "actor_loss": Array,
        "critic_loss": Array,
        "entropy": Array,
    },
)
