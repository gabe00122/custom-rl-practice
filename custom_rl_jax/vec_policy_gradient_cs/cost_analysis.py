import jax
from jax import random, numpy as jnp
import optax
from ..networks.mlp import Mlp
from .actor_critic import ActorCritic
from .regularization import l2_regularization, l2_init_regularization, entropy_loss


def main():
    vec_num = 1
    observation_space = 8
    action_space = 4

    key = random.PRNGKey(53245)
    key, actor_key, critic_key, action_key = random.split(key, 4)
    observation = jnp.zeros((observation_space,))
    vectorized_observation = jnp.zeros((vec_num, observation_space))

    actor_optimizer = optax.adam(2**-10, b1=0.97, b2=0.97)
    critic_optimizer = optax.adam(2**-9, b1=0.97, b2=0.97)

    actor_model = Mlp(features=[64, 64, action_space], last_layer_scale=0.01)
    critic_model = Mlp(features=[64, 64, 1], last_layer_scale=1.0)

    actor_critic, key = ActorCritic.init(actor_model, critic_model, actor_optimizer, critic_optimizer, observation_space, vec_num, key)

    actions = jnp.zeros((vec_num,), dtype=jnp.int32)
    advantages = jnp.zeros((vec_num,))

    #print(actor_critic.vec_act.lower(actor_critic, vectorized_observation, action_key).compiler_ir())

    lowered = {
        "act": actor_critic.act.lower(actor_critic, observation, action_key),
        "vectorized_act": actor_critic.vec_act.lower(actor_critic, vectorized_observation, action_key),
        "action_log_prob": actor_critic.action_log_prob.lower(actor_critic, actor_critic.actor_params, observation),
        "vectorized_state_value": actor_critic.vec_state_values.lower(
            actor_critic, actor_critic.critic_params, vectorized_observation
        ),
        "l2_regularization": jax.jit(l2_regularization).lower(actor_critic.actor_params, 0.001),
        "l2_init_regularization": jax.jit(l2_init_regularization).lower(
            actor_critic.actor_params, actor_critic.actor_params, 0.001
        ),
        "entropy_loss": jax.jit(entropy_loss).lower(jnp.ones((vec_num, vec_num))),
        "critic_loss": actor_critic.critic_loss.lower(
            actor_critic, actor_critic.critic_params, vectorized_observation, advantages
        ),
        "update_critic": actor_critic.update_critic.lower(actor_critic, vectorized_observation, advantages),
        "actor_loss": actor_critic.actor_loss.lower(
            actor_critic, actor_critic.actor_params, vectorized_observation, actions, advantages
        ),
        "update_actor": actor_critic.update_actor.lower(actor_critic, vectorized_observation, actions, advantages),
        "vectorized_train_step": actor_critic.vec_train_step.lower(
            actor_critic,
            vectorized_observation,
            actions,
            vectorized_observation,
            advantages,
            jnp.zeros((vec_num,), dtype=jnp.bool_),
            action_key,
        ),
    }

    for k, v in lowered.items():
        print(f"{k}: {v.compile().cost_analysis()[0]['flops']}")


if __name__ == "__main__":
    main()
