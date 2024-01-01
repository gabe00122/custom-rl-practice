import jax
from jax import random, numpy as jnp
import optax
from ..networks.mlp import Mlp
from .actor_critic import ActorCritic
from .regularization import l2_regularization, l2_init_regularization, entropy_loss


def main():
    vec_num = 4
    observation_space = 8
    action_space = 4

    actor_optimizer = optax.adam(2**-10, b1=0.97, b2=0.97)
    critic_optimizer = optax.adam(2**-9, b1=0.97, b2=0.97)

    actor_model = Mlp(features=[64, 64, action_space], last_layer_scale=0.01)
    critic_model = Mlp(features=[64, 64, 1], last_layer_scale=1.0)

    actor_critic = ActorCritic(
        actor_model=actor_model,
        critic_model=critic_model,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        action_space=action_space,
        observation_space=observation_space,
    )

    key = random.PRNGKey(53245)
    params, key = actor_critic.init(key)

    # dummy inputs
    observation = jnp.zeros((observation_space,))
    vectorized_observation = jnp.zeros((vec_num, observation_space))
    actions = jnp.zeros((vec_num,), dtype=jnp.int32)
    advantages = jnp.zeros((vec_num,))

    lowered = {
        # "act": actor_critic.act.lower(actor_critic, params, observation, key),
        # "vectorized_act": actor_critic.vec_act.lower(actor_critic, params, vectorized_observation, key),
        # "action_log_prob": actor_critic.action_log_prob.lower(actor_critic, params.actor_params, observation),
        # "vectorized_state_value": actor_critic.vec_state_values.lower(
        #     actor_critic, params.critic_params, vectorized_observation
        # ),
        # "l2_regularization": jax.jit(l2_regularization).lower(params.actor_params, 0.001),
        # "l2_init_regularization": jax.jit(l2_init_regularization).lower(
        #     params.actor_params, params.actor_params, 0.001
        # ),
        # "entropy_loss": jax.jit(entropy_loss).lower(jnp.ones((action_space, action_space))),
        # "critic_loss": actor_critic.critic_loss.lower(
        #     actor_critic, params.critic_params, params.init_critic_params, vectorized_observation, advantages
        # ),
        # "update_critic": actor_critic.update_critic.lower(actor_critic, params, vectorized_observation, advantages),
        # "actor_loss": actor_critic.actor_loss.lower(
        #     actor_critic, params.actor_params, params.init_actor_params, vectorized_observation, actions, advantages
        # ),
        # "update_actor": actor_critic.update_actor.lower(actor_critic, params, vectorized_observation, actions, advantages),
        "vectorized_train_step": actor_critic.train_step.lower(
            actor_critic,
            params,
            vectorized_observation,
            actions,
            advantages,
            vectorized_observation,
            jnp.zeros((vec_num,), dtype=jnp.bool_),
            jnp.ones((vec_num,)),
        ),
    }

    for k, v in lowered.items():
        print(f"{k}: {v.compile().cost_analysis()[0]['flops']}")


if __name__ == "__main__":
    main()
