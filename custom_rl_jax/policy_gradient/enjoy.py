import gymnasium as gym
from ..networks.mlp import MlpSkip as Mlp
from .actor_critic import actor_critic
from jax import random
import orbax.checkpoint as ocp
from pathlib import Path


def main():
    env = gym.make("LunarLander-v2", render_mode="human")
    action_space = env.action_space.n
    state_space = env.observation_space.shape[0]

    key = random.PRNGKey(53245)

    actor_model = Mlp(features=[128, 128, action_space], last_layer_scale=0.01)
    critic_model = Mlp(features=[128, 128, 1], last_layer_scale=1.0)
    _, act = actor_critic(actor_model, critic_model)

    cp_path = Path('./gcu-less-l2/params').absolute()
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    raw_restored = orbax_checkpointer.restore(cp_path)
    actor_params = raw_restored['actor_training_state']['params']
    #critic_params = raw_restored['critic']

    key = random.PRNGKey(42)

    for _ in range(10):
        obs, info = env.reset()

        done = False
        total_reward = 0
        while not done:
            key, act_key = random.split(key)
            action = act(actor_params, obs, act_key)

            obs, reward, truncated, terminated, _ = env.step(action.item())
            total_reward += reward
            done = truncated or terminated

        print(total_reward)



if __name__ == '__main__':
    main()
