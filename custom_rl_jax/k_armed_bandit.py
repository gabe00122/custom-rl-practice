import matplotlib.pyplot as plt
import numpy as np
import jax
from jax import lax, random, numpy as jnp


class KArmedBandit:
    def __init__(self, k: int):
        self.k = k

    def init(self, key):
        return {"means": random.normal(key, (self.k,))}

    def act(self, state: dict[str, jnp.array], rng_key: random.PRNGKey, action: int):
        return state["means"][action] + random.normal(rng_key)


class Greedy:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def init(self, k: int):
        return {"estimations": jnp.zeros((k,))}

    def exploration(self, state: dict[str, jnp.array], rng_key: random.PRNGKey) -> jax.Array:
        return random.randint(rng_key, (), 0, state["estimations"].shape[0])

    def exploitation(self, state: dict[str, jnp.array]) -> int:
        return jnp.argmax(state["estimations"])

    def act(self, state: dict[str, jnp.array], rng_key: random.PRNGKey) -> int:
        test_key, exploration_key = random.split(rng_key)
        return lax.cond(random.uniform(test_key) < self.epsilon,
                        lambda: self.exploration(state, exploration_key),
                        lambda: self.exploitation(state)
                        )

    def update(self, state: dict[str, jnp.array], action: int, reward: float, n: int) -> dict[str, jnp.array]:
        estimations = state["estimations"]
        prev_estimation = estimations[action]
        next_estimation = prev_estimation + ((reward - prev_estimation) / (n + 1))
        updated_estimations = estimations.at[action].set(next_estimation)
        return {"estimations": updated_estimations}


if __name__ == "__main__":
    rng_key = random.PRNGKey(1)
    rng_key, subkey = random.split(rng_key)

    k = 10
    length = 500
    trails = 10000
    bandit = KArmedBandit(k)
    greedy = Greedy(0.9)


    def episode(state, i):
        rng_key = state
        rng_key, subkey = random.split(rng_key)
        bandit_state = bandit.init(subkey)
        greedy_state = greedy.init(k)

        def step(state, i):
            rng_key, greedy_state = state
            rng_key, greedy_key, reward_key = random.split(rng_key, 3)

            action = greedy.act(greedy_state, greedy_key)
            reward = bandit.act(bandit_state, reward_key, action)
            greedy_state = greedy.update(greedy_state, action, reward, i)
            return (rng_key, greedy_state), reward

        (rng_key, _), rewards = lax.scan(step, (rng_key, greedy_state), jnp.arange(length))
        return rng_key, rewards


    def episode_mean_reward(i, state):
        rng_key, mean_rewards = state
        rng_key, current_rewards = episode(rng_key, i)
        mean_rewards = mean_rewards + ((current_rewards - mean_rewards) / (i + 1))
        return rng_key, mean_rewards


    rng_key, rewards = lax.fori_loop(0, trails, episode_mean_reward, (rng_key, jnp.zeros((length,))))

    # greedy.epsilon = 1
    # rng_key, rewards2 = lax.fori_loop(0, trails, episode_mean_reward, (rng_key, jnp.zeros((length,))))

    plt.plot(np.array(rewards))
    # plt.plot(np.array(rewards2))
    plt.show()
