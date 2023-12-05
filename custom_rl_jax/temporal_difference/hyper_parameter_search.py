import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import pandas as pd
from hyper_params import HyperParams
import expected_sarsa
import sarsa
import q_learning
import double_q_learning
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


def training_run(name: str, policy, train_fn, params: HyperParams, env: gym.Env, episodes: int) -> np.ndarray:
    log_every = 1000
    rewards = np.zeros((episodes,))
    for i in range(episodes):
        rate = ((episodes - i) / episodes) * 0.2
        params['learning_rate'] = rate
        params['exploration'] = rate

        rewards[i] = train_fn(policy, params, env)
        if i % log_every == 0:
            print("{}: {}".format(name, i))

    return rewards


def training_run_eval(name: str, policy, train_fn, eval_fn, params: HyperParams, env: gym.Env,
                      episodes: int) -> tuple[np.ndarray, np.ndarray]:
    log_every = 1000
    rewards = np.zeros((episodes,))
    greedy_rewards = np.zeros((episodes,))

    for i in range(episodes):
        # rate = ((episodes - i) / episodes) * 0.2
        # params['learning_rate'] = rate
        # params['exploration'] = rate

        rewards[i] = train_fn(policy, params, env)
        greedy_rewards[i] = eval_fn(policy, params, env)
        if i % log_every == 0:
            print("{}: {}".format(name, i))

    return rewards, greedy_rewards


def main():
    env = gym.make("FrozenLake-v1", desc=generate_random_map(11))

    state_space = env.observation_space.n
    action_space = env.action_space.n

    hyper_params: HyperParams = {
        'exploration': 0.05,
        'discount': 0.99,
        'learning_rate': 0.05,
    }
    episodes = 300000

    s = training_run('sarsa', sarsa.create_policy(state_space, action_space), sarsa.train_episode, hyper_params, env,
                     episodes)
    es = training_run('expected_sarsa', expected_sarsa.create_policy(state_space, action_space),
                      expected_sarsa.train_episode,
                      hyper_params, env, episodes)
    q, greedy_q = training_run_eval('q_learning', q_learning.create_policy(state_space, action_space), q_learning.train_episode,
                          q_learning.eval_episode, hyper_params, env,
                          episodes)
    dq, greedy_dq = training_run_eval('double q_learning', double_q_learning.create_policy(state_space, action_space),
                           double_q_learning.train_episode, double_q_learning.eval_episode,
                           hyper_params, env, episodes)

    df = pd.DataFrame({
        'Sarsa': s,
        'Expected Sarsa': es,
        'Q learning': q,
        'Double Q Learning': dq,
        'Greedy Q learning': greedy_q,
        'Greedy Double Q Learning': greedy_dq
    })

    df = df.rolling(1000).mean()
    df.plot()

    # plt.plot(s, label='sarsa')
    # plt.plot(es, label='expected sarsa')
    plt.show()


if __name__ == '__main__':
    main()
