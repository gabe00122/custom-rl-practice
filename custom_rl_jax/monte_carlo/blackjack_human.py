import gymnasium as gym

env = gym.make('Blackjack-v1', natural=False, sab=False)


def human_play():
    obs, _ = env.reset()
    print('Player: {}, Dealer: {}, Usable Ace: {}'.format(obs[0], obs[1], obs[2]))
    while True:
        action = int(input('Action: '))
        obs, reward, terminated, truncated, info = env.step(action)
        print('Player: {}, Dealer: {}, Usable Ace: {}'.format(obs[0], obs[1], obs[2]))
        print('Reward: {}'.format(reward))
        if terminated:
            print('Game terminated')
            break
        if truncated:
            print('Game truncated')
            break
