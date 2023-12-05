import gymnasium as gym
import pygame
from gymnasium.utils.play import play


class FrozenLake:
    pass


def main():
    env = gym.make("FrozenLake-v1", render_mode="human")
    env.reset()
    while True:
        event = pygame.event.wait()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                env.step(0)
            elif event.key == pygame.K_s:
                env.step(1)
            elif event.key == pygame.K_d:
                env.step(2)
            elif event.key == pygame.K_w:
                env.step(3)


if __name__ == '__main__':
    main()
