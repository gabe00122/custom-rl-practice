import pygame
from pygame import Vector2
from jaxtyping import Int, Array
from .simple_gridworld import State


class Visualizer:
    scale = 10

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((16 * self.scale * 2, 16 * self.scale * 2))
        self.clock = pygame.time.Clock()

    def draw(self, state: State):
        self.screen.fill((0, 0, 0))
        position = array_to_vector2(state.position) * self.scale * 2 + Vector2(self.scale)
        goal = array_to_vector2(state.goal) * self.scale * 2 + Vector2(self.scale)

        pygame.draw.circle(self.screen, (255, 0, 0), position, self.scale)
        pygame.draw.circle(self.screen, (0, 255, 0), goal, self.scale)
        pygame.display.flip()
        self.clock.tick_busy_loop(10)


def array_to_vector2(array: Int[Array, "2"]) -> Vector2:
    x, y = array.tolist()
    return Vector2(x, y)
