import pygame
import math
import config
from utils import toroidal_difference

class Obstacle:
    """
    Represents a circular static obstacle in the simulation.
    """
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    def is_point_overlapping(self, px, py, world_w, world_h):
        """
        Checks if a point (px, py) overlaps this obstacle, taking toroidal wrapping into account.
        """
        dx, dy = toroidal_difference(self.x, self.y, px, py, world_w, world_h)
        return math.hypot(dx, dy) < self.radius

    def is_circle_overlapping(self, cx, cy, cradius, world_w, world_h):
        """
        Checks if a circle with center (cx, cy) and radius cradius overlaps this obstacle,
        taking toroidal wrapping into account.
        """
        dx, dy = toroidal_difference(self.x, self.y, cx, cy, world_w, world_h)
        return math.hypot(dx, dy) < (self.radius + cradius)

    def draw(self, surface):
        """
        Renders the obstacle as a filled circle with a subtle highlight border for modern aesthetics.
        """
        # Main filled circle
        pygame.draw.circle(surface, config.OBSTACLE_COLOR, (int(self.x), int(self.y)), int(self.radius))
        # Subtle light-slate border
        pygame.draw.circle(surface, (65, 70, 85), (int(self.x), int(self.y)), int(self.radius), 2)


def get_fixed_obstacles():
    """
    Generates the hardcoded list of obstacles for Phase 1.
    Total: 7 obstacles placed strategically to have open space, narrow gaps, and dead ends.
    """
    return [
        # Gap 1 (Vertical path): gap of 10px between them (center dist 100, combined radius 90)
        Obstacle(300, 200, 40),
        Obstacle(300, 300, 50),
        
        # Gap 2 (Horizontal path): gap of 10px between them (center dist 120, combined radius 110)
        Obstacle(700, 500, 60),
        Obstacle(820, 500, 50),
        
        # Dead-end/Cluster: U-shaped pocket facing left (open to left, closed to right/up/down)
        Obstacle(900, 200, 40),
        Obstacle(970, 250, 40),
        Obstacle(900, 300, 40),
    ]
