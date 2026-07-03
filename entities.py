import pygame
import random
import math
import numpy as np

# Import constants and helpers
import config
from utils import toroidal_difference
from neural_network import NeuralNetwork

class Agent:
    """
    Base Agent class encapsulating position, velocity, heading, and 
    common physics-based steering and boundary wrap-around behavior.
    """
    def __init__(self, x, y, heading, speed_factor, max_speed, max_accel, max_turn_rate, color, length, width, radius=6.0):
        """
        Initializes an Agent with position, velocity, and physics constraints.
        """
        self.x = x
        self.y = y
        self.speed_factor = speed_factor
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_turn_rate = max_turn_rate
        self.color = color
        self.length = length
        self.width = width
        self.radius = radius
        
        # Initialize velocity vector based on spawn heading and personal max speed
        personal_max_speed = self.speed_factor * self.max_speed
        self.vx = personal_max_speed * math.cos(heading)
        self.vy = personal_max_speed * math.sin(heading)
        
        # Wander states (used by Predator or defaults)
        self.target_heading = heading
        self.wander_timer = random.randint(0, config.WANDER_INTERVAL)

    @property
    def heading(self):
        """
        Derives heading angle in radians from the velocity vector.
        Heading is not stored independently.
        """
        return math.atan2(self.vy, self.vx)

    def move(self, ax, ay, world_w, world_h, obstacles=None, collision_mode='OFF'):
        """
        Updates velocity vector and position using acceleration, capping top-speed
        and clamping heading change rate.
        """
        old_heading = self.heading

        new_vx = self.vx + ax
        new_vy = self.vy + ay
        
        speed = math.hypot(new_vx, new_vy)
        if speed > 0.0001:
            temp_heading = math.atan2(new_vy, new_vx)
        else:
            temp_heading = old_heading
            
        heading_diff = temp_heading - old_heading
        # Normalize to [-pi, pi]
        heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi
        
        # Clamp turn rate
        if abs(heading_diff) > self.max_turn_rate:
            heading_diff = math.copysign(self.max_turn_rate, heading_diff)
            actual_heading = old_heading + heading_diff
        else:
            actual_heading = temp_heading
            
        personal_max_speed = self.speed_factor * self.max_speed
        if speed > personal_max_speed:
            speed = personal_max_speed
            
        self.vx = speed * math.cos(actual_heading)
        self.vy = speed * math.sin(actual_heading)
        
        if obstacles and collision_mode != 'OFF':
            from physics.collision_solver import resolve_agent_movement
            resolve_agent_movement(self, obstacles, world_w, world_h, collision_mode, N=config.COLLISION_SUBSTEPS)
        else:
            self.x += self.vx
            self.y += self.vy
            self.handle_boundaries(world_w, world_h)

    def handle_boundaries(self, world_w, world_h):
        """
        Wraps coordinate positions around screen edges dynamically.
        Velocity/heading vectors are preserved.
        """
        self.x = self.x % world_w
        self.y = self.y % world_h

    def draw(self, surface, draw_details=True):
        """
        Renders the agent as an oriented isosceles triangle.
        Supports simplified uniform coloring when draw_details is False.
        """
        h = self.heading
        cos_h = math.cos(h)
        sin_h = math.sin(h)
        
        # Point 1: Nose
        tip_x = self.x + self.length * cos_h
        tip_y = self.y + self.length * sin_h
        
        # Point 2: Left tail corner
        left_x = self.x - (self.length / 2) * cos_h - (self.width / 2) * sin_h
        left_y = self.y - (self.length / 2) * sin_h + (self.width / 2) * cos_h
        
        # Point 3: Right tail corner
        right_x = self.x - (self.length / 2) * cos_h + (self.width / 2) * sin_h
        right_y = self.y - (self.length / 2) * sin_h - (self.width / 2) * cos_h
        
        color = (100, 140, 180) if not draw_details else self.color
        
        pygame.draw.polygon(surface, color, [
            (tip_x, tip_y),
            (left_x, left_y),
            (right_x, right_y)
        ])


class Fish(Agent):
    """
    Prey Fish agent driven independently by a Neural Network brain.
    """
    def __init__(self, x, y, heading, speed_factor, color):
        super().__init__(
            x=x, y=y, heading=heading, speed_factor=speed_factor,
            max_speed=config.MAX_SPEED, max_accel=config.MAX_ACCELERATION, max_turn_rate=config.MAX_TURN_RATE,
            color=color, length=14, width=8, radius=config.FISH_COLLISION_RADIUS
        )
        
        # Brain Network (5 inputs -> 8 hidden -> 2 outputs)
        self.brain = NeuralNetwork(input_size=5, hidden_size=8, output_size=2)
        
        # Fitness Score
        self.fitness = 0.0
        
        # Loaded champion flag (for visual highlights)
        self.is_champion = False
        
        # Sensor Fields
        self.sensor_nearest_dist = float('inf')
        self.sensor_nearest_angle = 0.0
        self.sensor_wall_dist = 0.0
        self.sensor_predator_dist = float('inf')
        self.sensor_predator_angle = 0.0
        
        # Decision outputs cache
        self.last_steer_output = 0.0
        self.last_accel_output = 0.0
        self.ax_input = 0.0
        self.ay_input = 0.0
        
        # Debug references
        self.nearest_neighbor = None
        self.nearest_neighbor_dist = float('inf')

    def sense(self, all_fish, predator, world_w, world_h):
        """
        Updates fish sensors by scanning neighbors, boundaries, and the predator.
        """
        nearest_fish = None
        min_dist = float('inf')
        best_dx = 0.0
        best_dy = 0.0
        
        # Neighbor search
        for other in all_fish:
            if other is self:
                continue
                
            dx, dy = toroidal_difference(self.x, self.y, other.x, other.y, world_w, world_h)
            dist = math.hypot(dx, dy)
            if dist < min_dist:
                min_dist = dist
                nearest_fish = other
                best_dx = dx
                best_dy = dy
                
        self.nearest_neighbor = nearest_fish
        self.nearest_neighbor_dist = min_dist
        
        # Populate nearest neighbor sensors
        if nearest_fish is not None and min_dist <= config.VISION_RADIUS:
            self.sensor_nearest_dist = min_dist
            abs_angle = math.atan2(best_dy, best_dx)
            rel_angle = abs_angle - self.heading
            self.sensor_nearest_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
        else:
            self.sensor_nearest_dist = float('inf')
            self.sensor_nearest_angle = 0.0
            
        # Calculate wall distance along the heading vector
        self.sensor_wall_dist = self.get_distance_to_edge(world_w, world_h)
        
        # Calculate predator distance/angle sensors
        p_dx, p_dy = toroidal_difference(self.x, self.y, predator.x, predator.y, world_w, world_h)
        dist_p = math.hypot(p_dx, p_dy)
        self.sensor_predator_dist = dist_p
        
        if dist_p > 0.0001:
            abs_angle_p = math.atan2(p_dy, p_dx)
            rel_angle_p = abs_angle_p - self.heading
            self.sensor_predator_angle = (rel_angle_p + math.pi) % (2 * math.pi) - math.pi
        else:
            self.sensor_predator_angle = 0.0

    def think(self, world_w, world_h):
        """
        Feeds normalized sensors to the brain, runs forward inference,
        and translates outputs to steering acceleration forces.
        """
        # Normalizations
        norm_dist = min(self.sensor_nearest_dist / config.VISION_RADIUS, 1.0)
        norm_angle = self.sensor_nearest_angle / math.pi
        norm_wall = min(self.sensor_wall_dist / max(world_w, world_h), 1.0)
        norm_pred_dist = min(self.sensor_predator_dist / config.VISION_RADIUS, 1.0)
        norm_pred_angle = self.sensor_predator_angle / math.pi
        
        inputs = [norm_dist, norm_angle, norm_wall, norm_pred_dist, norm_pred_angle]
        
        # Forward pass
        outputs = self.brain.forward(inputs)
        self.last_steer_output = outputs[0]
        self.last_accel_output = outputs[1]
        
        # Desired heading
        desired_heading = self.heading + self.last_steer_output * self.max_turn_rate
        
        # Desired speed
        personal_max_speed = self.speed_factor * config.MAX_SPEED
        desired_speed = ((self.last_accel_output + 1.0) / 2.0) * personal_max_speed
        
        desired_vx = desired_speed * math.cos(desired_heading)
        desired_vy = desired_speed * math.sin(desired_heading)
        
        self.ax_input = desired_vx - self.vx
        self.ay_input = desired_vy - self.vy
        
        # Clamp acceleration
        accel_mag = math.hypot(self.ax_input, self.ay_input)
        if accel_mag > self.max_accel:
            self.ax_input = (self.ax_input / accel_mag) * self.max_accel
            self.ay_input = (self.ay_input / accel_mag) * self.max_accel

    def think_manual(self, keys):
        """
        Manually steers the fish using keyboard arrow keys.
        """
        steer = 0.0
        accel = 0.0
        
        if keys[pygame.K_LEFT]:
            steer = -1.0
        if keys[pygame.K_RIGHT]:
            steer = 1.0
        if keys[pygame.K_UP]:
            accel = 1.0
        if keys[pygame.K_DOWN]:
            accel = -1.0
            
        self.last_steer_output = steer
        self.last_accel_output = accel
        
        # Desired heading
        desired_heading = self.heading + steer * self.max_turn_rate
        
        # Desired speed
        personal_max_speed = self.speed_factor * config.MAX_SPEED
        desired_speed = ((accel + 1.0) / 2.0) * personal_max_speed
        
        desired_vx = desired_speed * math.cos(desired_heading)
        desired_vy = desired_speed * math.sin(desired_heading)
        
        self.ax_input = desired_vx - self.vx
        self.ay_input = desired_vy - self.vy
        
        # Clamp acceleration
        accel_mag = math.hypot(self.ax_input, self.ay_input)
        if accel_mag > self.max_accel:
            self.ax_input = (self.ax_input / accel_mag) * self.max_accel
            self.ay_input = (self.ay_input / accel_mag) * self.max_accel

    def update(self, world_w, world_h, obstacles=None, collision_mode='OFF'):
        """
        Moves the fish based on neural decisions and updates its fitness score.
        """
        # =====================================================================
        # STAGE 6 HOOK: Fitness score tracker
        # - Fish gains fitness points for every frame they survive.
        # =====================================================================
        self.move(self.ax_input, self.ay_input, world_w, world_h, obstacles, collision_mode)
        
        # Fitness Accumulation
        self.fitness += config.FITNESS_SURVIVAL_REWARD
        
        current_speed = math.hypot(self.vx, self.vy)
        self.fitness += current_speed * config.FITNESS_SPEED_MULT
        
        if self.sensor_wall_dist < config.FITNESS_WALL_PENALTY_THRESHOLD:
            wall_ratio = self.sensor_wall_dist / config.FITNESS_WALL_PENALTY_THRESHOLD
            penalty = (1.0 - wall_ratio) * config.FITNESS_WALL_PENALTY_MAX
            self.fitness -= penalty
            
        if self.fitness < 0.0:
            self.fitness = 0.0

    def draw(self, surface, draw_details=True):
        """
        Extends draw to render a special highlight if the fish is a loaded champion.
        """
        super().draw(surface, draw_details)
        if draw_details and self.is_champion:
            pygame.draw.circle(surface, (255, 215, 0), (int(self.x), int(self.y)), 12, 1)

    def get_distance_to_edge(self, world_w, world_h):
        """
        Computes distance to screen boundary along heading.
        """
        h = self.heading
        cos_h = math.cos(h)
        sin_h = math.sin(h)
        
        dist_x = float('inf')
        if cos_h > 0:
            dist_x = (world_w - self.x) / cos_h
        elif cos_h < 0:
            dist_x = -self.x / cos_h
            
        dist_y = float('inf')
        if sin_h > 0:
            dist_y = (world_h - self.y) / sin_h
        elif sin_h < 0:
            dist_y = -self.y / sin_h
            
        return min(dist_x, dist_y)


class Predator(Agent):
    """
    Predator agent. Chases the closest prey fish using rule-based intercept.
    """
    def __init__(self, x, y, heading):
        super().__init__(
            x=x, y=y, heading=heading, speed_factor=1.0,
            max_speed=config.PREDATOR_MAX_SPEED, max_accel=config.PREDATOR_MAX_ACCELERATION, max_turn_rate=config.PREDATOR_MAX_TURN_RATE,
            color=(231, 76, 60), length=22, width=12, radius=config.PREDATOR_COLLISION_RADIUS
        )
        self.target_fish = None
        self.target_fish_dist = float('inf')

    def hunt(self, fish_school, world_w, world_h, obstacles=None, collision_mode='OFF'):
        """
        Steers toward closest fish, otherwise wanders.
        """
        nearest_fish = None
        min_dist = float('inf')
        best_dx = 0.0
        best_dy = 0.0
        
        # Nearest target search
        for fish in fish_school:
            dx, dy = toroidal_difference(self.x, self.y, fish.x, fish.y, world_w, world_h)
            dist = math.hypot(dx, dy)
            if dist < min_dist:
                min_dist = dist
                nearest_fish = fish
                best_dx = dx
                best_dy = dy
                
        self.target_fish = nearest_fish
        self.target_fish_dist = min_dist
        
        if nearest_fish is not None and min_dist <= config.PREDATOR_VISION_RADIUS:
            # Active Chase
            self.target_heading = math.atan2(best_dy, best_dx)
        else:
            # Wander (temporary, pre-neural behavior)
            self.wander_timer -= 1
            if self.wander_timer <= 0:
                self.wander_timer = random.randint(int(config.WANDER_INTERVAL * 0.7), int(config.WANDER_INTERVAL * 1.3))
                self.target_heading = self.heading + random.uniform(-config.WANDER_FORCE, config.WANDER_FORCE)

        # Steering
        desired_vx = self.max_speed * math.cos(self.target_heading)
        desired_vy = self.max_speed * math.sin(self.target_heading)
        
        ax = desired_vx - self.vx
        ay = desired_vy - self.vy
        
        accel_mag = math.hypot(ax, ay)
        if accel_mag > self.max_accel:
            ax = (ax / accel_mag) * self.max_accel
            ay = (ay / accel_mag) * self.max_accel
            
        self.move(ax, ay, world_w, world_h, obstacles, collision_mode)
