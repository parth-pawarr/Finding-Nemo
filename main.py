import pygame
import random
import math
import sys

# Simulation Window Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 60
NUM_FISH = 100

# =====================================================================
# STAGE 2: Physics Tuning Constants (Fish)
# =====================================================================
MAX_SPEED = 4.0               # Maximum speed limit of a fish (pixels/frame)
MAX_ACCELERATION = 0.15       # Maximum steering force acceleration (pixels/frame^2)
MAX_TURN_RATE = 0.08          # Maximum heading change per frame (radians/frame)
SPEED_VARIATION_MIN = 0.8     # Each fish moves between 80% and 100% of MAX_SPEED
WANDER_INTERVAL = 45          # Frequency of selecting a new wander heading target (frames)
WANDER_FORCE = 0.5            # Maximum angle deflection for random wander choice (radians)

# =====================================================================
# STAGE 3: Sensor Tuning Constants (Fish)
# =====================================================================
VISION_RADIUS = 120           # Sensor range to see neighbor fish (pixels)

# =====================================================================
# STAGE 4: Predator Tuning Constants
# =====================================================================
PREDATOR_MAX_SPEED = 4.8      # Predator max speed (faster than prey)
PREDATOR_MAX_ACCELERATION = 0.20 # Acceleration rate for predator lunging
PREDATOR_MAX_TURN_RATE = 0.05 # Predator max turn rate (wider/heavier arcs)
PREDATOR_VISION_RADIUS = 250  # Predator vision radius (can see much further)
PREDATOR_CATCH_RADIUS = 15    # Collision circle radius to catch/eat a fish

# Rendering Constants
BG_COLOR = (10, 15, 30)       # Deep navy/black underwater background
FPS_COLOR = (200, 200, 200)   # Light grey text for FPS counter

def generate_fish_color():
    """
    Generates a color within a subtle silver-blue range.
    Produces individual variations so the school does not look like a flat blob.
    """
    r = random.randint(100, 180)  # Metallic silver base
    g = random.randint(140, 200)  # Hint of soft green/teal
    b = random.randint(200, 255)  # Dominant blue component
    return (r, g, b)


class Agent:
    """
    Base Agent class encapsulating position, velocity, heading, and 
    common physics-based steering and boundary wrap-around behavior.
    """
    def __init__(self, x, y, heading, speed_factor, max_speed, max_accel, max_turn_rate, color, length, width):
        self.x = x
        self.y = y
        self.speed_factor = speed_factor
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_turn_rate = max_turn_rate
        self.color = color
        self.length = length
        self.width = width
        
        # Initialize velocity vector based on spawn heading and personal max speed
        personal_max_speed = self.speed_factor * self.max_speed
        self.vx = personal_max_speed * math.cos(heading)
        self.vy = personal_max_speed * math.sin(heading)
        
        # Wander states (used as defaults when no external steer is active)
        self.target_heading = heading
        self.wander_timer = random.randint(0, WANDER_INTERVAL)

    @property
    def heading(self):
        """
        Derives heading angle in radians from the velocity vector.
        Heading is not stored independently.
        """
        return math.atan2(self.vy, self.vx)

    def move(self, ax, ay):
        """
        Updates velocity vector and position using acceleration, capping top-speed
        and clamping heading change rate.
        """
        # Capture old heading for turn-rate clamping
        old_heading = self.heading

        # Apply acceleration to velocity vector
        new_vx = self.vx + ax
        new_vy = self.vy + ay
        
        # Calculate heading of the new proposed velocity vector
        speed = math.hypot(new_vx, new_vy)
        if speed > 0.0001:
            temp_heading = math.atan2(new_vy, new_vx)
        else:
            temp_heading = old_heading
            
        # Clamp the heading rotation per frame to max_turn_rate
        heading_diff = temp_heading - old_heading
        # Normalize angle difference to [-pi, pi]
        heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi
        
        if abs(heading_diff) > self.max_turn_rate:
            heading_diff = math.copysign(self.max_turn_rate, heading_diff)
            actual_heading = old_heading + heading_diff
        else:
            actual_heading = temp_heading
            
        # Clamp velocity magnitude to personal maximum speed
        personal_max_speed = self.speed_factor * self.max_speed
        if speed > personal_max_speed:
            speed = personal_max_speed
            
        # Re-project velocity vector incorporating speed limit and turn-rate limit
        self.vx = speed * math.cos(actual_heading)
        self.vy = speed * math.sin(actual_heading)
        
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Keep agent inside screen boundaries
        self.handle_boundaries()

    def handle_boundaries(self):
        """
        Wraps coordinate positions around screen edges.
        Velocity/heading vectors are preserved.
        """
        self.x = self.x % WINDOW_WIDTH
        self.y = self.y % WINDOW_HEIGHT

    def draw(self, surface):
        """
        Renders the agent as an oriented isosceles triangle.
        """
        h = self.heading
        cos_h = math.cos(h)
        sin_h = math.sin(h)
        
        # Point 1: Nose (pointing along heading vector)
        tip_x = self.x + self.length * cos_h
        tip_y = self.y + self.length * sin_h
        
        # Point 2: Left tail corner (rotated back and offset left)
        left_x = self.x - (self.length / 2) * cos_h - (self.width / 2) * sin_h
        left_y = self.y - (self.length / 2) * sin_h + (self.width / 2) * cos_h
        
        # Point 3: Right tail corner (rotated back and offset right)
        right_x = self.x - (self.length / 2) * cos_h + (self.width / 2) * sin_h
        right_y = self.y - (self.length / 2) * sin_h - (self.width / 2) * cos_h
        
        # Draw the agent polygon
        pygame.draw.polygon(surface, self.color, [
            (tip_x, tip_y),
            (left_x, left_y),
            (right_x, right_y)
        ])


class Fish(Agent):
    """
    Prey Fish agent. Contains sensors, wandering behavior, and a flee reflex.
    """
    def __init__(self, x, y, heading, speed_factor, color):
        super().__init__(
            x=x, y=y, heading=heading, speed_factor=speed_factor,
            max_speed=MAX_SPEED, max_accel=MAX_ACCELERATION, max_turn_rate=MAX_TURN_RATE,
            color=color, length=14, width=8
        )
        
        # Stage 3 Sensor Fields
        self.sensor_nearest_dist = float('inf')   # Distance to nearest neighbor
        self.sensor_nearest_angle = 0.0           # Relative angle to nearest neighbor (radians)
        self.sensor_wall_dist = 0.0               # Distance to edge along heading vector
        
        # Stage 4 Sensor Fields
        self.sensor_predator_dist = float('inf')  # Distance to predator
        self.sensor_predator_angle = 0.0          # Relative angle to predator (radians)
        
        # State variables
        self.is_fleeing = False                   # Flag indicating flee reflex override
        self.nearest_neighbor = None              # Reference to closest fish object
        self.nearest_neighbor_dist = float('inf')  # True unclamped toroidal distance to nearest fish

    def sense(self, all_fish, predator):
        """
        Updates fish sensors by scanning neighbors, screen boundaries, and the predator.
        """
        nearest_fish = None
        min_dist = float('inf')
        best_dx = 0.0
        best_dy = 0.0
        
        # O(N) neighbor search per fish -> O(N^2) total simulation pass
        for other in all_fish:
            if other is self:
                continue
                
            # Toroidal coordinate differences to handle edge wrapping
            dx = other.x - self.x
            if dx > WINDOW_WIDTH / 2: dx -= WINDOW_WIDTH
            elif dx < -WINDOW_WIDTH / 2: dx += WINDOW_WIDTH
                
            dy = other.y - self.y
            if dy > WINDOW_HEIGHT / 2: dy -= WINDOW_HEIGHT
            elif dy < -WINDOW_HEIGHT / 2: dy += WINDOW_HEIGHT
                
            dist = math.hypot(dx, dy)
            if dist < min_dist:
                min_dist = dist
                nearest_fish = other
                best_dx = dx
                best_dy = dy
                
        # Store closest neighbor details for debug drawing
        self.nearest_neighbor = nearest_fish
        self.nearest_neighbor_dist = min_dist
        
        # Populate nearest neighbor sensors (Stage 5 network inputs)
        if nearest_fish is not None and min_dist <= VISION_RADIUS:
            self.sensor_nearest_dist = min_dist
            abs_angle = math.atan2(best_dy, best_dx)
            rel_angle = abs_angle - self.heading
            self.sensor_nearest_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
        else:
            self.sensor_nearest_dist = float('inf')
            self.sensor_nearest_angle = 0.0
            
        # Calculate wall distance along the heading vector
        self.sensor_wall_dist = self.get_distance_to_edge()
        
        # =====================================================================
        # STAGE 4: Predator Sensing & Flight Reflex
        # =====================================================================
        p_dx = predator.x - self.x
        if p_dx > WINDOW_WIDTH / 2: p_dx -= WINDOW_WIDTH
        elif p_dx < -WINDOW_WIDTH / 2: p_dx += WINDOW_WIDTH
        
        p_dy = predator.y - self.y
        if p_dy > WINDOW_HEIGHT / 2: p_dy -= WINDOW_HEIGHT
        elif p_dy < -WINDOW_HEIGHT / 2: p_dy += WINDOW_HEIGHT
        
        dist_p = math.hypot(p_dx, p_dy)
        self.sensor_predator_dist = dist_p
        
        if dist_p > 0.0001:
            abs_angle_p = math.atan2(p_dy, p_dx)
            rel_angle_p = abs_angle_p - self.heading
            self.sensor_predator_angle = (rel_angle_p + math.pi) % (2 * math.pi) - math.pi
        else:
            self.sensor_predator_angle = 0.0
            
        # Flight trigger: predator is within vision and is closest sensed thing
        # TEMPORARY: This is rules-based flight, to be replaced by learned Neural Avoidance in Stage 5
        self.is_fleeing = False
        if dist_p <= VISION_RADIUS and dist_p <= self.nearest_neighbor_dist:
            self.is_fleeing = True
            # Set target heading directly away from predator
            self.target_heading = math.atan2(p_dy, p_dx) + math.pi

    def update(self):
        """
        Updates fish velocity and position based on wander or flee steering goals.
        """
        if not self.is_fleeing:
            # Simple wander logic
            self.wander_timer -= 1
            if self.wander_timer <= 0:
                self.wander_timer = random.randint(int(WANDER_INTERVAL * 0.7), int(WANDER_INTERVAL * 1.3))
                self.target_heading = self.heading + random.uniform(-WANDER_FORCE, WANDER_FORCE)
        else:
            # Fleeing target heading was calculated in sense()
            pass

        # Calculate steering force (acceleration vector) towards target heading
        personal_max_speed = self.speed_factor * MAX_SPEED
        desired_vx = personal_max_speed * math.cos(self.target_heading)
        desired_vy = personal_max_speed * math.sin(self.target_heading)
        
        # Steering force = Desired Velocity - Current Velocity
        ax = desired_vx - self.vx
        ay = desired_vy - self.vy
        
        # Clamp acceleration magnitude to MAX_ACCELERATION
        accel_mag = math.hypot(ax, ay)
        if accel_mag > self.max_accel:
            ax = (ax / accel_mag) * self.max_accel
            ay = (ay / accel_mag) * self.max_accel
            
        # Execute base agent physics movement
        self.move(ax, ay)

    def get_distance_to_edge(self):
        """
        Computes the distance to the screen edge along the fish's heading vector.
        """
        h = self.heading
        cos_h = math.cos(h)
        sin_h = math.sin(h)
        
        dist_x = float('inf')
        if cos_h > 0:
            dist_x = (WINDOW_WIDTH - self.x) / cos_h
        elif cos_h < 0:
            dist_x = -self.x / cos_h
            
        dist_y = float('inf')
        if sin_h > 0:
            dist_y = (WINDOW_HEIGHT - self.y) / sin_h
        elif sin_h < 0:
            dist_y = -self.y / sin_h
            
        return min(dist_x, dist_y)


class Predator(Agent):
    """
    Predator agent. Inherits Agent physics. Actively chases the closest fish.
    """
    def __init__(self, x, y, heading):
        super().__init__(
            x=x, y=y, heading=heading, speed_factor=1.0,  # Predator does not use speed variations
            max_speed=PREDATOR_MAX_SPEED, max_accel=PREDATOR_MAX_ACCELERATION, max_turn_rate=PREDATOR_MAX_TURN_RATE,
            color=(231, 76, 60), length=22, width=12  # Larger size, bright crimson/red
        )
        self.target_fish = None                   # Current prey target reference
        self.target_fish_dist = float('inf')       # True toroidal distance to current target

    def hunt(self, fish_school):
        """
        Scans prey fish, tracks the closest one, and steers to intercept it.
        Wanders if no prey is within PREDATOR_VISION_RADIUS.
        """
        nearest_fish = None
        min_dist = float('inf')
        best_dx = 0.0
        best_dy = 0.0
        
        # Scan all fish to identify the nearest prey
        for fish in fish_school:
            dx = fish.x - self.x
            if dx > WINDOW_WIDTH / 2: dx -= WINDOW_WIDTH
            elif dx < -WINDOW_WIDTH / 2: dx += WINDOW_WIDTH
                
            dy = fish.y - self.y
            if dy > WINDOW_HEIGHT / 2: dy -= WINDOW_HEIGHT
            elif dy < -WINDOW_HEIGHT / 2: dy += WINDOW_HEIGHT
                
            dist = math.hypot(dx, dy)
            if dist < min_dist:
                min_dist = dist
                nearest_fish = fish
                best_dx = dx
                best_dy = dy
                
        # Store tracking parameters for logic and debugging
        self.target_fish = nearest_fish
        self.target_fish_dist = min_dist
        
        if nearest_fish is not None and min_dist <= PREDATOR_VISION_RADIUS:
            # Active Chase: steer directly towards toroidal target offset
            self.target_heading = math.atan2(best_dy, best_dx)
        else:
            # Wander: default wandering behavior when nothing is seen
            self.wander_timer -= 1
            if self.wander_timer <= 0:
                self.wander_timer = random.randint(int(WANDER_INTERVAL * 0.7), int(WANDER_INTERVAL * 1.3))
                self.target_heading = self.heading + random.uniform(-WANDER_FORCE, WANDER_FORCE)

        # Steering force towards target heading
        desired_vx = self.max_speed * math.cos(self.target_heading)
        desired_vy = self.max_speed * math.sin(self.target_heading)
        
        ax = desired_vx - self.vx
        ay = desired_vy - self.vy
        
        # Clamp acceleration
        accel_mag = math.hypot(ax, ay)
        if accel_mag > self.max_accel:
            ax = (ax / accel_mag) * self.max_accel
            ay = (ay / accel_mag) * self.max_accel
            
        # Execute base agent physics movement
        self.move(ax, ay)


def main():
    # Initialize Pygame
    pygame.init()
    pygame.font.init()
    
    # Setup window and clock
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("NeuroSwarm - Stage 4 Predator Simulation")
    clock = pygame.time.Clock()
    
    # Load default font for HUD rendering
    font = pygame.font.SysFont(None, 24)
    
    # Spawn predator in the center of the window
    predator = Predator(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2, random.uniform(0, 2 * math.pi))
    
    # Spawn 100 fish with random positions, initial headings, speed factors, and silver/blue colors
    fish_school = []
    for _ in range(NUM_FISH):
        x = random.uniform(0, WINDOW_WIDTH)
        y = random.uniform(0, WINDOW_HEIGHT)
        heading = random.uniform(0, 2 * math.pi)
        speed_factor = random.uniform(SPEED_VARIATION_MIN, 1.0)
        color = generate_fish_color()
        fish_school.append(Fish(x, y, heading, speed_factor, color))
        
    debug_mode = False
    running = True
    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_d:
                    debug_mode = not debug_mode
                    
        # 2. Update Sensors
        for fish in fish_school:
            fish.sense(fish_school, predator)
            
        predator.hunt(fish_school)
        
        # 3. Update Physics & Logic
        for fish in fish_school:
            fish.update()
            
        # 4. Catch Detection (collisions checked using toroidal boundaries)
        surviving_fish = []
        for fish in fish_school:
            dx = fish.x - predator.x
            if dx > WINDOW_WIDTH / 2: dx -= WINDOW_WIDTH
            elif dx < -WINDOW_WIDTH / 2: dx += WINDOW_WIDTH
            
            dy = fish.y - predator.y
            if dy > WINDOW_HEIGHT / 2: dy -= WINDOW_HEIGHT
            elif dy < -WINDOW_HEIGHT / 2: dy += WINDOW_HEIGHT
            
            dist = math.hypot(dx, dy)
            if dist > PREDATOR_CATCH_RADIUS:
                surviving_fish.append(fish)
        fish_school = surviving_fish
        
        # Statistics
        fish_remaining = len(fish_school)
        fish_eaten_count = NUM_FISH - fish_remaining
            
        # 5. Rendering
        # Clear screen with dark background
        screen.fill(BG_COLOR)
        
        # Draw school of fish
        for fish in fish_school:
            fish.draw(screen)
            
        # Draw predator
        predator.draw(screen)
        
        # Draw Debug Overlays
        if debug_mode:
            # selected fish debug (mouse closest)
            mouse_pos = pygame.mouse.get_pos()
            selected_fish = None
            min_mouse_dist = float('inf')
            for fish in fish_school:
                d = math.hypot(fish.x - mouse_pos[0], fish.y - mouse_pos[1])
                if d < min_mouse_dist:
                    min_mouse_dist = d
                    selected_fish = fish
                    
            if selected_fish is not None:
                # A. Draw vision radius circle
                pygame.draw.circle(screen, (80, 100, 120), (int(selected_fish.x), int(selected_fish.y)), VISION_RADIUS, 1)
                
                # B. Draw line to nearest neighbor (green if within range, gray if outside)
                if selected_fish.nearest_neighbor is not None:
                    is_in_range = selected_fish.nearest_neighbor_dist <= VISION_RADIUS
                    line_color = (46, 204, 113) if is_in_range else (127, 140, 141)
                    
                    pygame.draw.line(screen, line_color, 
                                     (int(selected_fish.x), int(selected_fish.y)), 
                                     (int(selected_fish.nearest_neighbor.x), int(selected_fish.nearest_neighbor.y)), 2)
                    pygame.draw.circle(screen, line_color, (int(selected_fish.nearest_neighbor.x), int(selected_fish.nearest_neighbor.y)), 5, 1)
                
                # C. Draw wall sensor ray along heading vector
                end_x = selected_fish.x + selected_fish.sensor_wall_dist * math.cos(selected_fish.heading)
                end_y = selected_fish.y + selected_fish.sensor_wall_dist * math.sin(selected_fish.heading)
                pygame.draw.line(screen, (230, 126, 34), (int(selected_fish.x), int(selected_fish.y)), (int(end_x), int(end_y)), 2)
                pygame.draw.circle(screen, (230, 126, 34), (int(end_x), int(end_y)), 4)
                
                # Highlight the selected fish
                pygame.draw.circle(screen, (241, 196, 15), (int(selected_fish.x), int(selected_fish.y)), 10, 1)
                
                # Render selected fish sensors HUD
                sensor_text_1 = font.render(f"Nearest Dist: {selected_fish.sensor_nearest_dist:.1f}", True, (200, 200, 200))
                sensor_text_2 = font.render(f"Nearest Angle: {selected_fish.sensor_nearest_angle:.2f} rad", True, (200, 200, 200))
                sensor_text_3 = font.render(f"Wall Dist: {selected_fish.sensor_wall_dist:.1f}", True, (200, 200, 200))
                sensor_text_4 = font.render(f"Predator Dist: {selected_fish.sensor_predator_dist:.1f}", True, (200, 200, 200))
                sensor_text_5 = font.render(f"Fleeing: {selected_fish.is_fleeing}", True, (231, 76, 60) if selected_fish.is_fleeing else (200, 200, 200))
                screen.blit(sensor_text_1, (10, WINDOW_HEIGHT - 110))
                screen.blit(sensor_text_2, (10, WINDOW_HEIGHT - 90))
                screen.blit(sensor_text_3, (10, WINDOW_HEIGHT - 70))
                screen.blit(sensor_text_4, (10, WINDOW_HEIGHT - 50))
                screen.blit(sensor_text_5, (10, WINDOW_HEIGHT - 30))
                
            # D. Predator Debug (always drawn when debug mode is enabled)
            # Draw predator vision circle (crimson red outline)
            pygame.draw.circle(screen, (150, 50, 50), (int(predator.x), int(predator.y)), PREDATOR_VISION_RADIUS, 1)
            
            # Draw line to targeted prey fish
            if predator.target_fish is not None:
                is_chasing = predator.target_fish_dist <= PREDATOR_VISION_RADIUS
                line_color = (231, 76, 60) if is_chasing else (127, 140, 141)
                pygame.draw.line(screen, line_color, 
                                 (int(predator.x), int(predator.y)), 
                                 (int(predator.target_fish.x), int(predator.target_fish.y)), 2)
                pygame.draw.circle(screen, line_color, (int(predator.target_fish.x), int(predator.target_fish.y)), 8, 1)

        # Draw Stats HUD in top-left
        fps_val = clock.get_fps()
        fps_text = font.render(f"FPS: {int(fps_val)}", True, FPS_COLOR)
        screen.blit(fps_text, (10, 10))
        
        count_text = font.render(f"Fish Remaining: {fish_remaining} | Eaten: {fish_eaten_count}", True, (231, 76, 60))
        screen.blit(count_text, (10, 30))
        
        if debug_mode:
            dbg_ind = font.render("DEBUG MODE ACTIVE", True, (241, 196, 15))
            screen.blit(dbg_ind, (10, 50))
            
        # Flip display and tick clock
        pygame.display.flip()
        clock.tick(FPS)
        
    # Clean cleanup and exit
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
