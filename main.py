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
# STAGE 2: Physics Tuning Constants
# =====================================================================
MAX_SPEED = 4.0               # Maximum speed limit of a fish (pixels/frame)
MAX_ACCELERATION = 0.15       # Maximum steering force acceleration (pixels/frame^2)
MAX_TURN_RATE = 0.08          # Maximum heading change per frame (radians/frame, 0.08 rad ≈ 4.6 deg)
SPEED_VARIATION_MIN = 0.8     # Each fish moves between 80% and 100% of MAX_SPEED
WANDER_INTERVAL = 45          # Frequency of selecting a new wander heading target (frames)
WANDER_FORCE = 0.5            # Maximum angle deflection for random wander choice (radians)

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

class Fish:
    def __init__(self, x, y, heading, speed_factor, color):
        self.x = x
        self.y = y
        self.speed_factor = speed_factor  # Per-fish speed variation coefficient
        self.color = color
        
        # Initialize velocity based on spawn heading and personal maximum speed
        personal_max_speed = self.speed_factor * MAX_SPEED
        self.vx = personal_max_speed * math.cos(heading)
        self.vy = personal_max_speed * math.sin(heading)
        
        # Temporary Wander state (Stage 2 placeholder)
        self.target_heading = heading
        self.wander_timer = random.randint(0, WANDER_INTERVAL)
        
        # Dimensions of the triangle representing the fish
        self.length = 14
        self.width = 8

    @property
    def heading(self):
        """
        Derives heading angle in radians from the velocity vector.
        Heading is not stored independently.
        """
        return math.atan2(self.vy, self.vx)

    def update(self):
        """
        Updates fish position and velocity using steering physics and turn-rate limits.
        """
        # =====================================================================
        # TEMPORARY: Wander behavior to exercise physics (pre-sensor, pre-neural)
        # To be replaced with sensor/neural inputs in Stage 3 / Stage 5
        # =====================================================================
        self.wander_timer -= 1
        if self.wander_timer <= 0:
            # Pick a randomized new interval to break up synchronization
            self.wander_timer = random.randint(int(WANDER_INTERVAL * 0.7), int(WANDER_INTERVAL * 1.3))
            # Adjust the target heading with a small random deflection
            self.target_heading = self.heading + random.uniform(-WANDER_FORCE, WANDER_FORCE)

        # =====================================================================
        # STAGE 3 HOOK: Sensors & Neighbors
        # - Query nearby neighbor fish within sensor range.
        # - Calculate alignment, cohesion, and separation steering force vectors.
        # - Replace the temporary wander heading logic with flocking forces.
        # =====================================================================
        
        # =====================================================================
        # STAGE 4 HOOK: Predator Avoidance
        # - Detect predator proximity and add a strong fleeing acceleration.
        # =====================================================================

        # Calculate steering force (acceleration vector) towards the target heading
        personal_max_speed = self.speed_factor * MAX_SPEED
        desired_vx = personal_max_speed * math.cos(self.target_heading)
        desired_vy = personal_max_speed * math.sin(self.target_heading)
        
        # Steering force = Desired Velocity - Current Velocity
        ax = desired_vx - self.vx
        ay = desired_vy - self.vy
        
        # Clamp acceleration magnitude to MAX_ACCELERATION
        accel_mag = math.hypot(ax, ay)
        if accel_mag > MAX_ACCELERATION:
            ax = (ax / accel_mag) * MAX_ACCELERATION
            ay = (ay / accel_mag) * MAX_ACCELERATION

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
            
        # Clamp the heading rotation per frame to MAX_TURN_RATE
        heading_diff = temp_heading - old_heading
        # Normalize angle difference to [-pi, pi] to handle wrap-around correctly
        heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi
        
        if abs(heading_diff) > MAX_TURN_RATE:
            heading_diff = math.copysign(MAX_TURN_RATE, heading_diff)
            actual_heading = old_heading + heading_diff
        else:
            actual_heading = temp_heading
            
        # Clamp velocity magnitude to personal maximum speed
        if speed > personal_max_speed:
            speed = personal_max_speed
            
        # Re-project velocity vector incorporating speed limit and turn-rate limit
        self.vx = speed * math.cos(actual_heading)
        self.vy = speed * math.sin(actual_heading)
        
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Keep fish inside screen boundaries
        self.handle_boundaries()

    def handle_boundaries(self):
        """
        Handles screen edge boundaries by wrapping position.
        Velocity and heading are preserved across edge wraps.
        """
        self.x = self.x % WINDOW_WIDTH
        self.y = self.y % WINDOW_HEIGHT

    def draw(self, surface):
        """
        Renders the fish as an oriented isosceles triangle.
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
        
        # Draw the fish polygon
        pygame.draw.polygon(surface, self.color, [
            (tip_x, tip_y),
            (left_x, left_y),
            (right_x, right_y)
        ])


def main():
    # Initialize Pygame
    pygame.init()
    pygame.font.init()
    
    # Setup window and clock
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("NeuroSwarm - Stage 2 Physics Simulation")
    clock = pygame.time.Clock()
    
    # Load default font for FPS counter
    font = pygame.font.SysFont(None, 24)
    
    # Spawn 100 fish with random positions, initial headings, speed factors, and silver/blue colors
    fish_school = []
    for _ in range(NUM_FISH):
        x = random.uniform(0, WINDOW_WIDTH)
        y = random.uniform(0, WINDOW_HEIGHT)
        heading = random.uniform(0, 2 * math.pi)
        speed_factor = random.uniform(SPEED_VARIATION_MIN, 1.0)
        color = generate_fish_color()
        fish_school.append(Fish(x, y, heading, speed_factor, color))
        
    running = True
    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    
        # 2. Update Physics & Logic (O(N) update loop)
        for fish in fish_school:
            fish.update()
            
        # 3. Rendering
        # Clear screen with dark background
        screen.fill(BG_COLOR)
        
        # Draw the school of fish
        for fish in fish_school:
            fish.draw(screen)
            
        # Draw FPS counter in top-left
        fps_val = clock.get_fps()
        fps_text = font.render(f"FPS: {int(fps_val)}", True, FPS_COLOR)
        screen.blit(fps_text, (10, 10))
        
        # Flip display and tick clock
        pygame.display.flip()
        clock.tick(FPS)
        
    # Clean cleanup and exit
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
