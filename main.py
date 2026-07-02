import pygame
import random
import math
import sys

# Simulation Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 60
NUM_FISH = 100
SPEED = 3.0  # Constant movement speed (pixels per frame)

# Color Constants
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
    def __init__(self, x, y, heading, color):
        self.x = x
        self.y = y
        self.heading = heading  # Angle in radians
        self.color = color
        
        # Dimensions of the triangle representing the fish
        self.length = 14
        self.width = 8

    def update(self):
        """
        Updates the position of the fish based on heading and constant speed.
        """
        # =====================================================================
        # STAGE 2 HOOK: Physics & Acceleration
        # - Implement velocity, acceleration, friction, and steering forces.
        # - Rather than updating position directly with a constant speed, we
        #   will apply force vectors, update velocity, and cap max speed/turn rate.
        # =====================================================================
        
        # =====================================================================
        # STAGE 3 HOOK: Sensors & Neighbors
        # - Fish will query nearby neighbors within a sensor range.
        # - Behaviors (Alignment, Cohesion, Separation) will alter heading/forces.
        # =====================================================================
        
        # =====================================================================
        # STAGE 4 HOOK: Predator Avoidance
        # - Fish will detect predator proximity and steer away rapidly.
        # =====================================================================

        # Direct constant-speed translation in the direction of the heading angle
        vx = SPEED * math.cos(self.heading)
        vy = SPEED * math.sin(self.heading)
        
        self.x += vx
        self.y += vy
        
        # Keep fish inside screen boundaries
        self.handle_boundaries()

    def handle_boundaries(self):
        """
        Handles screen edge boundaries.
        Currently wraps fish to the opposite side of the screen.
        """
        # Wrapping logic (avoids clustering at edges)
        self.x = self.x % WINDOW_WIDTH
        self.y = self.y % WINDOW_HEIGHT

        # NOTE: Structured to make bouncing easily swappable if desired later.
        # To change to bouncing, uncomment the block below and comment the wrapping:
        """
        if self.x < 0:
            self.x = 0
            self.heading = math.pi - self.heading
        elif self.x > WINDOW_WIDTH:
            self.x = WINDOW_WIDTH
            self.heading = math.pi - self.heading
            
        if self.y < 0:
            self.y = 0
            self.heading = -self.heading
        elif self.y > WINDOW_HEIGHT:
            self.y = WINDOW_HEIGHT
            self.heading = -self.heading
        """

    def draw(self, surface):
        """
        Renders the fish as an oriented isosceles triangle.
        """
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)
        
        # Point 1: Nose (pointing along the heading vector)
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
    pygame.display.set_caption("NeuroSwarm - Stage 1 Simulation Foundation")
    clock = pygame.time.Clock()
    
    # Load default font for FPS counter
    font = pygame.font.SysFont(None, 24)
    
    # Spawn 100 fish with random positions, initial headings, and silver/blue colors
    fish_school = []
    for _ in range(NUM_FISH):
        x = random.uniform(0, WINDOW_WIDTH)
        y = random.uniform(0, WINDOW_HEIGHT)
        heading = random.uniform(0, 2 * math.pi)
        color = generate_fish_color()
        fish_school.append(Fish(x, y, heading, color))
        
    running = True
    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    
        # 2. Physics & Logic Update (O(N) update loop)
        # =====================================================================
        # STAGE 4 HOOK: Predator Logic
        # - Update predator agent(s) positions and state here.
        # =====================================================================
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
