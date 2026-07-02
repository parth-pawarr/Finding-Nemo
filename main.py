import pygame
import random
import math
import sys
import numpy as np
import json
import csv
import time

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

# =====================================================================
# STAGE 6: Neuroevolution Tuning Constants
# =====================================================================
GENERATION_TIME_LIMIT_FRAMES = 3600  # Max time limit per generation (60s @ 60 FPS)
ELITE_PERCENT = 0.15                 # Top 15% of fish are preserved exactly (elitism)
TOURNAMENT_SIZE = 5                  # Tournament selection pool size
MUTATION_RATE = 0.08                 # Probability of mutating each weight (8%)
MUTATION_STRENGTH = 0.15             # Scale (std dev) of Gaussian mutation noise

# Fitness Weights
FITNESS_SURVIVAL_REWARD = 1.0        # Reward added to fitness every frame survived
FITNESS_SPEED_MULT = 0.5             # Reward scaled by speed (encourages active swimming)
FITNESS_WALL_PENALTY_THRESHOLD = 60.0 # Distance to wall under which penalty starts
FITNESS_WALL_PENALTY_MAX = 3.0       # Max penalty/frame subtracted when touching wall

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


class NeuralNetwork:
    """
    Lightweight Feed-Forward Neural Network brain for the agents.
    Uses basic matrix math with NumPy.
    Dimensions: 5 inputs -> 8 hidden neurons -> 2 outputs.
    """
    def __init__(self, input_size=5, hidden_size=8, output_size=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases randomly in range [-1.0, 1.0]
        self.w1 = np.random.uniform(-1.0, 1.0, (input_size, hidden_size))
        self.b1 = np.random.uniform(-1.0, 1.0, (1, hidden_size))
        
        self.w2 = np.random.uniform(-1.0, 1.0, (hidden_size, output_size))
        self.b2 = np.random.uniform(-1.0, 1.0, (1, output_size))

    def forward(self, inputs):
        """
        Runs inputs through the network. Returns a flat array of size 2.
        All layers use tanh activations (squashed output values to [-1.0, 1.0]).
        """
        x = np.array(inputs).reshape(1, -1)
        h = np.tanh(np.dot(x, self.w1) + self.b1)
        out = np.tanh(np.dot(h, self.w2) + self.b2)
        return out.flatten()

    def get_weights(self):
        """
        Flattens all weights and biases into a single 1D numpy array.
        This provides direct access for mutation and crossover in Stage 6.
        """
        return np.concatenate([
            self.w1.flatten(),
            self.b1.flatten(),
            self.w2.flatten(),
            self.b2.flatten()
        ])

    def set_weights(self, flat_weights):
        """
        Reconstructs matrices and bias vectors from a flat 1D array.
        Used for updating the brain with evolved weights.
        """
        idx = 0
        w1_size = self.input_size * self.hidden_size
        self.w1 = flat_weights[idx : idx + w1_size].reshape(self.input_size, self.hidden_size)
        idx += w1_size
        
        b1_size = self.hidden_size
        self.b1 = flat_weights[idx : idx + b1_size].reshape(1, self.hidden_size)
        idx += b1_size
        
        w2_size = self.hidden_size * self.output_size
        self.w2 = flat_weights[idx : idx + w2_size].reshape(self.hidden_size, self.output_size)
        idx += w2_size
        
        b2_size = self.output_size
        self.b2 = flat_weights[idx : idx + b2_size].reshape(1, self.output_size)


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
        
        # Wander states
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
        old_heading = self.heading

        new_vx = self.vx + ax
        new_vy = self.vy + ay
        
        speed = math.hypot(new_vx, new_vy)
        if speed > 0.0001:
            temp_heading = math.atan2(new_vy, new_vx)
        else:
            temp_heading = old_heading
            
        heading_diff = temp_heading - old_heading
        heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi
        
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
        
        self.x += self.vx
        self.y += self.vy
        
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
        
        # Point 1: Nose
        tip_x = self.x + self.length * cos_h
        tip_y = self.y + self.length * sin_h
        
        # Point 2: Left tail corner
        left_x = self.x - (self.length / 2) * cos_h - (self.width / 2) * sin_h
        left_y = self.y - (self.length / 2) * sin_h + (self.width / 2) * cos_h
        
        # Point 3: Right tail corner
        right_x = self.x - (self.length / 2) * cos_h + (self.width / 2) * sin_h
        right_y = self.y - (self.length / 2) * sin_h - (self.width / 2) * cos_h
        
        pygame.draw.polygon(surface, self.color, [
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
            max_speed=MAX_SPEED, max_accel=MAX_ACCELERATION, max_turn_rate=MAX_TURN_RATE,
            color=color, length=14, width=8
        )
        
        # Brain Network (5 inputs -> 8 hidden -> 2 outputs)
        self.brain = NeuralNetwork(input_size=5, hidden_size=8, output_size=2)
        
        # Fitness Score
        self.fitness = 0.0
        
        # Stage 7 loaded champion flag (for visual marking)
        self.is_champion = False
        
        # Stage 3 Sensor Fields
        self.sensor_nearest_dist = float('inf')
        self.sensor_nearest_angle = 0.0
        self.sensor_wall_dist = 0.0
        
        # Stage 4 Sensor Fields
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

    def sense(self, all_fish, predator):
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
                
        self.nearest_neighbor = nearest_fish
        self.nearest_neighbor_dist = min_dist
        
        # Populate nearest neighbor sensors
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
        
        # Calculate predator distance/angle sensors
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

    def think(self):
        """
        Feeds normalized sensors to the brain, runs forward inference,
        and translates outputs to steering acceleration forces.
        """
        norm_dist = min(self.sensor_nearest_dist / VISION_RADIUS, 1.0)
        norm_angle = self.sensor_nearest_angle / math.pi
        norm_wall = min(self.sensor_wall_dist / 1200.0, 1.0)
        norm_pred_dist = min(self.sensor_predator_dist / VISION_RADIUS, 1.0)
        norm_pred_angle = self.sensor_predator_angle / math.pi
        
        inputs = [norm_dist, norm_angle, norm_wall, norm_pred_dist, norm_pred_angle]
        
        # Forward pass
        outputs = self.brain.forward(inputs)
        self.last_steer_output = outputs[0]
        self.last_accel_output = outputs[1]
        
        # Desired heading
        desired_heading = self.heading + self.last_steer_output * self.max_turn_rate
        
        # Desired speed
        personal_max_speed = self.speed_factor * MAX_SPEED
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

    def update(self):
        """
        Moves the fish based on neural decisions and updates its fitness score.
        """
        self.move(self.ax_input, self.ay_input)
        
        # Fitness Accumulation
        self.fitness += FITNESS_SURVIVAL_REWARD
        
        current_speed = math.hypot(self.vx, self.vy)
        self.fitness += current_speed * FITNESS_SPEED_MULT
        
        if self.sensor_wall_dist < FITNESS_WALL_PENALTY_THRESHOLD:
            wall_ratio = self.sensor_wall_dist / FITNESS_WALL_PENALTY_THRESHOLD
            penalty = (1.0 - wall_ratio) * FITNESS_WALL_PENALTY_MAX
            self.fitness -= penalty
            
        if self.fitness < 0.0:
            self.fitness = 0.0

    def draw(self, surface):
        """
        Extends draw to render a special highlight if the fish is a loaded champion.
        """
        super().draw(surface)
        if self.is_champion:
            # Draw visual highlight ring around the loaded champion
            pygame.draw.circle(surface, (255, 215, 0), (int(self.x), int(self.y)), 12, 1)

    def get_distance_to_edge(self):
        """
        Computes distance to screen boundary along heading.
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
    Predator agent. Chases the closest prey fish using rule-based intercept.
    """
    def __init__(self, x, y, heading):
        super().__init__(
            x=x, y=y, heading=heading, speed_factor=1.0,
            max_speed=PREDATOR_MAX_SPEED, max_accel=PREDATOR_MAX_ACCELERATION, max_turn_rate=PREDATOR_MAX_TURN_RATE,
            color=(231, 76, 60), length=22, width=12
        )
        self.target_fish = None
        self.target_fish_dist = float('inf')

    def hunt(self, fish_school):
        """
        Steers toward closest fish, otherwise wanders.
        """
        nearest_fish = None
        min_dist = float('inf')
        best_dx = 0.0
        best_dy = 0.0
        
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
                
        self.target_fish = nearest_fish
        self.target_fish_dist = min_dist
        
        if nearest_fish is not None and min_dist <= PREDATOR_VISION_RADIUS:
            self.target_heading = math.atan2(best_dy, best_dx)
        else:
            self.wander_timer -= 1
            if self.wander_timer <= 0:
                self.wander_timer = random.randint(int(WANDER_INTERVAL * 0.7), int(WANDER_INTERVAL * 1.3))
                self.target_heading = self.heading + random.uniform(-WANDER_FORCE, WANDER_FORCE)

        # Steering
        desired_vx = self.max_speed * math.cos(self.target_heading)
        desired_vy = self.max_speed * math.sin(self.target_heading)
        
        ax = desired_vx - self.vx
        ay = desired_vy - self.vy
        
        accel_mag = math.hypot(ax, ay)
        if accel_mag > self.max_accel:
            ax = (ax / accel_mag) * self.max_accel
            ay = (ay / accel_mag) * self.max_accel
            
        self.move(ax, ay)


# =====================================================================
# STAGE 6: Neuroevolution Core Operators
# =====================================================================
def tournament_selection(population):
    """
    Selects TOURNAMENT_SIZE random fish candidates and returns the best.
    """
    candidates = random.sample(population, TOURNAMENT_SIZE)
    return max(candidates, key=lambda f: f.fitness)

def crossover(parent_a, parent_b):
    """
    Merges flat weight representations of two parents using uniform crossover.
    """
    w_a = parent_a.brain.get_weights()
    w_b = parent_b.brain.get_weights()
    child_w = np.copy(w_a)
    mask = np.random.rand(len(w_a)) < 0.5
    child_w[mask] = w_b[mask]
    return child_w

def mutate(weights):
    """
    Adds Gaussian random noise to a fraction of weights.
    """
    mutated_w = np.copy(weights)
    mask = np.random.rand(len(weights)) < MUTATION_RATE
    noise = np.random.normal(0, MUTATION_STRENGTH, size=np.sum(mask))
    mutated_w[mask] += noise
    return mutated_w

def evolve_population(entire_population, loaded_weights=None):
    """
    Generates 100 new fish by keeping elites and breeding/mutating others.
    Optionally injects a loaded champion genome.
    """
    # Sort by fitness descending
    sorted_pop = sorted(entire_population, key=lambda f: f.fitness, reverse=True)
    
    # Calculate elites
    elite_count = int(NUM_FISH * ELITE_PERCENT)
    
    new_genomes = []
    
    # Inject loaded champion weights if available
    champion_injected = False
    if loaded_weights is not None:
        new_genomes.append(loaded_weights)
        champion_injected = True
        
    # Preserve remaining elites
    start_idx = 1 if champion_injected else 0
    for i in range(start_idx, elite_count):
        new_genomes.append(sorted_pop[i].brain.get_weights())
        
    # Breed the rest
    while len(new_genomes) < NUM_FISH:
        parent_a = tournament_selection(sorted_pop)
        parent_b = tournament_selection(sorted_pop)
        
        # Crossover
        child_w = crossover(parent_a, parent_b)
        
        # Mutation
        child_w = mutate(child_w)
        
        new_genomes.append(child_w)
        
    # Spawn 100 new fish
    new_fish_school = []
    for i in range(NUM_FISH):
        x = random.uniform(0, WINDOW_WIDTH)
        y = random.uniform(0, WINDOW_HEIGHT)
        heading = random.uniform(0, 2 * math.pi)
        speed_factor = random.uniform(SPEED_VARIATION_MIN, 1.0)
        
        # Visual marking for injected champion
        if i == 0 and champion_injected:
            color = (255, 215, 0)  # Distinct Gold
            fish = Fish(x, y, heading, speed_factor, color)
            fish.brain.set_weights(new_genomes[i])
            fish.is_champion = True
        else:
            color = generate_fish_color()
            fish = Fish(x, y, heading, speed_factor, color)
            fish.brain.set_weights(new_genomes[i])
            
        new_fish_school.append(fish)
        
    return new_fish_school


# =====================================================================
# STAGE 7: Analytics, HUD Panel, and Portability
# =====================================================================
def save_genome(weights, fitness, generation, filepath="best_genome.json"):
    """
    Serializes a flat weight array and training metadata to a JSON file.
    """
    try:
        data = {
            "weights": weights.tolist(), # Convert numpy array to list
            "fitness": float(fitness),
            "generation": int(generation),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[SUCCESS] Saved champion genome to {filepath} (Fitness: {fitness:.2f}, Gen: {generation})")
    except Exception as e:
        print(f"[ERROR] Failed to save genome: {e}")

def load_genome(filepath="best_genome.json"):
    """
    Loads weights from a saved JSON genome file.
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        weights = np.array(data["weights"])
        print(f"[SUCCESS] Loaded genome from {filepath} (Fitness: {data['fitness']:.2f}, Saved Gen: {data['generation']})")
        return weights
    except Exception as e:
        print(f"[ERROR] Failed to load genome from {filepath}: {e}")
        return None

def export_stats_csv(stats_history, filepath="generation_history.csv"):
    """
    Exports the recorded generation metrics history to a CSV file.
    """
    try:
        if not stats_history:
            print("[WARNING] No generation statistics to export yet.")
            return
        keys = stats_history[0].keys()
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(stats_history)
        print(f"[SUCCESS] Exported stats history CSV to {filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to export stats CSV: {e}")

def draw_fitness_graph(surface, stats_history):
    """
    Draws a fitness progression line graph (best vs avg) to a cached Pygame surface.
    Updates only on generation changes.
    """
    surface.fill((20, 25, 40, 220))  # Semi-transparent navy background
    width, height = surface.get_size()
    padding = 22
    plot_w = width - 2 * padding
    plot_h = height - 2 * padding
    
    # Outer border
    pygame.draw.rect(surface, (100, 110, 130), (0, 0, width, height), 2)
    
    font = pygame.font.SysFont(None, 14)
    
    if not stats_history:
        # Placeholder text
        placeholder = font.render("Waiting for Gen 1 completion...", True, (150, 160, 180))
        surface.blit(placeholder, (padding + 10, height / 2 - 8))
        return

    # Auto-scale y-axis based on historical data maxima
    max_fit = max(max(g["best_fitness"], g["avg_fitness"]) for g in stats_history)
    max_val = max(1.0, max_fit * 1.1)  # Leave 10% headroom
    
    num_gens = len(stats_history)
    
    # Draw horizontal grid lines and y-axis labels
    for i in range(4):
        ratio = i / 3.0
        y_pos = padding + plot_h - (ratio * plot_h)
        pygame.draw.line(surface, (50, 60, 80), (padding, y_pos), (padding + plot_w, y_pos), 1)
        
        lbl_val = ratio * max_val
        lbl = font.render(f"{int(lbl_val)}", True, (120, 130, 150))
        surface.blit(lbl, (2, y_pos - 6))
        
    # Map data to coordinate coordinates
    best_coords = []
    avg_coords = []
    
    for idx, gen in enumerate(stats_history):
        x_ratio = idx / (num_gens - 1) if num_gens > 1 else 0.5
        x_pos = padding + x_ratio * plot_w
        
        best_y = padding + plot_h - (gen["best_fitness"] / max_val) * plot_h
        avg_y = padding + plot_h - (gen["avg_fitness"] / max_val) * plot_h
        
        best_coords.append((x_pos, best_y))
        avg_coords.append((x_pos, avg_y))
        
    # Draw Average line (cyan)
    if len(avg_coords) > 1:
        pygame.draw.lines(surface, (52, 152, 219), False, avg_coords, 2)
    elif len(avg_coords) == 1:
        pygame.draw.circle(surface, (52, 152, 219), (int(avg_coords[0][0]), int(avg_coords[0][1])), 3)
        
    # Draw Best line (yellow)
    if len(best_coords) > 1:
        pygame.draw.lines(surface, (241, 196, 15), False, best_coords, 2)
    elif len(best_coords) == 1:
        pygame.draw.circle(surface, (241, 196, 15), (int(best_coords[0][0]), int(best_coords[0][1])), 3)
        
    # Title and Legend
    title_font = pygame.font.SysFont(None, 15, bold=True)
    title = title_font.render("Fitness Progression", True, (200, 200, 200))
    surface.blit(title, (padding, 4))
    
    best_leg = font.render("Yellow: Best", True, (241, 196, 15))
    avg_leg = font.render("Cyan: Avg", True, (52, 152, 219))
    surface.blit(best_leg, (width - 150, 4))
    surface.blit(avg_leg, (width - 75, 4))
    
    # X-axis Labels
    start_lbl = font.render("Gen 1", True, (120, 130, 150))
    end_lbl = font.render(f"Gen {num_gens}", True, (120, 130, 150))
    surface.blit(start_lbl, (padding, height - 16))
    surface.blit(end_lbl, (width - padding - 40, height - 16))

def draw_hud(screen, font, fps, gen, time_rem, fish_rem, best_fit_current, all_time_best):
    """
    Renders a unified status details HUD in the top-left corner.
    """
    panel_w = 260
    panel_h = 175
    
    hud_surface = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    hud_surface.fill((20, 25, 40, 220))  # Semi-transparent navy panel
    pygame.draw.rect(hud_surface, (100, 110, 130), (0, 0, panel_w, panel_h), 2)
    
    lines = [
        ("FPS:", f"{int(fps)}", (200, 200, 200)),
        ("Generation:", f"{gen}", (200, 200, 200)),
        ("Time Remaining:", f"{time_rem:.1f}s", (200, 200, 200)),
        ("Fish Alive:", f"{fish_rem} / 100", (46, 204, 113) if fish_rem > 0 else (231, 76, 60)),
        ("Current Gen Best:", f"{best_fit_current:.1f}", (241, 196, 15)),
        ("All-Time Best:", f"{all_time_best:.1f}", (241, 196, 15)),
        ("Mutation Rate/Str:", f"{MUTATION_RATE * 100:.0f}% / {MUTATION_STRENGTH:.2f}", (155, 89, 182))
    ]
    
    y_offset = 10
    text_font = pygame.font.SysFont(None, 15)
    
    for label, val, color in lines:
        lbl_r = text_font.render(label, True, (170, 180, 190))
        val_r = text_font.render(val, True, color)
        hud_surface.blit(lbl_r, (15, y_offset))
        hud_surface.blit(val_r, (140, y_offset))
        y_offset += 20
        
    # Controls help list
    help_txt = text_font.render("Keys: [D] Dbg | [G] Graph | [S] Save | [L] Load | [E] CSV", True, (130, 140, 150))
    hud_surface.blit(help_txt, (10, panel_h - 22))
    
    screen.blit(hud_surface, (10, 10))


def main():
    # Initialize Pygame
    pygame.init()
    pygame.font.init()
    
    # Setup window and clock
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("NeuroSwarm - Stage 7 Analytics & Genome Portability")
    clock = pygame.time.Clock()
    
    # Load default font
    font = pygame.font.SysFont(None, 24)
    
    # Evolution variables
    generation_num = 1
    generation_timer = 0
    stats_history = []
    
    # Global state genomes variables
    all_time_best_weights = None
    all_time_best_fitness = 0.0
    loaded_champion_weights = None
    
    # Graph caching surface setup
    graph_surface = pygame.Surface((300, 200), pygame.SRCALPHA)
    draw_fitness_graph(graph_surface, stats_history) # Draw initial empty graph
    
    # Spawn predator in the center
    predator = Predator(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2, random.uniform(0, 2 * math.pi))
    
    # Spawn Initial Generation (Gen 1)
    fish_school = []
    for _ in range(NUM_FISH):
        x = random.uniform(0, WINDOW_WIDTH)
        y = random.uniform(0, WINDOW_HEIGHT)
        heading = random.uniform(0, 2 * math.pi)
        speed_factor = random.uniform(SPEED_VARIATION_MIN, 1.0)
        color = generate_fish_color()
        fish_school.append(Fish(x, y, heading, speed_factor, color))
        
    dead_fish_pool = []
    
    debug_mode = False
    show_graph = False
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
                elif event.key == pygame.K_g:
                    show_graph = not show_graph
                elif event.key == pygame.K_s:
                    # Save currently cached all-time best
                    # Scan current active population for a potential updates first
                    active_pop = fish_school + dead_fish_pool
                    if active_pop:
                        c_best_fish = max(active_pop, key=lambda f: f.fitness)
                        if c_best_fish.fitness > all_time_best_fitness:
                            all_time_best_fitness = c_best_fish.fitness
                            all_time_best_weights = c_best_fish.brain.get_weights()
                            
                    if all_time_best_weights is not None:
                        save_genome(all_time_best_weights, all_time_best_fitness, generation_num, "best_genome.json")
                    else:
                        print("[WARNING] No genome has been evaluated yet to save.")
                elif event.key == pygame.K_l:
                    # Load weights from disk
                    loaded_w = load_genome("best_genome.json")
                    if loaded_w is not None:
                        loaded_champion_weights = loaded_w
                elif event.key == pygame.K_e:
                    # Export statistics to CSV
                    export_stats_csv(stats_history, "generation_history.csv")
                    
        # 2. Update Sensors
        for fish in fish_school:
            fish.sense(fish_school, predator)
            
        predator.hunt(fish_school)
        
        # 3. Brain Inference
        for fish in fish_school:
            fish.think()
        
        # 4. Update Physics
        for fish in fish_school:
            fish.update()
            
        # 5. Catch Detection & Eaten Pool Accumulation
        surviving_fish = []
        for fish in fish_school:
            dx = fish.x - predator.x
            if dx > WINDOW_WIDTH / 2: dx -= WINDOW_WIDTH
            elif dx < -WINDOW_WIDTH / 2: dx += WINDOW_WIDTH
            
            dy = fish.y - predator.y
            if dy > WINDOW_HEIGHT / 2: dy -= WINDOW_HEIGHT
            elif dy < -WINDOW_HEIGHT / 2: dy += WINDOW_HEIGHT
            
            dist = math.hypot(dx, dy)
            if dist <= PREDATOR_CATCH_RADIUS:
                dead_fish_pool.append(fish)
            else:
                surviving_fish.append(fish)
        fish_school = surviving_fish
        
        # Update Gen Timer
        generation_timer += 1
        
        # Gen completion check (All dead OR Time limit exceeded)
        generation_ended = (len(fish_school) == 0) or (generation_timer >= GENERATION_TIME_LIMIT_FRAMES)
        
        if generation_ended:
            entire_population = fish_school + dead_fish_pool
            
            # Extract Gen stats
            fitness_scores = [f.fitness for f in entire_population]
            best_fit = max(fitness_scores) if fitness_scores else 0.0
            avg_fit = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0
            survivor_count = len(fish_school)
            
            # Update all-time best tracker
            if best_fit > all_time_best_fitness:
                all_time_best_fitness = best_fit
                # Find the actual fish that achieved this fitness
                best_agent = max(entire_population, key=lambda f: f.fitness)
                all_time_best_weights = best_agent.brain.get_weights()
            
            # Log metrics
            stats_history.append({
                "generation": generation_num,
                "best_fitness": best_fit,
                "avg_fitness": avg_fit,
                "survivors": survivor_count
            })
            
            # Print transition stats to terminal console
            print(f"--- Generation {generation_num} Summary ---")
            print(f"Best Fitness: {best_fit:.2f}")
            print(f"Average Fitness: {avg_fit:.2f}")
            print(f"Survivors: {survivor_count} / 100")
            print("-----------------------------------")
            
            # Invalidate/re-render cached graph surface
            draw_fitness_graph(graph_surface, stats_history)
            
            # Breed next generation (inject loaded weights if set)
            fish_school = evolve_population(entire_population, loaded_champion_weights)
            loaded_champion_weights = None  # Consume champion injection
            dead_fish_pool = []
            
            # Reset Predator
            predator.x = WINDOW_WIDTH / 2
            predator.y = WINDOW_HEIGHT / 2
            predator.target_fish = None
            predator.vx = predator.max_speed * math.cos(random.uniform(0, 2 * math.pi))
            predator.vy = predator.max_speed * math.sin(random.uniform(0, 2 * math.pi))
            
            # Reset timers and increment counter
            generation_num += 1
            generation_timer = 0
            continue
            
        # Get live best fitness for stats display
        active_pop = fish_school + dead_fish_pool
        best_fitness_current = max(f.fitness for f in active_pop) if active_pop else 0.0
        
        # Calculate time remaining
        time_remaining_sec = max(0.0, (GENERATION_TIME_LIMIT_FRAMES - generation_timer) / FPS)
        fish_remaining = len(fish_school)
            
        # 6. Rendering
        screen.fill(BG_COLOR)
        
        # Draw fish
        for fish in fish_school:
            fish.draw(screen)
            
        # Draw predator
        predator.draw(screen)
        
        # Draw Debug Overlays
        if debug_mode:
            mouse_pos = pygame.mouse.get_pos()
            selected_fish = None
            min_mouse_dist = float('inf')
            for fish in fish_school:
                d = math.hypot(fish.x - mouse_pos[0], fish.y - mouse_pos[1])
                if d < min_mouse_dist:
                    min_mouse_dist = d
                    selected_fish = fish
                    
            if selected_fish is not None:
                pygame.draw.circle(screen, (80, 100, 120), (int(selected_fish.x), int(selected_fish.y)), VISION_RADIUS, 1)
                
                if selected_fish.nearest_neighbor is not None:
                    is_in_range = selected_fish.nearest_neighbor_dist <= VISION_RADIUS
                    line_color = (46, 204, 113) if is_in_range else (127, 140, 141)
                    pygame.draw.line(screen, line_color, 
                                     (int(selected_fish.x), int(selected_fish.y)), 
                                     (int(selected_fish.nearest_neighbor.x), int(selected_fish.nearest_neighbor.y)), 2)
                    pygame.draw.circle(screen, line_color, (int(selected_fish.nearest_neighbor.x), int(selected_fish.nearest_neighbor.y)), 5, 1)
                
                end_x = selected_fish.x + selected_fish.sensor_wall_dist * math.cos(selected_fish.heading)
                end_y = selected_fish.y + selected_fish.sensor_wall_dist * math.sin(selected_fish.heading)
                pygame.draw.line(screen, (230, 126, 34), (int(selected_fish.x), int(selected_fish.y)), (int(end_x), int(end_y)), 2)
                pygame.draw.circle(screen, (230, 126, 34), (int(end_x), int(end_y)), 4)
                
                pygame.draw.circle(screen, (241, 196, 15), (int(selected_fish.x), int(selected_fish.y)), 10, 1)
                
                outputs_text = font.render(f"S: {selected_fish.last_steer_output:+.2f} | A: {selected_fish.last_accel_output:+.2f}", True, (241, 196, 15))
                screen.blit(outputs_text, (int(selected_fish.x) - 50, int(selected_fish.y) - 25))
                
                sensor_text_1 = font.render(f"Nearest Dist: {selected_fish.sensor_nearest_dist:.1f}", True, (200, 200, 200))
                sensor_text_2 = font.render(f"Nearest Angle: {selected_fish.sensor_nearest_angle:.2f} rad", True, (200, 200, 200))
                sensor_text_3 = font.render(f"Wall Dist: {selected_fish.sensor_wall_dist:.1f}", True, (200, 200, 200))
                sensor_text_4 = font.render(f"Predator Dist: {selected_fish.sensor_predator_dist:.1f}", True, (200, 200, 200))
                sensor_text_5 = font.render(f"Survival Fitness: {selected_fish.fitness:.1f}", True, (241, 196, 15))
                screen.blit(sensor_text_1, (10, WINDOW_HEIGHT - 110))
                screen.blit(sensor_text_2, (10, WINDOW_HEIGHT - 90))
                screen.blit(sensor_text_3, (10, WINDOW_HEIGHT - 70))
                screen.blit(sensor_text_4, (10, WINDOW_HEIGHT - 50))
                screen.blit(sensor_text_5, (10, WINDOW_HEIGHT - 30))
                
            pygame.draw.circle(screen, (150, 50, 50), (int(predator.x), int(predator.y)), PREDATOR_VISION_RADIUS, 1)
            if predator.target_fish is not None:
                is_chasing = predator.target_fish_dist <= PREDATOR_VISION_RADIUS
                line_color = (231, 76, 60) if is_chasing else (127, 140, 141)
                pygame.draw.line(screen, line_color, 
                                 (int(predator.x), int(predator.y)), 
                                 (int(predator.target_fish.x), int(predator.target_fish.y)), 2)
                pygame.draw.circle(screen, line_color, (int(predator.target_fish.x), int(predator.target_fish.y)), 8, 1)

        # Draw Consolidated HUD (top-left)
        fps_val = clock.get_fps()
        draw_hud(screen, font, fps_val, generation_num, time_remaining_sec, fish_remaining, best_fitness_current, max(all_time_best_fitness, best_fitness_current))
        
        # Draw Cached Fitness Graph Overlay (bottom-right) if toggled on
        if show_graph:
            screen.blit(graph_surface, (WINDOW_WIDTH - 320, WINDOW_HEIGHT - 220))
            
        # =====================================================================
        # STAGE 8 HOOK: Training-Mode (Headless / Fast-Forward Simulation)
        # - This is where Stage 8 will implement fast-forward loops or disable
        #   screen rendering options to dramatically increase training throughput.
        # =====================================================================
            
        # Flip display and tick clock
        pygame.display.flip()
        clock.tick(FPS)
        
    # Clean cleanup and exit
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
