# =====================================================================
# NEUROSWARM GLOBAL CONFIGURATION CONSTANTS
# =====================================================================

# 1. Display Configurations
WINDOW_WIDTH = 1200             # Default window width (pixels)
WINDOW_HEIGHT = 800             # Default window height (pixels)
FPS = 60                        # Framerate target in real-time mode (frames/second)
NUM_FISH = 100                  # Size of the prey fish population

# 2. Prey Physics & Kinematics (Fish)
MAX_SPEED = 4.0                 # Target top speed of fish (pixels/frame)
MAX_ACCELERATION = 0.15         # Maximum steering force acceleration (pixels/frame^2)
MAX_TURN_RATE = 0.08            # Limit on angle change per frame (radians/frame)
SPEED_VARIATION_MIN = 0.8       # Minimum speed modifier (80% of MAX_SPEED)
WANDER_INTERVAL = 45            # Avg frames before picking new wander heading target
WANDER_FORCE = 0.5              # Max angle deflection for random wander adjustment (radians)

# 3. Vision & Sensors (Fish)
VISION_RADIUS = 120             # Vision radius within which fish sense neighbors (pixels)

# 4. Predator Configurations
PREDATOR_MAX_SPEED = 4.8        # Predator top speed (pixels/frame)
PREDATOR_MAX_ACCELERATION = 0.20 # Predator steering lunge acceleration (pixels/frame^2)
PREDATOR_MAX_TURN_RATE = 0.05   # Predator max turn rate (radians/frame)
PREDATOR_VISION_RADIUS = 250    # Distance within which predator spots fish (pixels)
PREDATOR_CATCH_RADIUS = 15      # Proximity range for eating a fish (pixels)

# 5. Neuroevolution (GA parameters)
GENERATION_TIME_LIMIT_FRAMES = 3600  # Max frames per generation (60s @ 60 FPS)
ELITE_PERCENT = 0.15                 # Top 15% preserved exactly as elites (cloned)
TOURNAMENT_SIZE = 5                  # Pool size for tournament parent selection
MUTATION_RATE = 0.08                 # Probability of mutating each weight (8%)
MUTATION_STRENGTH = 0.15             # Standard deviation of Gaussian mutation nudge

# 6. Fitness Scoring weights
FITNESS_SURVIVAL_REWARD = 1.0        # Reward added to fitness every frame survived
FITNESS_SPEED_MULT = 0.5             # Reward multiplier for movement speed
FITNESS_WALL_PENALTY_THRESHOLD = 60.0 # Distance to boundary under which penalty begins (pixels)
FITNESS_WALL_PENALTY_MAX = 3.0       # Maximum penalty subtracted when touching walls

# 7. Aesthetics / Colors (RGB tuples)
BG_COLOR = (10, 15, 30)         # Dark ocean deep navy
FPS_COLOR = (200, 200, 200)     # Light grey HUD text
