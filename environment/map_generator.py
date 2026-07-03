import random
import math
import config
from environment.obstacle import Obstacle
from utils import toroidal_difference

def validate_map_traversability(obstacles, width, height, min_gap):
    """
    Validates map traversability using a toroidal grid-based BFS.
    Returns True if the map has at least 85% reachability from the center spawn zone.
    """
    cell_size = 20
    cols = int(width // cell_size)
    rows = int(height // cell_size)
    
    # Initialize grid: False means open, True means blocked
    grid = [[False for _ in range(rows)] for _ in range(cols)]
    
    # Precompute cell centers and mark blocked cells
    total_open = 0
    for c in range(cols):
        for r in range(rows):
            cell_x = c * cell_size + cell_size / 2
            cell_y = r * cell_size + cell_size / 2
            
            # Check if cell is too close to any obstacle
            blocked = False
            for obs in obstacles:
                dx, dy = toroidal_difference(obs.x, obs.y, cell_x, cell_y, width, height)
                dist = math.hypot(dx, dy)
                if dist < (obs.radius + min_gap / 2.0):
                    blocked = True
                    break
            grid[c][r] = blocked
            if not blocked:
                total_open += 1
                
    if total_open == 0:
        return False
        
    # Start BFS from center of the map
    start_c = cols // 2
    start_r = rows // 2
    
    # If the center cell itself is blocked, the center is not clear for spawning.
    if grid[start_c][start_r]:
        return False
        
    queue = [(start_c, start_r)]
    visited = set()
    visited.add((start_c, start_r))
    
    reachable_count = 0
    
    while queue:
        curr_c, curr_r = queue.pop(0)
        reachable_count += 1
        
        # 4-way neighbors
        for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nc = (curr_c + dc) % cols
            nr = (curr_r + dr) % rows
            if not grid[nc][nr] and (nc, nr) not in visited:
                visited.add((nc, nr))
                queue.append((nc, nr))
                
    # Check if we can reach at least 85% of all open cells
    ratio = reachable_count / total_open
    return ratio >= 0.85


def generate_procedural_obstacles(width, height, count, min_r, max_r, min_gap, max_density):
    """
    Procedural map generator with overlap prevention, density cap, and
    BFS traversability validation (Pass C).
    """
    attempts = 0
    current_count = count
    
    while attempts < config.MAX_MAP_ATTEMPTS:
        attempts += 1
        obstacles = []
        max_allowed_area = max_density * width * height
        current_area = 0.0
        
        success = True
        for _ in range(current_count):
            placed = False
            for _ in range(100):
                radius = random.uniform(min_r, max_r)
                if current_area + math.pi * radius * radius > max_allowed_area:
                    if current_area + math.pi * min_r * min_r > max_allowed_area:
                        break  # Reached density cap
                    radius = min_r
                    
                x = random.uniform(0, width)
                y = random.uniform(0, height)
                
                # Check overlap (with toroidal difference)
                overlapping = False
                for obs in obstacles:
                    dx, dy = toroidal_difference(obs.x, obs.y, x, y, width, height)
                    dist = math.hypot(dx, dy)
                    if dist < (obs.radius + radius):
                        overlapping = True
                        break
                        
                if not overlapping:
                    obstacles.append(Obstacle(x, y, radius))
                    current_area += math.pi * radius * radius
                    placed = True
                    break
            
            # If we couldn't place enough obstacles due to crowding, break early and validate what we have
            if not placed and len(obstacles) < current_count:
                pass
                
        # Now validate traversability of this obstacle set
        if validate_map_traversability(obstacles, width, height, min_gap):
            return obstacles
            
        # If validation fails, we try again
        # If attempts are halfway through, try reducing the obstacle count to ease validation
        if attempts == config.MAX_MAP_ATTEMPTS // 2:
            current_count = max(5, current_count - 1)
            
    # Fallback: if all attempts fail, return the last generated set.
    return obstacles


def find_safe_spawn_position(obstacles, world_w, world_h, radius):
    """
    Finds a safe coordinate (x, y) that does not overlap any obstacle.
    """
    for _ in range(200):  # Limit attempts to prevent infinite loop
        x = random.uniform(0, world_w)
        y = random.uniform(0, world_h)
        # Check overlaps
        overlapping = False
        for obs in obstacles:
            if obs.is_circle_overlapping(x, y, radius, world_w, world_h):
                overlapping = True
                break
        if not overlapping:
            return x, y
            
    # Fallback to random position if all else fails
    return random.uniform(0, world_w), random.uniform(0, world_h)
