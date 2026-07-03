import math
import random
from utils import toroidal_difference

def resolve_agent_movement(agent, obstacles, world_w, world_h, collision_mode, N=4):
    """
    Moves the agent by dividing its velocity into N sub-steps per frame.
    Resolves overlaps with static circular obstacles in each sub-step.
    
    Inputs:
        agent: The Agent instance to move and check.
        obstacles: List of Obstacle instances.
        world_w, world_h: Screen/World dimensions.
        collision_mode: 'STOP' or 'SLIDE'.
        N: Number of physics sub-steps.
    """
    if not obstacles or collision_mode == 'OFF':
        # Default simple movement
        agent.x += agent.vx
        agent.y += agent.vy
        agent.handle_boundaries(world_w, world_h)
        return

    # Perform sub-stepped physics
    for sub_step in range(N):
        # 1. Update position by a fraction of velocity
        agent.x += agent.vx / N
        agent.y += agent.vy / N
        agent.handle_boundaries(world_w, world_h)
        
        # 2. Iteratively resolve overlaps (up to 3 passes to handle multi-obstacle/corner constraints)
        for _ in range(3):
            collided = False
            for obs in obstacles:
                # Calculate vector from obstacle center to agent center
                dx, dy = toroidal_difference(obs.x, obs.y, agent.x, agent.y, world_w, world_h)
                dist = math.hypot(dx, dy)
                min_dist = agent.radius + obs.radius
                
                if dist < min_dist:
                    collided = True
                    
                    # Compute normal vector pointing from obstacle to agent
                    if dist > 0.0001:
                        nx = dx / dist
                        ny = dy / dist
                    else:
                        # Fallback for exact center overlap
                        angle = random.uniform(0, 2 * math.pi)
                        nx = math.cos(angle)
                        ny = math.sin(angle)
                        dist = 0.0001
                        
                    # Reposition agent out of obstacle
                    overlap = min_dist - dist
                    agent.x += nx * overlap
                    agent.handle_boundaries(world_w, world_h)
                    
                    # Resolve velocity based on collision mode
                    if collision_mode == 'STOP':
                        agent.vx = 0.0
                        agent.vy = 0.0
                    elif collision_mode == 'SLIDE':
                        # Project velocity onto surface tangent
                        v_dot_n = agent.vx * nx + agent.vy * ny
                        if v_dot_n < 0:
                            # Subtract normal component of velocity
                            agent.vx -= v_dot_n * nx
                            agent.vy -= v_dot_n * ny
            
            # If no collisions occurred in this pass, constraints are fully resolved
            if not collided:
                break
