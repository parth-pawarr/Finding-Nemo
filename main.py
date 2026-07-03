import pygame
import random
import math
import sys
import numpy as np
import logging

# Import modular components
import config
from utils import generate_fish_color, toroidal_difference
from entities import Fish, Predator
from evolution import evolve_population
from analytics import draw_fitness_graph, draw_hud, draw_help_panel, export_stats_csv
from persistence import save_genome, load_genome
from environment.obstacle import get_fixed_obstacles
from environment.map_generator import generate_procedural_obstacles, find_safe_spawn_position



# Setup module-level logger
logger = logging.getLogger(__name__)

# Active simulation boundaries (initialized to target window size, modified dynamically in fullscreen)
CURRENT_WIDTH = config.WINDOW_WIDTH
CURRENT_HEIGHT = config.WINDOW_HEIGHT


def main(): 
    global CURRENT_WIDTH, CURRENT_HEIGHT
    
    # Configure standard Python logging to output milestones to stdout console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    logger.info("Initializing NeuroSwarm Simulation...")
    
    # Initialize Pygame
    pygame.init()
    pygame.font.init()
    
    # Setup window and clock
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    pygame.display.set_caption("NeuroSwarm - Evolving Simulation Foundation")
    clock = pygame.time.Clock()
    
    # Load default font for HUD and debugging text
    font = pygame.font.SysFont(None, 24)
    
    # Evolution variables
    generation_num = 1
    generation_timer = 0
    stats_history = []
    
    # Global state genomes variables
    all_time_best_weights = None
    all_time_best_fitness = 0.0
    loaded_champion_weights = None
    
    # Graph caching surface setup (updated only on generation transitions)
    graph_surface = pygame.Surface((300, 200), pygame.SRCALPHA)
    draw_fitness_graph(graph_surface, stats_history)  # Draw initial empty graph
    
    # Initialize static obstacles
    use_fixed_map = False
    obstacles = (get_fixed_obstacles() if use_fixed_map else 
                 generate_procedural_obstacles(CURRENT_WIDTH, CURRENT_HEIGHT, config.OBSTACLE_COUNT, 
                                               config.OBSTACLE_MIN_RADIUS, config.OBSTACLE_MAX_RADIUS, 
                                               config.MIN_GAP_WIDTH, config.MAX_DENSITY))
    
    # Spawn predator in a safe position
    px, py = find_safe_spawn_position(obstacles, CURRENT_WIDTH, CURRENT_HEIGHT, config.PREDATOR_COLLISION_RADIUS)
    predator = Predator(px, py, random.uniform(0, 2 * math.pi))
    
    # Spawn Initial Generation (Gen 1 - Random brains)
    fish_school = []
    for _ in range(config.NUM_FISH):
        x, y = find_safe_spawn_position(obstacles, CURRENT_WIDTH, CURRENT_HEIGHT, config.FISH_COLLISION_RADIUS)
        heading = random.uniform(0, 2 * math.pi)
        speed_factor = random.uniform(config.SPEED_VARIATION_MIN, 1.0)
        color = generate_fish_color()
        fish_school.append(Fish(x, y, heading, speed_factor, color))
        
    dead_fish_pool = []  # Accumulate fish eaten during the generation
    
    # Mode variables
    debug_mode = False
    show_graph = False
    fullscreen = False
    training_mode = False
    draw_details = True
    show_help = True  # Help menu shown by default for discoverability
    collision_mode = 'STOP'
    manual_mode = False

    
    # Timer for training mode HUD updates
    last_training_hud_update_ms = 0
    
    logger.info("Simulation engine started successfully. Press [H] to toggle the controls menu.")
    
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
                    if draw_details:
                        debug_mode = not debug_mode
                        logger.info(f"Debug overlay toggled: {debug_mode}")
                elif event.key == pygame.K_v:
                    if draw_details:
                        show_graph = not show_graph
                        logger.info(f"Fitness graph toggled: {show_graph}")
                elif event.key == pygame.K_g:
                    if draw_details:
                        if use_fixed_map:
                            logger.warning("Cannot regenerate map while in FIXED map mode. Switch to PROCEDURAL mode using [O] first.")
                        else:
                            obstacles = generate_procedural_obstacles(CURRENT_WIDTH, CURRENT_HEIGHT, config.OBSTACLE_COUNT, 
                                                                       config.OBSTACLE_MIN_RADIUS, config.OBSTACLE_MAX_RADIUS, 
                                                                       config.MIN_GAP_WIDTH, config.MAX_DENSITY)
                            # Reposition fish safely
                            for fish in fish_school:
                                x, y = find_safe_spawn_position(obstacles, CURRENT_WIDTH, CURRENT_HEIGHT, config.FISH_COLLISION_RADIUS)
                                fish.x = x
                                fish.y = y
                            # Reposition predator safely
                            px, py = find_safe_spawn_position(obstacles, CURRENT_WIDTH, CURRENT_HEIGHT, config.PREDATOR_COLLISION_RADIUS)
                            predator.x = px
                            predator.y = py
                            predator.vx = predator.max_speed * math.cos(random.uniform(0, 2 * math.pi))
                            predator.vy = predator.max_speed * math.sin(random.uniform(0, 2 * math.pi))
                            logger.info("Procedural map regenerated and agents repositioned.")
                elif event.key == pygame.K_o:
                    if draw_details:
                        use_fixed_map = not use_fixed_map
                        if use_fixed_map:
                            obstacles = get_fixed_obstacles()
                            logger.info("Switched to FIXED map mode.")
                        else:
                            obstacles = generate_procedural_obstacles(CURRENT_WIDTH, CURRENT_HEIGHT, config.OBSTACLE_COUNT, 
                                                                       config.OBSTACLE_MIN_RADIUS, config.OBSTACLE_MAX_RADIUS, 
                                                                       config.MIN_GAP_WIDTH, config.MAX_DENSITY)
                            logger.info("Switched to PROCEDURAL map mode.")
                        
                        # Re-spawn agents safely on map switch
                        for fish in fish_school:
                            x, y = find_safe_spawn_position(obstacles, CURRENT_WIDTH, CURRENT_HEIGHT, config.FISH_COLLISION_RADIUS)
                            fish.x = x
                            fish.y = y
                        px, py = find_safe_spawn_position(obstacles, CURRENT_WIDTH, CURRENT_HEIGHT, config.PREDATOR_COLLISION_RADIUS)
                        predator.x = px
                        predator.y = py
                        predator.vx = predator.max_speed * math.cos(random.uniform(0, 2 * math.pi))
                        predator.vy = predator.max_speed * math.sin(random.uniform(0, 2 * math.pi))
                elif event.key == pygame.K_h:
                    if draw_details:
                        show_help = not show_help
                        logger.info(f"Help legend panel toggled: {show_help}")
                elif event.key == pygame.K_r:
                    draw_details = not draw_details
                    logger.info(f"Lighter rendering toggled: {not draw_details}")
                    if not draw_details:
                        debug_mode = False
                        show_graph = False
                        show_help = False
                elif event.key == pygame.K_s:
                    # Save currently cached all-time best
                    active_pop = fish_school + dead_fish_pool
                    if active_pop:
                        c_best_fish = max(active_pop, key=lambda f: f.fitness)
                        if c_best_fish.fitness > all_time_best_fitness:
                            all_time_best_fitness = c_best_fish.fitness
                            all_time_best_weights = c_best_fish.brain.get_weights()
                            
                    if all_time_best_weights is not None:
                        save_genome(all_time_best_weights, all_time_best_fitness, generation_num, "best_genome.json")
                    else:
                        logger.warning("No genome has been evaluated yet to save.")
                elif event.key == pygame.K_l:
                    # Load weights from disk
                    loaded_w = load_genome("best_genome.json")
                    if loaded_w is not None:
                        loaded_champion_weights = loaded_w
                elif event.key == pygame.K_e:
                    # Export statistics history to CSV
                    export_stats_csv(stats_history, "generation_history.csv")
                elif event.key == pygame.K_f:
                    # Fullscreen Mode Toggle
                    fullscreen = not fullscreen
                    if fullscreen:
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        CURRENT_WIDTH, CURRENT_HEIGHT = screen.get_size()
                    else:
                        screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
                        CURRENT_WIDTH, CURRENT_HEIGHT = config.WINDOW_WIDTH, config.WINDOW_HEIGHT
                    # Relocate Predator to screen center
                    predator.x = CURRENT_WIDTH / 2
                    predator.y = CURRENT_HEIGHT / 2
                    logger.info(f"Fullscreen mode toggled: {fullscreen} (Resolution: {CURRENT_WIDTH}x{CURRENT_HEIGHT})")
                elif event.key == pygame.K_t:
                    # Headless Fast Training Mode Toggle
                    training_mode = not training_mode
                    logger.info(f"Fast training mode toggled: {training_mode}")
                elif event.key == pygame.K_c:
                    if draw_details:
                        modes = ['OFF', 'STOP', 'SLIDE']
                        idx = (modes.index(collision_mode) + 1) % len(modes)
                        collision_mode = modes[idx]
                        logger.info(f"Collision mode set to: {collision_mode}")
                elif event.key == pygame.K_m:
                    if draw_details:
                        manual_mode = not manual_mode
                        logger.info(f"Manual control mode toggled: {manual_mode}")
                    
        # 2. Update Sensors
        for fish in fish_school:
            fish.sense(fish_school, predator, CURRENT_WIDTH, CURRENT_HEIGHT)
            
        predator.hunt(fish_school, CURRENT_WIDTH, CURRENT_HEIGHT, obstacles, collision_mode)
        
        # 3. Brain Inference
        keys = pygame.key.get_pressed()
        for idx, fish in enumerate(fish_school):
            if manual_mode and idx == 0:
                fish.think_manual(keys)
            else:
                fish.think(CURRENT_WIDTH, CURRENT_HEIGHT)
        
        # 4. Update Physics
        for fish in fish_school:
            fish.update(CURRENT_WIDTH, CURRENT_HEIGHT, obstacles, collision_mode)
            
        # 5. Catch Detection & Eaten Pool Accumulation (using toroidal math helpers)
        surviving_fish = []
        for fish in fish_school:
            dx, dy = toroidal_difference(predator.x, predator.y, fish.x, fish.y, CURRENT_WIDTH, CURRENT_HEIGHT)
            dist = math.hypot(dx, dy)
            if dist <= config.PREDATOR_CATCH_RADIUS:
                dead_fish_pool.append(fish)
            else:
                surviving_fish.append(fish)
        fish_school = surviving_fish
        
        # Update Gen Timer
        generation_timer += 1
        
        # Gen completion check (All dead OR Time limit exceeded)
        generation_ended = (len(fish_school) == 0) or (generation_timer >= config.GENERATION_TIME_LIMIT_FRAMES)
        
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
                best_agent = max(entire_population, key=lambda f: f.fitness)
                all_time_best_weights = best_agent.brain.get_weights()
            
            # Log metrics
            stats_history.append({
                "generation": generation_num,
                "best_fitness": best_fit,
                "avg_fitness": avg_fit,
                "survivors": survivor_count
            })
            
            # Print transition stats to terminal console using logger
            logger.info(f"--- Generation {generation_num} Summary ---")
            logger.info(f"Best Fitness: {best_fit:.2f}")
            logger.info(f"Average Fitness: {avg_fit:.2f}")
            logger.info(f"Survivors: {survivor_count} / {config.NUM_FISH}")
            logger.info("-----------------------------------")
            
            # Invalidate/re-render cached graph surface
            draw_fitness_graph(graph_surface, stats_history)
            
            # Breed next generation (inject loaded weights if set)
            fish_school = evolve_population(entire_population, loaded_champion_weights, CURRENT_WIDTH, CURRENT_HEIGHT)
            # Reposition fish safely so they do not spawn inside obstacles
            for fish in fish_school:
                x, y = find_safe_spawn_position(obstacles, CURRENT_WIDTH, CURRENT_HEIGHT, config.FISH_COLLISION_RADIUS)
                fish.x = x
                fish.y = y
                
            loaded_champion_weights = None  # Consume champion injection
            dead_fish_pool = []
            
            # Reset Predator safely
            px, py = find_safe_spawn_position(obstacles, CURRENT_WIDTH, CURRENT_HEIGHT, config.PREDATOR_COLLISION_RADIUS)
            predator.x = px
            predator.y = py
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
        time_remaining_sec = max(0.0, (config.GENERATION_TIME_LIMIT_FRAMES - generation_timer) / config.FPS)
        fish_remaining = len(fish_school)
            
        # 6. Rendering (Skipped or minimized in Fast-Forward Training mode)
        if training_mode:
            current_time_ms = pygame.time.get_ticks()
            # Draw ONLY HUD updates once every 1000ms
            if current_time_ms - last_training_hud_update_ms >= 1000:
                last_training_hud_update_ms = current_time_ms
                
                screen.fill(config.BG_COLOR)
                fps_val = clock.get_fps()
                draw_hud(screen, font, fps_val, generation_num, time_remaining_sec, fish_remaining, best_fitness_current, max(all_time_best_fitness, best_fitness_current), collision_mode, manual_mode, "FIXED" if use_fixed_map else "PROCEDURAL")
                
                # Training notification indicator
                train_lbl = font.render("FAST FF-TRAINING ACTIVE (Agent Rendering Suspended)", True, (231, 76, 60))
                screen.blit(train_lbl, (10, 200))
                help_lbl = font.render("Press [T] to resume visual rendering mode", True, (200, 200, 200))
                screen.blit(help_lbl, (10, 225))
                
                pygame.display.flip()
            
            # Tick without capping FPS for fast-forward simulations
            clock.tick()
        else:
            # Clear screen with dark background
            screen.fill(config.BG_COLOR)
            
            # Draw static obstacles
            for obstacle in obstacles:
                obstacle.draw(screen)

            
            # Draw fish school
            for idx, fish in enumerate(fish_school):
                if manual_mode and idx == 0:
                    # Highlight manually controlled fish
                    pygame.draw.circle(screen, (255, 140, 0), (int(fish.x), int(fish.y)), 12, 2)
                fish.draw(screen, draw_details)
                
            # Draw predator
            predator.draw(screen, draw_details)
            
            # Draw Debug Overlays (only if details are enabled)
            if draw_details and debug_mode:
                mouse_pos = pygame.mouse.get_pos()
                selected_fish = None
                min_mouse_dist = float('inf')
                for fish in fish_school:
                    d = math.hypot(fish.x - mouse_pos[0], fish.y - mouse_pos[1])
                    if d < min_mouse_dist:
                        min_mouse_dist = d
                        selected_fish = fish
                        
                if selected_fish is not None:
                    pygame.draw.circle(screen, (80, 100, 120), (int(selected_fish.x), int(selected_fish.y)), config.VISION_RADIUS, 1)
                    
                    if selected_fish.nearest_neighbor is not None:
                        is_in_range = selected_fish.nearest_neighbor_dist <= config.VISION_RADIUS
                        line_color = (46, 204, 113) if is_in_range else (127, 140, 141)
                        # Draw toroidal direct connection
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
                    screen.blit(sensor_text_1, (10, CURRENT_HEIGHT - 110))
                    screen.blit(sensor_text_2, (10, CURRENT_HEIGHT - 90))
                    screen.blit(sensor_text_3, (10, CURRENT_HEIGHT - 70))
                    screen.blit(sensor_text_4, (10, CURRENT_HEIGHT - 50))
                    screen.blit(sensor_text_5, (10, CURRENT_HEIGHT - 30))
                    
                pygame.draw.circle(screen, (150, 50, 50), (int(predator.x), int(predator.y)), config.PREDATOR_VISION_RADIUS, 1)
                if predator.target_fish is not None:
                    is_chasing = predator.target_fish_dist <= config.PREDATOR_VISION_RADIUS
                    line_color = (231, 76, 60) if is_chasing else (127, 140, 141)
                    pygame.draw.line(screen, line_color, 
                                     (int(predator.x), int(predator.y)), 
                                     (int(predator.target_fish.x), int(predator.target_fish.y)), 2)
                    pygame.draw.circle(screen, line_color, (int(predator.target_fish.x), int(predator.target_fish.y)), 8, 1)

            # Draw HUD Overlays if details are enabled
            if draw_details:
                fps_val = clock.get_fps()
                draw_hud(screen, font, fps_val, generation_num, time_remaining_sec, fish_remaining, best_fitness_current, max(all_time_best_fitness, best_fitness_current), collision_mode, manual_mode, "FIXED" if use_fixed_map else "PROCEDURAL")
                
                # Draw Keybindings Reference Panel if toggled active
                if show_help:
                    draw_help_panel(screen, font, CURRENT_WIDTH)
                    
                # Draw Cached Line Graph Overlay if toggled active
                if show_graph:
                    screen.blit(graph_surface, (CURRENT_WIDTH - 320, CURRENT_HEIGHT - 220))
            
            # Flip display and limit frame rates in normal visual modes
            pygame.display.flip()
            clock.tick(config.FPS)
        
    # Clean cleanup and exit
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
