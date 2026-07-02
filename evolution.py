import random
import math
import numpy as np

# Import constants and classes
import config
from utils import generate_fish_color
from entities import Fish

def tournament_selection(population):
    """
    Selects TOURNAMENT_SIZE random fish candidates from the population
    and returns the one with the highest fitness score.
    
    Inputs:
        population: List of Fish objects (both alive and dead)
        
    Outputs:
        best_candidate: The Fish object with the highest fitness in the tournament
    """
    candidates = random.sample(population, config.TOURNAMENT_SIZE)
    return max(candidates, key=lambda f: f.fitness)


def crossover(parent_a, parent_b):
    """
    Merges flat weight representations of two parents using uniform crossover.
    Each weight is chosen from parent_a or parent_b with 50% probability.
    
    Inputs:
        parent_a, parent_b: Fish objects containing neural network brains
        
    Outputs:
        child_w: 1D NumPy array representing the combined weights
    """
    w_a = parent_a.brain.get_weights()
    w_b = parent_b.brain.get_weights()
    child_w = np.copy(w_a)
    mask = np.random.rand(len(w_a)) < 0.5
    child_w[mask] = w_b[mask]
    return child_w


def mutate(weights):
    """
    Adds Gaussian random noise to a fraction of the weight values.
    
    Inputs:
        weights: 1D NumPy float array
        
    Outputs:
        mutated_w: 1D NumPy float array with applied mutations
    """
    mutated_w = np.copy(weights)
    mask = np.random.rand(len(weights)) < config.MUTATION_RATE
    noise = np.random.normal(0, config.MUTATION_STRENGTH, size=np.sum(mask))
    mutated_w[mask] += noise
    return mutated_w


def evolve_population(entire_population, loaded_weights=None, world_w=config.WINDOW_WIDTH, world_h=config.WINDOW_HEIGHT):
    """
    Executes elitism, tournament selection, crossover, and mutation to
    produce a new generation of 100 fish.
    
    Inputs:
        entire_population: List of 100 Fish objects from the previous generation
        loaded_weights: Optional 1D NumPy array containing loaded champion weights
        world_w, world_h: Current screen boundaries to spawn fish within (floats)
        
    Outputs:
        new_fish_school: List of 100 newly spawned evolved Fish objects
    """
    # Sort by fitness descending
    sorted_pop = sorted(entire_population, key=lambda f: f.fitness, reverse=True)
    
    # Calculate elites count
    elite_count = int(config.NUM_FISH * config.ELITE_PERCENT)
    
    new_genomes = []
    
    # Inject loaded champion weights if available
    champion_injected = False
    if loaded_weights is not None:
        new_genomes.append(loaded_weights)
        champion_injected = True
        
    # Preserve remaining elites exactly (zero mutation, zero crossover)
    start_idx = 1 if champion_injected else 0
    for i in range(start_idx, elite_count):
        new_genomes.append(sorted_pop[i].brain.get_weights())
        
    # Breed the rest of the slots
    while len(new_genomes) < config.NUM_FISH:
        parent_a = tournament_selection(sorted_pop)
        parent_b = tournament_selection(sorted_pop)
        
        # Crossover
        child_w = crossover(parent_a, parent_b)
        
        # Mutation
        child_w = mutate(child_w)
        
        new_genomes.append(child_w)
        
    # Spawn new fish school with evolved weights
    new_fish_school = []
    for i in range(config.NUM_FISH):
        x = random.uniform(0, world_w)
        y = random.uniform(0, world_h)
        heading = random.uniform(0, 2 * math.pi)
        speed_factor = random.uniform(config.SPEED_VARIATION_MIN, 1.0)
        
        if i == 0 and champion_injected:
            color = (255, 215, 0)  # Gold
            fish = Fish(x, y, heading, speed_factor, color)
            fish.brain.set_weights(new_genomes[i])
            fish.is_champion = True
        else:
            color = generate_fish_color()
            fish = Fish(x, y, heading, speed_factor, color)
            fish.brain.set_weights(new_genomes[i])
            
        new_fish_school.append(fish)
        
    return new_fish_school
