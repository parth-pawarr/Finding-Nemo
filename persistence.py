import json
import time
import numpy as np
import logging

# Configure logger
logger = logging.getLogger(__name__)

def save_genome(weights, fitness, generation, filepath="best_genome.json"):
    """
    Serializes a flat weight array and training metadata to a JSON file.
    
    Inputs:
        weights: 1D NumPy float array
        fitness: Fitness score achieved by the genome (float)
        generation: Current generation index (integer)
        filepath: Target filepath destination (string, default best_genome.json)
    """
    try:
        data = {
            "weights": weights.tolist(),  # Convert numpy array to list for JSON serialization
            "fitness": float(fitness),
            "generation": int(generation),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved champion genome to {filepath} (Fitness: {fitness:.2f}, Gen: {generation})")
    except Exception as e:
        logger.error(f"Failed to save genome to {filepath}: {e}", exc_info=True)


def load_genome(filepath="best_genome.json"):
    """
    Loads weights from a saved JSON genome file.
    
    Inputs:
        filepath: Target filepath to load (string, default best_genome.json)
        
    Outputs:
        weights: 1D NumPy float array, or None if load fails
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        weights = np.array(data["weights"])
        logger.info(f"Loaded genome from {filepath} (Fitness: {data['fitness']:.2f}, Saved Gen: {data['generation']})")
        return weights
    except Exception as e:
        logger.error(f"Failed to load genome from {filepath}: {e}")
        return None
