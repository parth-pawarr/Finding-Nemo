import random

def toroidal_difference(x1, y1, x2, y2, width, height):
    """
    Computes the shortest coordinate differences (dx, dy) from point (x1, y1)
    to point (x2, y2) taking wrap-around toroidal boundary geometry into account.
    
    Inputs:
        x1, y1: Coordinates of starting point (floats)
        x2, y2: Coordinates of target point (floats)
        width: Current width boundary of the toroidal world (float)
        height: Current height boundary of the toroidal world (float)
        
    Outputs:
        dx, dy: Shortest offset vector components (floats)
    """
    dx = x2 - x1
    if dx > width / 2:
        dx -= width
    elif dx < -width / 2:
        dx += width
        
    dy = y2 - y1
    if dy > height / 2:
        dy -= height
    elif dy < -height / 2:
        dy += height
        
    return dx, dy


def generate_fish_color():
    """
    Generates a color within a subtle silver-blue metallic range.
    Produces individual variations so the school does not look like a flat blob.
    
    Outputs:
        (r, g, b): Tuple of RGB values (integers in [0, 255])
    """
    r = random.randint(100, 180)  # Metallic silver base
    g = random.randint(140, 200)  # Hint of soft green/teal
    b = random.randint(200, 255)  # Dominant blue component
    return (r, g, b)
