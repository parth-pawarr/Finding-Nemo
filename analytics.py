import pygame
import csv
import logging
import time

# Import constants
import config

# Configure logging
logger = logging.getLogger(__name__)

def draw_fitness_graph(surface, stats_history):
    """
    Draws a fitness progression line graph (best vs avg) to a Pygame surface.
    Auto-scales the axes dynamically as generations complete.
    
    Inputs:
        surface: The Pygame Surface to draw onto (size 300x200 recommended)
        stats_history: List of dictionary records containing historical generation stats
    """
    # Clear the surface with a semi-transparent dark blue-gray
    surface.fill((20, 25, 40, 220))
    width, height = surface.get_size()
    padding = 22
    plot_w = width - 2 * padding
    plot_h = height - 2 * padding
    
    # Outer border
    pygame.draw.rect(surface, (100, 110, 130), (0, 0, width, height), 2)
    
    font = pygame.font.SysFont(None, 14)
    
    if not stats_history:
        # Placeholder text when there is no generation data yet
        placeholder = font.render("Waiting for Gen 1 completion...", True, (150, 160, 180))
        surface.blit(placeholder, (padding + 10, height / 2 - 8))
        return

    # Auto-scale y-axis based on historical data maxima
    max_fit = max(max(g["best_fitness"], g["avg_fitness"]) for g in stats_history)
    max_val = max(1.0, max_fit * 1.1)  # Leave 10% headroom to keep lines below top border
    
    num_gens = len(stats_history)
    
    # Draw horizontal grid lines and y-axis labels
    for i in range(4):
        ratio = i / 3.0
        y_pos = padding + plot_h - (ratio * plot_h)
        pygame.draw.line(surface, (50, 60, 80), (padding, y_pos), (padding + plot_w, y_pos), 1)
        
        lbl_val = ratio * max_val
        lbl = font.render(f"{int(lbl_val)}", True, (120, 130, 150))
        surface.blit(lbl, (2, y_pos - 6))
        
    # Map stats history records to surface coordinates
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
    
    Inputs:
        screen: The Pygame Screen surface
        font: Font object used for rendering text
        fps: Current frame rate (float)
        gen: Current generation index (integer)
        time_rem: Time remaining in seconds (float)
        fish_rem: Fish remaining/alive (integer)
        best_fit_current: Highest fitness reached in current generation (float)
        all_time_best: All-time highest fitness recorded (float)
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
        ("Mutation Rate/Str:", f"{config.MUTATION_RATE * 100:.0f}% / {config.MUTATION_STRENGTH:.2f}", (155, 89, 182))
    ]
    
    y_offset = 10
    text_font = pygame.font.SysFont(None, 15)
    
    for label, val, color in lines:
        lbl_r = text_font.render(label, True, (170, 180, 190))
        val_r = text_font.render(val, True, color)
        hud_surface.blit(lbl_r, (15, y_offset))
        hud_surface.blit(val_r, (140, y_offset))
        y_offset += 20
        
    help_txt = text_font.render("Press [H] for controls shortcuts legend", True, (241, 196, 15))
    hud_surface.blit(help_txt, (10, panel_h - 22))
    
    screen.blit(hud_surface, (10, 10))


def draw_help_panel(screen, font, world_w):
    """
    Draws a Keyboard Shortcuts Dashboard overlay in the top-right corner.
    
    Inputs:
        screen: The Pygame Screen surface
        font: Font object used for rendering text
        world_w: Current screen width boundary (float)
    """
    panel_w = 320
    panel_h = 230
    
    help_surface = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    help_surface.fill((20, 25, 40, 230))  # Slate navy background
    pygame.draw.rect(help_surface, (241, 196, 15), (0, 0, panel_w, panel_h), 2)  # Yellow border
    
    title_font = pygame.font.SysFont(None, 15, bold=True)
    title = title_font.render("KEYBOARD CONTROLS MENU", True, (241, 196, 15))
    help_surface.blit(title, (15, 12))
    
    text_font = pygame.font.SysFont(None, 14)
    controls = [
        ("ESC", "Exit simulation"),
        ("D", "Toggle Debug Overlay (hover select details)"),
        ("G", "Toggle Live Fitness Graph"),
        ("S", "Save all-time best genome to disk"),
        ("L", "Load genome to inject Golden Champion"),
        ("E", "Export stats history to CSV"),
        ("F", "Toggle Fullscreen mode (rescales view)"),
        ("T", "Toggle Headless Fast Training mode"),
        ("R", "Toggle Simple Rendering (clean view)"),
        ("H", "Toggle Help Controls panel (this menu)")
    ]
    
    y_offset = 35
    for key, desc in controls:
        key_txt = text_font.render(f"[{key}]", True, (241, 196, 15))
        desc_txt = text_font.render(desc, True, (200, 200, 200))
        help_surface.blit(key_txt, (15, y_offset))
        help_surface.blit(desc_txt, (60, y_offset))
        y_offset += 19
        
    screen.blit(help_surface, (world_w - 330, 10))


def export_stats_csv(stats_history, filepath="generation_history.csv"):
    """
    Exports the recorded generation metrics history to a CSV file.
    
    Inputs:
        stats_history: List of dictionary records containing stats
        filepath: Save filepath destination (string, default generation_history.csv)
    """
    try:
        if not stats_history:
            logger.warning("No generation statistics to export yet.")
            return
        keys = stats_history[0].keys()
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(stats_history)
        logger.info(f"Exported stats history CSV to {filepath}")
    except Exception as e:
        logger.error(f"Failed to export stats CSV: {e}", exc_info=True)
