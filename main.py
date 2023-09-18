# Nathaniel Alden Homans Youngren
# September 16, 2023

import time
import numpy as np
import pygame as pg

from wave_function_collapse import SimpleWFC
import display_vars as dv


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                   # INSTRUCTIONS: #                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 1. Set up the rules grid with 0-9 and left/right click.

# 2. Press spacebar to generate rules and begin the wave function collapse.
#       (Continue to step with spacebar if manual control is enabled, toggle to manual with 'm' key)

# 5. Press 'r' to revise your rules, or escape to quit. (Shift+'r' to reset rules fully)
#       (Additional controls are listed below)

# TODO: Implement file loading/saving for rules and grid state 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                   # ALL CONTROLS: #                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# m1 -> Set cells to selected type
# m2 -> Reset cells to default type

# '0' ... '9' -> Change type of placed cells to 0 - 9


# 't' -> Toggle text display
# 'y' -> Toggle text content (coords/type)

# 'm' -> Toggle manual control

# ' ' -> Start simulation, or step if manual control is enabled
# 'r' -> Reset pathfinding
# 'R' -> Reset completely
# ESC -> Quit


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                        # SETUP: #                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



# WFC CELL VARS
GRID_W, GRID_H = 10, 10           # Grid size
DEFAULT_TYPE = 1                # Default cost of cells
CELL_KEYS = {pg.K_0: 0,         # Dict mapping keypress to cell type
             pg.K_1: 1,
             pg.K_2: 2,
             pg.K_3: 3,
             pg.K_4: 4,
             pg.K_5: 5,
             pg.K_6: 6,
             pg.K_7: 7,
             pg.K_8: 8,
             pg.K_9: 9}

# RULE VARS
RULES_GRID_W, RULES_GRID_H = 5, 5 # Width and height of rules grid
RULES_GRID = np.full((RULES_GRID_W, RULES_GRID_H), fill_value=DEFAULT_TYPE) # Grid of rules

# STEP SPEED VARS
STEPS_PER_SECOND = 100       # Number of steps per second (if manual control is disabled)
STEPS_PER_FRAME = 0         # Maximum steps per frame update (if < 1, no limit)

# Dict to track state information, avoids individual global variables or passing/returning many arguments.
STATE_DICT =   {'manual_control': False,    # If true, manual control is enabled (If false, auto-step is enabled)
                
                'cell_type': 1,            # Cost of cells placed with left click
                
                'show_text': True,          # If true, show text on mouseover
                'text_content': 0,          # 0 = coords/type
                
                # Internal state vars
                'running': True,            # Main loop control
                'searching': False,         # Wfc loop control, starts after rules are set
                'resetting': 2,             # Flag to reset grid (1), or grid and rules (2)
                
                'temp_portal': None,        # Temp var to store portal start position during portal creation
                }

# DISPLAY VARS
SQUARE_CELLS = True         # If true, cells will always be square
LOCK_RULES_SIZE = True      # If true, rule cell size will match grid cell size

# DISPLAY CONSTANTS
CELL_W, CELL_H = dv.SCREEN_W / GRID_W, dv.SCREEN_H / GRID_H
if SQUARE_CELLS: CELL_W = CELL_H = min(CELL_W, CELL_H)
ORIGIN_X = (dv.SCREEN_W - CELL_W * GRID_W) / 2
ORIGIN_Y = (dv.SCREEN_H - CELL_H * GRID_H) / 2

# RULES DISPLAY CONSTANTS (TODO: Rename to R_ prefix)
if LOCK_RULES_SIZE: RULES_CELL_W, RULES_CELL_H = CELL_W, CELL_H
else: RULES_CELL_W, RULES_CELL_H = dv.SCREEN_W / RULES_GRID_W, dv.SCREEN_H / RULES_GRID_H
    
if SQUARE_CELLS: RULES_CELL_W = RULES_CELL_H = min(RULES_CELL_W, RULES_CELL_H)
RULES_ORIGIN_X = (dv.SCREEN_W - RULES_CELL_W * RULES_GRID_W) / 2
RULES_ORIGIN_Y = (dv.SCREEN_H - RULES_CELL_H * RULES_GRID_H) / 2
    


def main():
    # Var to store the current pathfinding simulation
    sim = None
    
    # Initialize pygame window
    pg.init()
    screen = pg.display.set_mode((dv.SCREEN_W, dv.SCREEN_H))

    # Initialize font
    pg.font.init()
    font = pg.font.Font(dv.TEXT_FONT, dv.TEXT_SIZE)

    # Timer to auto-step the simulation
    pg.time.set_timer(pg.USEREVENT+1, 1000//STEPS_PER_SECOND)

    # Main loop
    while STATE_DICT['running']:
        
        # Reset the simulation if flagged
        if STATE_DICT['resetting']:
            sim = SimpleWFC(h=GRID_H, w=GRID_W, random_seed=time.time())
            if STATE_DICT['resetting'] == 2:
                RULES_GRID[:] = DEFAULT_TYPE
            STATE_DICT['searching'] = False
            STATE_DICT['resetting'] = False
        
        # Update window title
        pg.display.set_caption(f'Simple Wave Function Collapse: steps: {sim.step_count} ~ avg confusion: {np.nan}')
        
        # Handle input events
        parse_events(sim)
        
        # Draw current state of pathfinding sim
        draw_state(screen, sim)

        # Draw text at mouse position
        if STATE_DICT['show_text']:
            draw_mouse_text(screen, font, sim)
        
        # Update display
        pg.display.flip()
    
    
def parse_events(sim: SimpleWFC):
    """ Handle pygame events and update the simulation/visualization accordingly.
        Simulation is stepped inside this function, either manually or via timer.

    Args:
        sim (SimpleWFC): Wave function collapse simulation to update.
    """
    
    step_count = 0 # Used to track/limit the number of steps per frame
    
    # Handle input events
    for event in pg.event.get():
        
        # Handle quit event
        if event.type == pg.QUIT:
            STATE_DICT['running'] = False
        
        # If searching, and manual control is disabled, step the simulation on a timer
        elif event.type == pg.USEREVENT+1:
            if not 0 < STEPS_PER_FRAME <= step_count:
                if not sim.finished and not STATE_DICT['manual_control'] and STATE_DICT['searching']:
                    _ = sim.step()
                    step_count += 1
                    
        # On left click, set start/end if they are not yet set
        elif event.type == pg.MOUSEBUTTONDOWN:
            clicked_cell = get_cell(event.pos)
            if not STATE_DICT['searching']:
                RULES_GRID[clicked_cell[0], clicked_cell[1]] = STATE_DICT['cell_type']
            
        # Handle keypresses
        elif event.type == pg.KEYDOWN:
            
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            # Escape key quits
            if event.key == pg.K_ESCAPE:
                STATE_DICT['running'] = False
            
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            # R key resets the simulation at the beginning of the next main loop
            elif event.key == pg.K_r:
                
                STATE_DICT['resetting'] = 1
                if event.mod == 1: STATE_DICT['resetting'] = 2
            
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            # M key toggles manual control
            elif event.key == pg.K_m:
                STATE_DICT['manual_control'] = not STATE_DICT['manual_control']
                print('Manual control:', STATE_DICT['manual_control'])

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            # 't' Key toggles text display
            elif event.key == pg.K_t:
                STATE_DICT['show_text'] = not STATE_DICT['show_text']
                print('Show text:', STATE_DICT['show_text'])
            
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            # 'y' Key toggles text content
            elif event.key == pg.K_y:
                STATE_DICT['text_content'] = (STATE_DICT['text_content'] + 1) % 3

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            # Check dict of predefined keys to determine cost change (default: 0-9 Keys set cell type)
            elif event.key in CELL_KEYS:
                STATE_DICT['cell_type'] = CELL_KEYS[event.key]
                print('Cell type:', STATE_DICT['cell_type'])
            
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # Spacebar begins the simulation (also steps if manual control is enabled)
            elif event.key == pg.K_SPACE:                
                if not STATE_DICT['searching']:
                    print('TODO: SET RULES NOW')
                    sim.generate_rules(RULES_GRID)
                    STATE_DICT['searching'] = True
                
                # If manual control is enabled, step the simulation
                if STATE_DICT['manual_control'] and not sim.finished:
                    if not 0 < STEPS_PER_FRAME <= step_count:
                        sim.step()
                        step_count += 1
                    
                    if sim.finished:
                        print(f'Finished in: {sim.step_count} steps.')
            
        
    # Adjust any moused-over cell costs if holding left or right click
    #   Only allow cost changes before searching begins
    if not STATE_DICT['searching']:
        try:
            clicks = pg.mouse.get_pressed()
            if any(clicks):
                clicked_cell = get_cell(pg.mouse.get_pos())
                
                if clicks[0]: # Left click changes cell type
                    RULES_GRID[clicked_cell[0], clicked_cell[1]] = STATE_DICT['cell_type']
                    
                elif clicks[2]: # Right click resets cell to default
                    RULES_GRID[clicked_cell[0], clicked_cell[1]] = DEFAULT_TYPE

        except AttributeError:
            pass



def draw_state(surf, sim):
    """ Draw the current state of the pathfinding simulation to the given surface.
            Renders the contents of each cell, minimizing draw calls at the expense of readability and checks-per-cell.
            Portals are drawn as paired triangles, with direction indicating entrance/exit.

    Args:
        surf (pygame.Surface): Surface to draw to (presumably the screen).
        sim (A_Star_Portals): Simulation from which to get state information.
    """
    surf.fill(dv.BG_COLOR) # Used for grid lines between cells and empty border space.

    # # #
    # Draw cell grid
    
    # Width and height of each cell, minus the border width
    if STATE_DICT['searching']:
        cell_w = CELL_W # If searching, draw the full grid of cells    
        cell_h = CELL_H
        grid_w = GRID_W
        grid_h = GRID_H
        origin_x = ORIGIN_X
        origin_y = ORIGIN_Y
    else:
        cell_w = CELL_W # If not searching, draw the rules grid
        cell_h = CELL_H
        grid_w = RULES_GRID_W
        grid_h = RULES_GRID_H
        origin_x = RULES_ORIGIN_X
        origin_y = RULES_ORIGIN_Y
        
    width  = cell_w - dv.BORDER_PX*2
    height = cell_h - dv.BORDER_PX*2

    # Draw the cell type of each cell
    for w in range(grid_w):
        for h in range(grid_h):
            x = origin_x + cell_w * w           # Top left corner of the cell
            y = origin_y + cell_h * h   
            _x = x + dv.BORDER_PX               # Offset corner by border width
            _y = y + dv.BORDER_PX       
            rect_vars = (_x, _y, width, height) # Rect vars for pg.draw.rect
            
            if STATE_DICT['searching']:
                cell_val = sim.grid[w, h]
            else:
                cell_val = RULES_GRID[w, h]
            
            # Change to draw a blend of all possible colors for unresolved cells
            pg.draw.rect(surf, dv.COLOR_DICT.get(cell_val, dv.DEFAULT_COLOR), rect_vars) 
   

def draw_mouse_text(surf, font, sim):
    """ Draw text at the mouse position, indicating the cell coordinates or heuristics.

    Args:
        surf (pygame.Surface): Surface to draw text to.
        font (pygame.font.Font): Font to use for text.
        sim (A_Star_Portals): Pathfinding simulation to get heuristics from.
    """
    mouse_pos = pg.mouse.get_pos()
    clicked_cell = get_cell(mouse_pos)
    text_pos = np.add(mouse_pos, dv.TEXT_OFFSET)

    if STATE_DICT['text_content'] == 0: # Show cell coordinates and type
        text = f'{clicked_cell}'
        if STATE_DICT['searching']:
            text += f'  {sim.grid[clicked_cell]}'
        else:
            text += f'  {RULES_GRID[clicked_cell]}'
            
    draw_text(surf, text, text_pos, font, dv.TEXT_COLOR, dv.TEXT_ALPHA)



def draw_text(surf, text, pos, font, color, alpha):
    """ Blit text onto the screen at the given position.

    Args:
        surf (pygame.Surface): Surface to draw text to.
        text (str): Text to draw.
        pos (int, int): Coords at which to center the text.
        font (pygame.font.Font): Font to use for text.
        color (int, int, int): RGB color of text.
        alpha (int): Alpha value of text from 0 to 255.
    """
    text_surface = font.render(text, True, color)
    text_surface.set_alpha(alpha)
    rect = text_surface.get_rect()
    rect.center = pos
    surf.blit(text_surface, rect)


def get_cell(pos):
    """ Convert pixel coordinates into cell coordinates.
            Always returns a valid cell coordinate, even if the pixel position is outside the grid.

    Args:
        pos (int, int): Pixel position, presumably from mouse.get_pos()

    Returns:
        (int, int): Cell coordinates, clamped to grid size.
    """
    pos = np.array(pos)
    
    if STATE_DICT['searching']:
        pos[0] = (pos[0] - ORIGIN_X) / CELL_W
        pos[1] = (pos[1] - ORIGIN_Y) / CELL_H
        np.clip(pos[:1], 0, GRID_W-1, out=pos[:1])
        np.clip(pos[1:], 0, GRID_H-1, out=pos[1:])
    else:
        pos[0] = (pos[0] - RULES_ORIGIN_X) / RULES_CELL_W
        pos[1] = (pos[1] - RULES_ORIGIN_Y) / RULES_CELL_H
        np.clip(pos[:1], 0, RULES_GRID_W-1, out=pos[:1])
        np.clip(pos[1:], 0, RULES_GRID_H-1, out=pos[1:])

    return tuple(pos.astype(int))
    
if __name__ == '__main__':
    main()
