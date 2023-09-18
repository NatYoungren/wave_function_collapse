# Nathaniel Alden Homans Youngren
# September 14, 2023


# This simple implementation of the Wave Function Collapse algorithm is built with minimal research.
#   This is primarily an exploration of my initial non-technical understanding of the algorithm.
#   The goal is to build a general understanding of what these pattern generation algorithms are doing,
#       and then use that basis to examine more complete WFC implementations.

# Resources:
# https://robertheaton.com/2018/12/17/wavefunction-collapse-algorithm/

import numpy as np


# NOTE: Numpy arrays are ordered (height, width, depth), i.e. (y, x, z)?
#       Try to stick to this conventioned as we want to use image data.
# TODO: Reorder loops to be (y, x)/(h, w) instead of (x, y) for consistency with numpy arrays?


# Rules format:
# 0 -> Tile
# 1 -> Allowed above
# 2 -> Allowed right
# 3 -> Allowed below
# 4 -> Allowed left
# {Placed tile: [[allowed tiles above], [allowed tiles to the right], [allowed tiles below], [allowed tiles to the left]]}


class SimpleWFC:
    
    def __init__(self, h:int, w:int,
                 initial_rules:dict=None,
                 random_seed=None) -> None:
        
        self.w = w # Width of grid
        self.h = h # Height of grid
        
        # Grid of tiles
        self.grid = np.zeros((w, h))
        self.options_grid = None
        self.probability_grid = None    # NOTE: This is derivative of the options grid.
        
        # Rules determining which tiles can be placed next to each other
        self.rules = initial_rules
        
        if random_seed is not None: np.random.seed(np.int64(random_seed))
        
        self.step_count = 0
    
    
    @property
    def finished(self) -> bool:
        return not np.any(self.grid == 0)
    
    @property
    def tile_types(self) -> list:
        return list(self.rules.keys())
    
    
    def step(self) -> None:
        
        if self.rules is None: raise ValueError('Rules must be defined before stepping.')
        if self.step_count == 0: self.generate_options_grid()
        
        # If the grid contains no unresolved tiles, return
        if self.finished: return
                
        # Get the tile with the highest non-1 probability
        pos = self.select_next_tile()
        
        # Select tiletype from options
        tiletype = self.select_tiletype(*pos)
        
        print('Setting tile', pos, 'to type', tiletype, 'out of', self.options_grid[pos[0]][pos[1]])
        
        # Resolve the tile
        self.update_grid(*pos, tiletype)
        
        
        self.step_count += 1
        
        return
    
    
    def select_tiletype(self, x:int, y:int) -> int:
        """ Select and return a tiletype from the options grid.

        Args:
            x (int): X coordinate of the tile.
            y (int): Y coordinate of the tile.

        Returns:
            int: Tiletype to be placed at the given coordinates.
        """
        # TODO: Weigh choice based on how often option appeared in ruleset.
        # TODO: Weigh choice based on current representation of that tiletype in the grid compared to the ruleset.
        return np.random.choice(self.options_grid[x][y])
    
    def select_next_tile(self) -> tuple:
        """ Select and return the coordinates of the next tile to resolve.
                Tile with the highest probability is selected (least possibilities).

        Returns:
            tuple: (X, Y) coordinates of the next tile to resolve.
        """
        # Mask out all tiles which are already resolved
        uncertain_grid = np.ma.masked_where(self.grid != 0, self.probability_grid)

        # Identify the tile with the highest probability
        return_pos = np.argmax(uncertain_grid)
        
        # Return 2D coordinate of that tile
        return tuple(np.unravel_index(return_pos, self.probability_grid.shape))
    
    
    def update_grid(self, x:int, y:int, t:int) -> None: # TODO: Break the grid updating into a subfunc and use it to generate options grid?
        if t not in self.tile_types: raise ValueError(f'Tile type {t} not in ruleset.')
        if y < 0 or y >= self.h or x < 0 or x >= self.w: raise ValueError(f'({x}, {y}) is out of bounds.')
        if self.grid[x, y] != 0: raise ValueError(f'Tile ({x}, {y}) is already resolved.') # TODO: May want to allow overwriting tiles?
        
        self.grid[x, y] = t
        self.options_grid[x][y] = []
        self.probability_grid[x, y] = -1
        
        #               up        right     down      left
        for _x, _y in [[x, y-1], [x+1, y], [x, y+1], [x-1, y]]:
            if _x < 0 or _x >= self.w or _y < 0 or _y >= self.h or self.grid[_x, _y] != 0:
                continue # OOB
            else:
                self.options_grid[x][y] = self.get_tile_options(x, y)
                if not self.options_grid[x][y]:
                    self.probability_grid[x, y] = -1
                else:
                    self.probability_grid[x, y] = 1/len(self.options_grid[x][y])
        
                
    def generate_options_grid(self) -> None:
        """ Generates and stores a grid of options for each tile, and a grid of probability for each tile (1/num options).
        """
        
        probability_grid = np.zeros((self.grid.shape))
        
        # uncertain_grid = np.ma.masked_where(grid == 0, grid)
        # print(np.ma.masked_where(grid == 0, grid))
        # certain_grid = np.ma.masked_where(grid !=0 , grid)
        # print(np.ma.masked_where(grid !=0 , grid))
        
        options_grid = []

        for x in range(self.w):
            options_grid.append([])
            
            for y in range(self.h):
                options_grid[x].append([])
                
                if self.grid[x, y] != 0:
                    # If the tile is already placed, set the probability to 1 TODO: Is this correct?
                    probability_grid[x, y] = -1
                    
                else:
                    # sample = self.get_sample(x, y, grid=grid)
                    options = self.get_tile_options(x, y)
                    
                    if not options:
                        probability_grid[x, y] = -1
                        continue

                    probability_grid[x, y] = 1/len(options)
                    options_grid[x][y] = options
        
        self.probability_grid = probability_grid
        self.options_grid = options_grid
        
        # print(probability_grid)
        # print(options_grid)
        
        
    def get_tile_options(self, x:int, y:int, restrictive=False, default=0) -> list:
        """ Return a list of tiles which can be placed at a given location, based on the ruleset and surrounding tiles.

        Args:
            x (int): X coordinate of the tile.
            y (int): Y coordinate of the tile.
            restrictive (bool, optional): If True, apply more stringent rule-matching. Defaults to False. NOTE: UNIMPLEMENTED
            default (int, optional): Default tiletype to return if no options are found. Defaults to 0. NOTE: UNIMPLEMENTED
            
        Returns:
            list: List of viable tiletypes.
        """
        # TODO: Implement restrictive options, where the neighbor pattern must be 1-1 with the ruleset.
        # TODO: Implement optional rotation and mirroring of the ruleset.
        
        options = []
        
        sample = self.get_sample(x, y)
        t, r = sample[0], sample[1:] # Tile type, surrounding tiles
        
        # TODO: REVISIT ALL OF THIS JANKY GARBAGE SET INTERSECTION CODE
        num_neighbors = len(r)
        if num_neighbors % 2 != 0: raise ValueError(f'Number of neighbors ({num_neighbors}) must be even.')
        
        
        surrounding_rules = []
        for i, _r in enumerate(r):
            if _r in self.rules:
                surrounding_rules.append(self.rules[_r][(i+(num_neighbors//2))%num_neighbors])
            else:
                surrounding_rules.append([])
                
        # print(surrounding_rules)
        initial_set, remaining_rules = set(surrounding_rules[0]), surrounding_rules[1:]
        intersection = initial_set.intersection(*map(set, remaining_rules))
        # return
        options = list(intersection)

        for _t, _r in self.rules.items():   # For each tile type and its directional rules
            for i, dir_r in enumerate(_r):  # For each direction and its rules
                if r[i] not in dir_r or t :       # If the tile in that direction is not allowed, abort
                    break
            else:                           # If no tiles break rules, add the tile type to the options
                
                
                options.append(_t) # TODO: This is not weighted by number of times the example appears in the ruleset.
                
        # print(x, y, t, options)
        # TODO: FIGURE OUT A BETTER WAY TO HANDLE NO OPTION SCENARIOS
        # if not options: options.append(default)
        print(x, y, options)
        return options

        
    def generate_rules(self, rules_grid:np.ndarray, default_rule:list=None, overwrite:bool=True) -> None:
        """ Generate a dict of rules from a given grid of tile types.

        Args:
            rules_grid (np.ndarray): 2D array of tile types.
            default_rule (list, optional): All tiles will be seeded with this as a viable any-direction neighbor. Defaults to None.
                                            If None, the default rule is [0].
            overwrite (bool, optional): If True, overwrite self.rules with resulting rules dict. Defaults to True.

        Returns:
            dict: Dict of rules.
                    {Tiletype: [[allowed types above], [allowed types to the right], [allowed types below], [allowed types to the left]]}
        """
        if default_rule is None: default_rule = [0]
        
        w, h = rules_grid.shape
        rules = {}
        for x in range(w):
            for y in range(h):
                sample = self.get_sample(x, y, rules_grid)
                t, r = sample[0], sample[1:]
                ruleset = rules.get(t, [default_rule.copy() for _ in range(4)])
                
                for i, _r in enumerate(r):
                    ruleset[i].append(_r)

                rules[t] = ruleset
                
        if overwrite: self.rules = rules
        return rules
    
    
    def get_sample(self, x:int, y:int, grid=None, oob_value=-1): #TODO: Y AND X ARE FLIPPED???
        """ Get a sample of the tile and its neighbors from a given grid.

        Args:
            x (int): X coordinate of the tile.
            y (int): Y coordinate of the tile.
            grid (np.ndarray, optional): 2D array from which to extract sample. Defaults to None.
            oob_value (int, optional): Default value to use if coordinates are out of bounds. Defaults to -1.

        Returns:
            list: Returns type of tile and surrounding tiles in the order [tile, up, right, down, left].
        """
        if grid is None: grid = self.grid
        
        sample = []
        w, h = grid.shape
        
        #               xy      up        right     down      left
        for _x, _y in [[x, y], [x, y-1], [x+1, y], [x, y+1], [x-1, y]]:
            if _x < 0 or _x >= w or _y < 0 or _y >= h:
                sample.append(oob_value)
            else:
                sample.append(grid[_x, _y])
            
        return sample
        

if __name__ == '__main__':
    wfc = SimpleWFC(5, 5)
    
    rules_grid = np.array([[1, 1, 1],
                           [3, 2, 1],
                           [1, 1, 1]])
    
    rules = wfc.generate_rules(rules_grid)
    
    for k, r in rules.items():
        print(f'\n{k}')
        for r, n in zip(r, ['up', 'right', 'down', 'left']):
            print(n, r)
    wfc.generate_options_grid()
    
    while not wfc.finished:
        print('STEP', wfc.step_count,'\n\n')
        wfc.step()
        print(wfc.grid)
