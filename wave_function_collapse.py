# Nathaniel Alden Homans Youngren
# September 14, 2023


# This simple implementation of the Wave Function Collapse algorithm is built with minimal research.
#   This is primarily an exploration of my initial non-technical understanding of the algorithm.
#   The goal is to build a general understanding of what these pattern generation algorithms are doing,
#       and then use that basis to examine more complete WFC implementations.

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
                 initial_rules:dict=None) -> None:
        
        self.h = h # Height of grid
        self.w = w # Width of grid
        
        # Grid of tiles
        self.grid = np.zeros((w, h))
        
        # Rules determining which tiles can be placed next to each other
        self.rules = initial_rules
        if self.rules is None: self.rules = {}
        
        self.step_count = 0
    
    
    @property
    def finished(self) -> bool:
        return not np.any(self.grid == 0)
    
    @property
    def tile_types(self) -> list:
        return list(self.rules.keys())
    
    
    def step(self) -> None:
        
        # If the grid contains no unresolved tiles, return
        if self.finished: return
        
        # Get the probability grid
        
        # Get the tile with the highest non-1 probability
        
        # Resolve the tile
        self.step_count += 1
        return
        
    
    def get_probability_grid(self, grid=None) -> np.ndarray:
        if grid is None: grid = self.grid
        # if rules is None: rules = self.rules
        
        grid = np.array(grid)
        h, w = grid.shape
        
        probability_grid = np.zeros((h, w))
        
        # uncertain_grid = np.ma.masked_where(grid == 0, grid)
        print(np.ma.masked_where(grid == 0, grid))
        # certain_grid = np.ma.masked_where(grid !=0 , grid)
        print(np.ma.masked_where(grid !=0 , grid))
        
        print(probability_grid)
        
        options_grid = []

        for y in range(h):
            options_grid.append([])
            
            for x in range(w):
                options_grid[y].append([])
                
                if grid[y, x] != 0:
                    # If the tile is already placed, set the probability to 1 TODO: Is this correct?
                    probability_grid[y, x] = -1
                    
                else:
                    # sample = self.get_sample(x, y, grid=grid)
                    options = self.get_tile_options(x, y, grid=grid)
                    
                    if not options:
                        probability_grid[y, x] = -1
                        continue
                    
                    probability_grid[y, x] = 1/len(options)
                    options_grid[y][x] = options
                    
        print(probability_grid)
        print(options_grid)
        return probability_grid

                    # probability_grid[y, x] = self.get_probability(sample, rules)
        
        
    def get_tile_options(self, x:int, y:int, grid=None, restrictive=False) -> list:
        if grid is None: grid = self.grid
        options = []
        
        sample = self.get_sample(x, y, grid=grid)
        t, r = sample[0], sample[1:]
        
        
        
        for _t, _r in self.rules.items():
            for i, dir_r in enumerate(_r):
                if r[i] not in dir_r:
                    break
            else:
                options.append(_t) # NOTE: This is not weighted by number of times the example appears in the ruleset.
                
        print(x, y, t, options)
        
        return options
                # if _r[i] == t:
                #     options.append(_t)

    
    # def get_probability(self, sample:list) -> float: # , rules:dict) -> float:
    #     if len(sample) != 5: raise ValueError('Sample must be of length 5.')
        
    #     t, r = sample[0], sample[1:]
        
    #     ruleset = self.rules.get(t, None)
    #     if ruleset is None: return -1 # 0?
        
    #     tile_types = self.tile_types
        
        
        
        
    #     for 
        
        
    
    
        
    def generate_rules(self, rules_grid:np.ndarray, default_rule:list=None, overwrite:bool=True) -> None:
        if default_rule is None: default_rule = [0]
        
        h, w = rules_grid.shape
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
        
        if grid is None: grid = self.grid
        
        sample = []
        h, w = grid.shape
        
        #               xy      up        right     down      left
        for _x, _y in [[x, y], [x, y-1], [x+1, y], [x, y+1], [x-1, y]]:
            if _x < 0 or _x >= w or _y < 0 or _y >= h:
                sample.append(oob_value)
            else:
                sample.append(grid[_y, _x])
            
        return sample
        

if __name__ == '__main__':
    wfc = SimpleWFC(5, 6)
    
    rules_grid = np.array([[1, 1, 1],
                           [3, 2, 1],
                           [1, 1, 1]])
    
    rules = wfc.generate_rules(rules_grid)
    
    for k, r in rules.items():
        print(f'\n{k}')
        for r, n in zip(r, ['up', 'right', 'down', 'left']):
            print(n, r)
    wfc.get_probability_grid()
    
