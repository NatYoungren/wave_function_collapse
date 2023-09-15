# Nathaniel Alden Homans Youngren
# September 14, 2023

import numpy as np
# import random

# NOTE: Numpy arrays are ordered (height, width, depth), i.e. (y, x, z)?
#       Try to stick to this conventioned as we want to use image data.


# Rules format:
# 0 -> Tile
# 1 -> Allowed above
# 2 -> Allowed right
# 3 -> Allowed below
# 4 -> Allowed left
# {Placed tile: [[allowed tiles above], [allowed tiles to the right], [allowed tiles below], [allowed tiles to the left]]}

class WaveFunctionCollapse:
    
    def __init__(self, h:int, w:int,
                #  tile_set:np.ndarray,
                 initial_grid:np.ndarray=None, # TODO: Remove some params.
                 initial_rules:dict=None) -> None:
        
        self.h = h
        self.w = w
        
        # Grid of tiles
        if initial_grid is not None:
            self.grid = initial_grid
        else:
            self.grid = np.zeros((w, h))
        
        # Rules determining which tiles can be placed next to each other
        if initial_rules is not None:
            self.rules = initial_rules
        else:
            self.rules = {}
    
    
    def generate_rules(self, rules_grid:np.ndarray) -> None:
        h, w = rules_grid.shape
        rules = {}
        for x in range(w):
            for y in range(h):
                sample = self.sample_grid(y, x, rules_grid)
                r = np.array(sample[1:]).reshape(4, 1)
                rules[sample[0]] = np.hstack((rules.get(sample[0], np.zeros((4, 1))), r))
                
        return rules
    
    def sample_grid(self, x:int, y:int, grid=None, oob_value=-1):
        
        if grid is None:
            grid = self.grid
            
        sample = []
        h, w = grid.shape
        for _x, _y in [[x, y], [x, y+1], [x+1, y], [x, y-1], [x-1, y]]:
            if _x < 0 or _x >= w or _y < 0 or _y >= h:
                sample.append(oob_value)
            else:
                sample.append(grid[_y, _x])
            
        return sample
        

if __name__ == '__main__':
    wfc = WaveFunctionCollapse(5, 5)
    
    rules_grid = np.array([[1, 1, 1],
                           [3, 2, 1],
                           [1, 1, 1]])
    rules = wfc.generate_rules(rules_grid)
    for k, r in rules.items():
        print(f'\n{k}')
        for r, n in zip(r, ['up', 'right', 'down', 'left']):
            print(n, r)