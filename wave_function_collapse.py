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
                 initial_rules:dict=None) -> None:
        
        self.h = h # Height of grid
        self.w = w # Width of grid
        
        # Grid of tiles
        self.grid = np.zeros((w, h))
        
        # Rules determining which tiles can be placed next to each other
        self.rules = initial_rules
        if self.rules is None: self.rules = {}
    
    @property
    def tile_types(self) -> list:
        return list(self.rules.keys())
    
    def get_probability_grid(self, grid=None) -> np.ndarray:# , rules=None) -> np.ndarray:
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
        
        for x in range(w):
            for y in range(h):
                if grid[y, x] != 0:
                    # If the tile is already placed, set the probability to 1 TODO: Is this correct?
                    probability_grid[y, x] = 1
                else:
                    sample = self.sample_grid(y, x, grid=grid)
                    probability_grid[y, x] = self.get_probability(sample, rules)
        
        
    # def get_probability(self, sample:list) -> float: # , rules:dict) -> float:
    #     if len(sample) != 5: raise ValueError('Sample must be of length 5.')
        
    #     t, r = sample[0], sample[1:]
        
    #     ruleset = self.rules.get(t, None)
    #     if ruleset is None: return -1 # 0?
        
    #     for 
        
        
    
    
        
    def generate_rules(self, rules_grid:np.ndarray, default_rule:list=None, overwrite:bool=True) -> None:
        if default_rule is None: default_rule = [0]
        
        h, w = rules_grid.shape
        rules = {}
        for x in range(w):
            for y in range(h):
                sample = self.sample_grid(y, x, rules_grid)
                
                t, r = sample[0], sample[1:]
                ruleset = rules.get(t, [default_rule.copy() for _ in range(4)])
                
                for i, _r in enumerate(r):
                    ruleset[i].append(_r)

                rules[t] = ruleset
                
        if overwrite: self.rules = rules
        return rules
    
    
    def sample_grid(self, x:int, y:int, grid=None, oob_value=-1):
        
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
    wfc = WaveFunctionCollapse(5, 5)
    
    rules_grid = np.array([[1, 1, 1],
                           [3, 2, 1],
                           [1, 1, 1]])
    
    rules = wfc.generate_rules(rules_grid)
    wfc.get_probability_grid(rules=rules)
    
    # for k, r in rules.items():
    #     print(f'\n{k}')
    #     for r, n in zip(r, ['up', 'right', 'down', 'left']):
    #         print(n, r)