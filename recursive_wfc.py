# Nat Youngren
# September 16, 2023
#

# Adapted from previous work on recursive sudoku solving.

import numpy as np
from copy import deepcopy
import time
import timeit



NUM_TILE_TYPES = 9


class RecursiveWFC:
    
    def __init__(self, w:int, h:int,
                 neighbor_range:int=1,
                 allow_diagonal:bool=False,
                 looping:bool=False,
                 verbose=False):
        self.w = w
        self.h = h
        
        self.neighbor_range = neighbor_range
        self.allow_diagonal = allow_diagonal
        self.looping = looping
        
        # Weights generated from rules grid, stored as a list of (tiletype, weight) tuples
        self.weights = None
        # Rules generated from rules grid, stored as a list of (tiletype, neighbor tiletype, x_offset, y_offset) tuples
        self.rules = None
        
        # Grid collapsed using rules/weights
        self.prob_field = None

        self.verbose = verbose

    def process_rules(self, rules_grid:np.ndarray):
        w, h = rules_grid.shape
        
        
        
        
        rules = []
        for x in range(w):
            for y in range(h):
                t = rules_grid[x, y]
                
                for _x, _y in self.get_neighbor_indexes(x, y, grid=rules_grid):
                    
                    rules.append([t, rules_grid[_x, _y], x-_x, y-_y]) # TODO: Change self.get_neighbor_indexes to just return the offset
        
        self.rules = np.array(sorted(rules, key=lambda l: l[0]))
        
        self.tiletypes = sorted(np.unique(rules_grid))
        self.weights = [np.count_nonzero(rules_grid==tt) for tt in self.tiletypes]
        
        if self.verbose:
            print('Tile types-weights:', list(zip(self.tiletypes, self.weights)))
            print('Rules:', self.rules)

    def get_neighbor_indexes(self, x:int, y:int, grid=None):
        if grid is None: grid = self.grid
        w, h = grid.shape[:2]
        
        if self.verbose: print('Getting neighbor indexes for:', x, y)
        
        neighbors = []
        for i in range(-self.neighbor_range, self.neighbor_range+1):
            for j in range(-self.neighbor_range, self.neighbor_range+1):
                
                if i == 0 and j == 0: # Skip self
                    continue
                
                xx = x + i
                yy = y + j
                
                if not self.allow_diagonal and i != 0 and j != 0: # Skip diagonals
                    if self.verbose: print(' > Skipping diagonal neighbor:', xx, yy)
                    continue
                    
                if self.looping: # Loop at the edges of the grid
                    xx %= w
                    yy %= h
                elif xx < 0 or xx >= w or yy < 0 or yy >= h:
                    if self.verbose: print(' > Skipping OOB neighbor:', xx, yy)
                    continue
                
                neighbors.append((xx, yy))
                
        return neighbors


    def generate_probability_field(self):
        
        puzzle = np.array(puzzle)
        prob_field = np.zeros((9, 9, 9))
        for x, row in enumerate(puzzle):
            for y, col in enumerate(puzzle.T):
                
                if puzzle[x][y] != 0:
                    prob_field[x][y][puzzle[x][y]-1] = 1
                    continue
                
                cell = get_cell(puzzle, x, y)
                opts = get_options(row, col, cell)
                for i in opts:
                    prob_field[x][y][i-1] = 1 # / len(opts)

        return prob_field


# # # # # # #
# Utilities
#

def get_cell(puzzle, x, y): # TODO: Faster way to do this?
    x = x // 3
    y = y // 3
    rows = puzzle[x*3:x*3+3]
    cols = rows[:,y*3:y*3+3]
    cell = cols.flatten()
    return cell

def get_options(row, col, cell): # TODO: Faster way to do this?
    taken = set(np.concatenate((row, col, cell)).flatten())
    return [i for i in range(1, 10) if i not in taken]




def collapse_probability_field(prob_field: np.array, x: int, y: int, i: int): # Make copy bool param
    pf = deepcopy(prob_field)
    pf = perpetuate_collapse(pf, x, y, i)
    pf[x,y,:] = 0
    pf[x][y][i] = 1
    
    return pf

def perpetuate_collapse(prob_field: np.array, x: int, y: int, i: int):
    prob_field[x,:,i] = 0
    prob_field[:,y,i] = 0
    xx = x // 3
    yy = y // 3
    prob_field[xx*3:xx*3+3, yy*3:yy*3+3, i] = 0

    return prob_field


def prob_field_to_puzzle(prob_field: np.array):
    out_puzzle = np.zeros((9, 9))
    for x, row in enumerate(prob_field):
        for y, col in enumerate(prob_field.T):
            if prob_field[x][y].sum() == 1:
                out_puzzle[x][y] = np.argmax(prob_field[x][y]) + 1
            elif prob_field[x][y].sum() > 1:
                # print('UNRESOLVED', x, y, prob_field[x][y])
                out_puzzle[x][y] = -1
            # elif prob_field[x][y].sum() == 0:
                # print('OVERRESOLVED', x, y, prob_field[x][y])
    return out_puzzle



# # # # # # #
# Solvers
#

#
# Recursive Ripple Solver
def ripple_solve(prob_field: np.array, resolved=None, verbose=False):
    if resolved is None:
        resolved = np.zeros((9, 9))
    prev_sum = 0
    while True:
        resolution_map = prob_field.sum(axis=2)
        # print(resolution_map, resolution_map.sum())
        if not resolution_map.all():
            return None
        
        new_sum = resolution_map.sum()
        if new_sum == 81:
            break
        
        # print(np.where(resolution_map == resolved))
        
        if prev_sum != new_sum:
            resolved_indices = np.argwhere(resolution_map == 1)
            for x, y in resolved_indices:
                if resolved[x][y]:
                    continue
                resolved[x][y] = 1
                prob_field = collapse_probability_field(prob_field, x, y, np.argmax(prob_field[x][y]))
        else:
            # v = np.argmin(resolution_map[resolution_map > 1])
            # x = v // 9
            # y = v % 9
            
            unresolved_indices = np.argwhere(resolution_map > 1)
            x, y = unresolved_indices[np.argmin(resolution_map > 1)]            
            for i in np.where(prob_field[x][y])[0]:
                # print('recursive')
                r = ripple_solve(collapse_probability_field(prob_field, x, y, i), deepcopy(resolved), verbose=verbose)# resolved=resolved, verbose=verbose)
                if r is not None:
                    return r
            return None
        
        prev_sum = new_sum
        
    return prob_field

# # # # # # #
# Evaluation
#

def evaluate(puzzles, solver, iterations=1):
    # for name, puzzle in puzzles.items():
    print('\n Solving blank\n')
    # puzzle = generate_probability_field(puzzle)
    
    puzzle = np.ones((9, 9, 9))
    
    t = timeit.timeit(lambda: solver(puzzle), number=iterations)
    
    start_time = time.time()
    solution = solver(puzzle)
    print(solution)
    end_time = time.time()
    
    solved_puzzle = prob_field_to_puzzle(solution)
    unsolved_nodes = np.count_nonzero(solved_puzzle == 0)
    print(solved_puzzle)
    print(f'Average time: {t / iterations} seconds.')
    print(f'Finished in {end_time - start_time} seconds with {unsolved_nodes} unsolved nodes.')
    
    
if __name__ == '__main__':
    r_wfc = RecursiveWFC(4, 4, verbose=True)
    rules_grid = np.array([[1, 1, 1],
                           [3, 2, 1],
                           [1, 1, 1]])
    
    r_wfc.process_rules(rules_grid=rules_grid)
    
    
    
# prob_field = generate_probability_field(puzzles['evil2'])

# print(prob_field_to_puzzle(ripple_solve_blank(prob_field, verbose=True)))