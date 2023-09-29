# Nat Youngren
# September 16, 2023
#

# Adapted from previous work on recursive sudoku solving.

import numpy as np
from copy import deepcopy
import time
import timeit


class RecursiveWFC:
    
    def __init__(self, w:int, h:int,
                 rules_grid:np.ndarray,
                 neighbor_range:int=1,
                 diagonal_neighbors:bool=False,
                 looping_neighbors:bool=False,
                 verbose=False):
        
        self.verbose = verbose
        
        # Size of the generated grid
        self.w = w
        self.h = h

        # Variables used during rules generation
        self.neighbor_range = neighbor_range
        self.diagonal_neighbors = diagonal_neighbors
        self.looping_neighbors = looping_neighbors
        
        # Store the initial image/grid that we generate the rules from
        self.rules_grid = rules_grid
        
        # List of all types of tiles present in rules_grid, this is our entire palette of tiles
        self.tiletypes = sorted(np.unique(rules_grid))
        
        # Frequency of each tiletype in rules grid, stored as a list of (tiletype, weight) tuples
        self.weights = [np.count_nonzero(rules_grid==tt) for tt in self.tiletypes]

        # Rules generated from rules grid, stored as an np.ndarray of (tiletype, neighbor tiletype, x_offset, y_offset)
        self.rules = self.generate_rules(rules_grid)
        
        # Grid to be collapsed using rules/weights
        self.prob_field = self.generate_probability_field()
        
        # Grid which is repeatedly overwritten during the WFC algorithm, used to visualize the algorithm
        self.working_field = deepcopy(self.prob_field)
        
        
        if self.verbose:
            print('Width:', self.w, 'Height:', self.h)
            print('Tile types:', self.tiletypes)
            print('Tile types-weights:', list(zip(self.tiletypes, self.weights)))
            print('Rules:', self.rules)
            

    def generate_rules(self, rules_grid:np.ndarray):
        """ Reads a rules grid and generates the rules and weights for the WFC algorithm.

        Args:
            rules_grid (np.ndarray): Grid of tiles which demonstrate all initial rules.

        Returns:
            list(np.ndarray): List of arrays of rules, each rule as: (tiletype, neighbor tiletype, x_offset, y_offset).
                                One array of rules for each tiletype. # TODO: Technically the stored tiletype is redundant.
        """
        w, h = rules_grid.shape
        

        rules = [[] for _ in self.tiletypes]
        for x in range(w):
            for y in range(h):
                t = rules_grid[x, y]
                
                for _x, _y in self.get_neighbor_indexes(x, y, grid=rules_grid):
                    rules.append([t, rules_grid[_x, _y], x-_x, y-_y]) # FIXME: Change self.get_neighbor_indexes to just return the offset
        
        # TODO: If rotation is allowed, add a rotated version of each rule here?
        # TODO: If reflection is allowed add a reflected version of each rule here?
        # NOTE: We could always calculate the rotated/reflected version of each rule on the fly, but that would require more logic.
        
        # Sort and return list of rules
        return np.array(sorted(rules, key=lambda l: l[0]))
    
    
    def generate_rules_flat(self, rules_grid:np.ndarray):
        """ Reads a rules grid and generates the rules and weights for the WFC algorithm.

        Args:
            rules_grid (np.ndarray): Grid of tiles which demonstrate all initial rules.

        Returns:
            np.ndarray: Array of rules, each rule as: (tiletype, neighbor tiletype, x_offset, y_offset).
        """
        w, h = rules_grid.shape
        

        rules = []
        for x in range(w):
            for y in range(h):
                t = rules_grid[x, y]
                
                for _x, _y in self.get_neighbor_indexes(x, y, grid=rules_grid):
                    rules.append([t, rules_grid[_x, _y], x-_x, y-_y]) # FIXME: Change self.get_neighbor_indexes to just return the offset
        
        # Sort and return list of rules
        return np.array(sorted(rules, key=lambda l: l[0]))


    def get_neighbor_indexes(self, x:int, y:int, grid=None):
        """ Returns the indexes of all tiles which are considered neighbors to x, y.
                The state of self.allow_diagonal and self.looping_neighbors are considered.
                If self.diagonal_neighbors is False, then non-orthoganal neighbors will be skipped.
                If self.looping_neighbors is True, then neighbors which are out of bounds will be looped to the other side of the grid.
        

        Args:
            x (int): X coordinate of tile to get neighbors for.
            y (int): Y Coordinate of tile to get neighbors for.
            grid (np.ndarray, optional): The grid whose first 2 dimensions are being considered. Defaults to None.
                                            If None, uses self.prob_field.

        Returns:
            list: List of (x, y) tuples which are considered neighbors to x, y.
                    Order will not be random but indexes may be skipped depending on class params.
        """
        if grid is None: grid = self.prob_field
        w, h = grid.shape[:2]
        
        if self.verbose: print('Getting neighbor indexes for:', x, y)
        
        neighbors = []
        for i in range(-self.neighbor_range, self.neighbor_range+1):
            for j in range(-self.neighbor_range, self.neighbor_range+1):
                
                if i == 0 and j == 0: # Skip self
                    continue
                
                xx = x + i # Coordinates of neighbor
                yy = y + j
                
                if not self.diagonal_neighbors and i != 0 and j != 0: # Skip diagonals
                    if self.verbose: print(' > Skipping diagonal neighbor:', xx, yy)
                    continue
                    
                if self.looping_neighbors: # If looping neighbors is allowed, wrap indexes beyond the edges
                    xx %= w
                    yy %= h
                    
                elif xx < 0 or xx >= w or yy < 0 or yy >= h: # If looping neighbors is not allowed, skip out of bounds neighbors
                    if self.verbose: print(' > Skipping OOB neighbor:', xx, yy)
                    continue
                
                neighbors.append((xx, yy)) # Store neighbors that pass all checks
                
        return neighbors


    def generate_probability_field(self):
        
        self.prob_field = np.zeros((self.w, self.h, len(self.tiletypes)))
        
        for x in range(self.w):
            for y in range(self.h):
                
                # neighbors = self.get_neighbor_indexes(x, y)
                for i, tt in enumerate(self.tiletypes):
                    self.prob_field[x, y, i] =  self.passes_rules(x, y, tt, self.prob_field)
                    # self.prob_field[x, y, tt] = self.weights[tt]
        
        
    def passes_rules(self, x:int, y:int, tiletype:int, grid:np.ndarray):
        """ Returns True if setting the given tiletype at x, y in the grid does not break any relational rules.
                Rules are stored in self.rules.
        Args:
            x (int): X coordinate of tile to check.
            y (int): Y coordinate of tile to check.
            tiletype (int): Tiletype to check.
            grid (np.ndarray): The probability_field to check against.
                                    # TODO: Consider having a None default to use self.working_field
                                    # TODO: Consider parameterizing rules, making this a static method

        Returns:
            bool: True if the given tiletype at x, y does not break rules.
        """
        
        for rule in self.rules:# np.where(self.rules[:, 0] == tiletype, self.rules):
            print(x, y, tiletype, rule)
            xx = x + rule[2]
            yy = y + rule[3]
            
            # if xx < 0 or xx >= self.w or yy < 0 or yy >= self.h:



    # THESE ARE THE CRITICAL RIPPLE SOLVE FUNCTIONS. Consider making them external to the class and use njit
    def collapse_probability_field(self, prob_field: np.ndarray, x: int, y: int, i: int): # Make copy bool param
        pf = prob_field.copy
        pf = self.perpetuate_collapse(pf, x, y, i)
        pf[x,y,:] = 0
        pf[x][y][i] = 1
        
        return pf

    def perpetuate_collapse(self, prob_field: np.ndarray, x: int, y: int, i: int):
        prob_field[x,:,i] = 0
        prob_field[:,y,i] = 0
        xx = x // 3
        yy = y // 3
        prob_field[xx*3:xx*3+3, yy*3:yy*3+3, i] = 0

        return prob_field




# for x, row in enumerate(puzzle):
#     for y, col in enumerate(puzzle.T):
        
#         if puzzle[x][y] != 0:
#             prob_field[x][y][puzzle[x][y]-1] = 1
#             continue
        
#         cell = get_cell(puzzle, x, y)
#         opts = get_options(row, col, cell)
#         for i in opts:
#             prob_field[x][y][i-1] = 1 # / len(opts)

# return prob_field


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







def prob_field_to_puzzle(prob_field: np.ndarray):
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
def ripple_solve(prob_field: np.ndarray, resolved=None, verbose=False):
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
    rules_grid = np.array([[1, 1, 1],
                           [3, 2, 1],
                           [1, 1, 1]])
    r_wfc = RecursiveWFC(4, 4, rules_grid=rules_grid, verbose=True)

    
    r_wfc.generate_rules(rules_grid=rules_grid)
    
    
    
# prob_field = generate_probability_field(puzzles['evil2'])

# print(prob_field_to_puzzle(ripple_solve_blank(prob_field, verbose=True)))