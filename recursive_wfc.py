import numpy as np
from copy import deepcopy
from numba import njit
import time
import timeit

#  NOTE: Taken from my sudoku_solver repository.
#
# Fully Recursive Solver
#  Essentially masked solve but without any collapse propagation.
#  Each cell collapse triggers a recursion, this is excessive but robust.
#  Goal is to minimize the work performed during each recursion and focus on quickly exploring the solution space.
@njit
def recursive_solve(prob_field: np.ndarray, collapsed_cells: np.ndarray = None):
    # During the initial call, create a blank collapsed_cells array.
    if collapsed_cells is None:
        collapsed_cells = np.zeros((9, 9), dtype=np.bool_)
        
    # NOTE: These are purely performance metrics.
    recursions = 1          # Total recursions, including initial solver call. (min 1)
    failed_recursions = 0   # Total recursions that failed to find a solution. (min 0)
    collapse_count = 0      # Total number of cells collapsed.
    
    # Sum the probability field along the value axis (into a 9x9 array).
    #   The value of each cell is equal to the number of remaining options for that cell.
    resolution_map = prob_field.sum(axis=2)

    # Abort if any cells are unsolvable.
    if not resolution_map.all():
        return None, recursions, failed_recursions, collapse_count
    
    # Return solution if all cells are solved.
    if resolution_map.sum() == 81:
        return prob_field, recursions, failed_recursions, collapse_count
    
    # Overwrite any previously collapsed cells with a high value (10).
    mask_2darray_inplace(resolution_map, collapsed_cells)

    # Find the cell with the lowest number of options (that has not been previously collapsed).
    c = np.argmin(resolution_map)
    x, y = c // 9, c % 9
    collapsed_cells[x, y] = 1
    
    # Determine the remaining options for that cell.
    indexes = np.where(prob_field[x][y])[0]
    
    # If that cell has multiple options, sort them by collapse value.
    if len(indexes) < 1:
        indexes[:] = [x for _, x in sorted(zip([collapse_value(prob_field, x, y, i) for i in indexes], indexes), reverse=False)]
    
    # Recurse over each option.
    for i in indexes:
        
        # Pass copies of the probability field and collapsed cells when recursing.
        pf = prob_field.copy()
        collapse_probability_field(pf, x, y, i)
        
        # Result, recursion_count, failed_recursions, collapse_count
        r, _rs, _frs, _c = recursive_solve(pf, collapsed_cells.copy())
        
        recursions += _rs           # Update metrics.
        failed_recursions += _frs   #
        collapse_count += _c + 1    #
        
        # If a solution was found, return it.
        if r is not None:
            return r,  recursions, failed_recursions, collapse_count

        failed_recursions += 1
        
    # If any cell was fully explored without success, the puzzle is unsolvable.
    return None, recursions, failed_recursions, collapse_count

# Overwrites values in arr based whose indexes in mask_arr are True, with a given maskval
@njit
def mask_2darray_inplace(arr: np.ndarray, mask_arr: np.ndarray, maskval=10):
    w, h = arr.shape[:2]
    for x in range(w):
        for y in range(h):
            if mask_arr[x][y]:
                arr[x][y] = maskval

# NOTE: This needs to be modified to take an adjustable rules array parameter, and collapse cell indexes based on those rules.
@njit
def collapse_probability_field(prob_field: np.ndarray, x: int, y: int, i: int):
    """ Collapses an x, y cell to a single given value-index (i).
        > This change would indicate that the cell is known to contain the value (i+1).
        
        Perpetuates that change by setting the value-index (i) to 0 for all other cells in the row, column, and region.
        Modifies prob_field in place.
        
    Args:
        prob_field (np.ndarray): 9x9x9 grid tracking the possibility of each cell containing each value.
        x (int): X coordinate of the cell to collapse.
        y (int): Y coordinate of the cell to collapse.
        i (int): Index of the value to collapse to. (0-8)
    """
    prob_field[x, :, i] = 0         # Set option i to 0 for all cells in the x column.
    prob_field[:, y, i] = 0         # Set option i to 0 for all cells in the y row.
    
    xx = x // 3                     # Set option i to 0 for all cells in the region.
    yy = y // 3                     # (xx, yy) is the top-left corner of the 3x3 region containing x, y.
    prob_field[xx*3:xx*3+3, yy*3:yy*3+3, i] = 0
    
    prob_field[x, y, :] = 0         # Set all options for the x, y cell to 0.
    prob_field[x, y, i] = 1         # Overwrite option i for the x, y cell to 1.
    
    
# NOTE: This heuristic also needs to take an adjustable rules array parameter.
# Low collapse value means the choice is less likely to lead to a broken board state.
#  NOTE:
@njit
def collapse_value(prob_field: np.ndarray, x: int, y: int, i: int):
    xx = x // 3
    yy = y // 3
    return min(prob_field[xx*3:xx*3+3, yy*3:yy*3+3, i].sum(), prob_field[x, :, i].sum(), prob_field[:, y, i].sum())