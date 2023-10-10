# Nathaniel Alden Homans Youngren
# October 2, 2023

import numpy as np
from copy import deepcopy
from numba import njit
from typing import Optional
import bisect
import random
import time
import timeit

from sudoku_functions import mask_2darray_inplace
from rule_utils import generate_simple_rules
from heuristic_utils import make_entropy_cell_heuristic, make_random_weight_map
from solver_utils import weighted_choice
from setup_utils import prob_field_to_array


# Fully Recursive Solver (Updated for non-sudoku)
# @njit
def wfc_recursive_solve(prob_field: np.ndarray, rules_arr: np.ndarray, rules_origin: tuple, wrap_edges: bool = False, collapsed_cells: Optional[np.ndarray] = None):
    # During the initial call, create a blank collapsed_cells array.
    if collapsed_cells is None:
        collapsed_cells = np.zeros(prob_field.shape[:2], dtype=np.bool_)
    
    print(prob_field_to_array(prob_field, [i+1 for i in range(prob_field.shape[2])]))
    
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
    if resolution_map.sum() == resolution_map.size:
        return prob_field, recursions, failed_recursions, collapse_count
    
    # Overwrite any previously collapsed cells with a high value (255).
    mask_2darray_inplace(resolution_map, collapsed_cells)
    
    # Find the cell with the lowest number of options (that has not been previously collapsed).
    c = np.argmin(resolution_map)
    x, y = c // collapsed_cells.shape[0], c % collapsed_cells.shape[0]
    collapsed_cells[x, y] = 1
    
    # Determine the remaining options for that cell.
    indexes = list(np.where(prob_field[x][y])[0])

    # Generate weights for each option. # TODO: Maybe select cells based on weights from the whole grid rather than just the current cell?
    if len(indexes) > 1:
        weights = [#rules_collapse_value(prob_field, x, y, i, rules_arr, rules_origin, wrap_edges) * 
                   ratio_collapse_value(prob_field, x, y, i, rules_arr, rules_origin, wrap_edges)
                   for i in indexes]
    else:
        weights = [1]

    
    # Recurse over each option.
    while len(indexes) > 0:
        print('indexes',indexes)
        print('weights',weights)
        _i = weighted_choice(weights)
        i = indexes.pop(_i)
        weights.pop(_i)
        print('Selected:', x, y, 'with', i)
        
        # Pass copies of the probability field and collapsed cells when recursing.
        pf = prob_field.copy()
        apply_collapse_rules(pf, x, y, i, rules_arr, rules_origin, wrap_edges)
        
        # Result, recursion_count, failed_recursions, collapse_count
        r, _rs, _frs, _c = wfc_recursive_solve(pf, rules_arr, rules_origin, wrap_edges, collapsed_cells.copy())
        
        recursions += _rs           # Update metrics.
        failed_recursions += _frs   #
        collapse_count += _c + 1    #
        
        # If a solution was found, return it.
        if r is not None:
            return r, recursions, failed_recursions, collapse_count

        failed_recursions += 1
        print('Failed:', x, y, 'with', i)
        
    # If any cell was fully explored without success, the puzzle is unsolvable.
    return None, recursions, failed_recursions, collapse_count


# NOTE: This needs to be modified to take an adjustable rules array parameter, and collapse cell indexes based on those rules.
@njit
def apply_collapse_rules(prob_field: np.ndarray, x: int, y: int, i: int, rules_arr: np.ndarray, rules_origin: tuple, wrap_edges: bool = False):
    """ Collapses an x, y cell to a single given value-index (i).
        > This change would indicate that the cell is known to contain the value (i+1).
        
        Perpetuates that change by removing nonviable options from the probability field.
        Modifies prob_field in place.
        
    Args:
        prob_field (np.ndarray): 3D grid tracking the possibility of each cell containing each value.
        x (int): X coordinate of the cell to collapse.
        y (int): Y coordinate of the cell to collapse.
        i (int): Index of the value to collapse to.
        
        rules_arr (np.ndarray): 4D grid tracking all possible neighbors for each celltype.
        rules_origin (tuple): (x, y) coordinates of the origin of the rules array (i.e. index to which the current cell is aligned).
        wrap_edges (bool): Whether to wrap at the edges of the input array.
    """
    for _x in range(rules_arr.shape[1]):
        for _y in range(rules_arr.shape[2]):
            for _i in range(rules_arr.shape[3]):
                __x = x - rules_origin[0] + _x
                __y = y - rules_origin[1] + _y
                
                # Can use subfuncs to avoid repeated checks.
                if wrap_edges:
                    __x %= prob_field.shape[0]
                    __y %= prob_field.shape[1]
                else:
                    if __x < 0 or __x >= prob_field.shape[0] or __y < 0 or __y >= prob_field.shape[1]:
                        continue
                
                # Simplify this.
                if rules_arr[i, _x, _y, _i] == 0:
                    prob_field[__x, __y, _i] = 0


def ratio_collapse_value(prob_field: np.ndarray, x: int, y: int, i: int, rules_arr: np.ndarray, rules_origin: tuple, wrap_edges: bool = False):
    # NOTE: Wrap not needed here.
    rules_count = rules_arr[i, rules_origin[0], rules_origin[1], i].sum()
    rules_ratio = rules_count / rules_arr[:, rules_origin[0], rules_origin[1], :].sum()
    print('rules_ratio:', i, rules_count, rules_arr[:, rules_origin[0], rules_origin[1], :].sum(), rules_ratio)
    
    if rules_ratio == 0:
        print(i, 'not found in rules')
        return 255
    
    resolution_map = prob_field.sum(axis=2)
    known_cells = np.where(resolution_map == 1)
    
    # TODO: Should we factor in possible cells as well/alternatively?
    # print('known_cells', known_cells)
    if len(known_cells[0]) == 0:
        print('no known cells')
        return rules_ratio
    
    type_count = 1
    for _x, _y in zip(*known_cells):
        if _x == x and _y == y:
            continue
        # _x, _y = known_cell // prob_field.shape[0], known_cell % prob_field.shape[0]
        if prob_field[_x, _y, i]:
            type_count += 1
    
    type_ratio = type_count / len(known_cells[0])
    
    print('type_ratio:', i, type_count, len(known_cells[0]), type_ratio)
    print('ratio:', i, type_ratio / rules_ratio)
    return type_ratio / rules_ratio


# @njit
def rules_collapse_value(prob_field: np.ndarray, x: int, y: int, i: int, rules_arr: np.ndarray, rules_origin: tuple, wrap_edges: bool = False):
    t, w, h, _ = rules_arr.shape
    _x, _y, _ = prob_field.shape
    
    # These values represent how far the rules array extends beyond the edges of the probability field.
    x_offset = max(0, rules_origin[0] - x)
    y_offset = max(0, rules_origin[1] - y)
    x_offset_end = max(0, x + (w - rules_origin[0]) - _x) # x position + width of rules - width of prob_field
    y_offset_end = max(0, y + (h - rules_origin[1]) - _y) # x: 0 + (3 - 1) - 3 = -1 (0)
                                                          # j: 2 + (3 - 1) - 3 = 1
    if wrap_edges: # TODO: Test wrapping.
        # These values are used to shift the prob_field such that the rules array does not extend beyond the edges of the prob_field.
        # x . . (x - rules_origin[0])
        # . . . (0 - 1)
        # . . j
        if x_offset and x_offset_end or y_offset and y_offset_end:
            raise ValueError("Error: rules array cannot wrap at both edges of the probability field.")
        
        roll_indexes = (x_offset - x_offset_end, y_offset - y_offset_end)
        
        print('preroll', prob_field[:, :, i])
        print('rolling', roll_indexes)
        print('postroll', np.roll(prob_field[:, :, i], roll_indexes, axis=(0, 1)))
        
        sample = np.roll(prob_field[:, :, i], roll_indexes, axis=(0, 1))
        sample = sample * rules_arr[i, :, :, i]
    else:
        # We use our offsets to crop the rules array to the size of the compared area in the probability field.
        sample = prob_field[max(0, x-rules_origin[0]):min(_x, x-rules_origin[0]+w), max(0, y-rules_origin[1]):min(_y, y-rules_origin[1]+h), i] * rules_arr[i, x_offset:w-x_offset_end, y_offset:h-y_offset_end, i]
    
    return np.count_nonzero(sample)# sample.sum()




if __name__ == '__main__':
    wrap_edges = True
    input_arr = np.array([[1, 2, 1],
                          [3, 4, 3],
                          [1, 2, 1]])
    
    input_types = np.unique(input_arr)
    num_types = len(input_types)
    print(input_types)
    
    rules_arr = np.zeros((num_types, 3, 3, num_types))
    
    generate_simple_rules(input_arr, input_types, rules_arr, wrap_edges=wrap_edges)
    
    # All cell-type relationships with 1
    print(rules_arr[:, :, :, 0])
    solution, r, fr, cc = wfc_recursive_solve(np.ones((7, 7, num_types), dtype=np.bool_), rules_arr, (1, 1), wrap_edges=wrap_edges)
    print('finished:')
    print(solution, r, fr, cc)
    print(prob_field_to_array(solution, input_types))