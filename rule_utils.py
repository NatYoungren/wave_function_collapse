# Nathaniel Alden Homans Youngren
# October 2, 2023

import numpy as np
from typing import Optional, Tuple

def generate_simple_rules(input_arr: np.ndarray, input_types: np.ndarray, rules_arr: np.ndarray, rules_origin: Optional[Tuple] = None, wrap_edges: bool = False):
    """ Generates a rules array from an input array.
        Input array can be any 2d shape, but must contain values from the input_types array.
        
        The rules array is a 3D array of shape (w, h, t).
        Each index in the rules array tracks the possibility of a neighboring cell containing a corresponding value.
        The rules array is initialized with 1s in all cells.
        The rules array is then collapsed to 0s for all cells that are known to contain a value.
        
        The rules array is used to track the possibility of each cell containing each value.
        The rules array is modified in place.
        
        NOTE: These rules are a simplified version of the rules used in the Wave Function Collapse algorithm.
              These are essentially masks which reduce the rules array to single set of allowed neighbors for each cell.
              No information is recorded about the various neighbor configurations.
        
    Args:
        input_arr (np.ndarray): (n, m) array of values.
        input_types (np.ndarray): (t) length array of possible cell values.
        rules_arr (np.ndarray): (t, w, h, t) array of rules. (0-1)
        rules_origin (tuple): (x, y) coordinates of the origin of the rules array (i.e. index to which the current cell is aligned).
        wrap_edges (bool): Whether to wrap at the edges of the input array.
    """
    n, m = input_arr.shape
    t = len(input_types)
    _t, w, h, __t = rules_arr.shape
    
    if not (t == _t == __t):
        raise ValueError("Error: input_types length does not match rules_arr depth.")
    
    if rules_origin is None:
        if h % 2 == 1 and w % 2 == 1:
            rules_origin = (h // 2, w // 2)
        else:
            raise ValueError("Error: rules_origin must be provided for even-sized rules_arr.")
        
    for _n in range(n):
        for _m in range(m):
            # Determine type-index of current cell.
            v = np.where(input_types == input_arr[_n, _m])[0]

            # Iterate over rules area and record which cell-types are neighbors of the current cell.
            for _w in range(w):
                for _h in range(h):
                    x, y = _n - rules_origin[0] + _w, _m - rules_origin[1] + _h
                    
                    if wrap_edges:
                        x %= n
                        y %= m
                    else:
                        if x < 0 or x >= n or y < 0 or y >= m:
                            continue
                    # Add edge type to rules array?
                    
                    _v = np.where(input_types == input_arr[x, y])[0]
                    
                    rules_arr[v, _w, _h, _v] += 1
