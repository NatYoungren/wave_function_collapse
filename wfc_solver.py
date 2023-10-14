# Nathaniel Alden Homans Youngren
# October 2, 2023

import numpy as np
from typing import Optional, Callable, Tuple#, Union, Any, Iterable, Sequence, List, Dict, Set, FrozenSet, Deque, Iterator, TypeVar, Generic, NamedTuple, cast

from heuristic_utils import make_entropy_cell_heuristic, make_pattern_heuristic, make_random_weight_map
from setup_utils import view_prob_field
from solver_utils import PF_TYPE, weighted_choice, default_is_broken, default_is_solved


#   FUTURE WORK:
# Adapt prob_field to having one layer for each rule-pattern.
# Identify rule patterns in the input array.
# Collapse cells based on possible pattern adjacency.

# Each x/y prob_field index still maps directly to a pixel/cell, but collapses to a rule pattern rather than a color.
# Rule patterns are by passing a rules shape lens over the entire rules array, and storing each possible pattern.
# Rules can be rotated/flipped to generate more patterns.
# Call a location heuristic to identify the next wave (cell) to collapse.
# Call a pattern heuristic to identify the next pattern to collapse to.



# Solver loop:
#  Check if unsolvable.
#  Check if solved.
#  (Optional) Propagate until a choice must be made.
#  Identify next cell to collapse.
#  Identify next pattern to collapse cell to.
#  Store copy of the probability field.
#  Collapse cell to pattern and recurse.  (TODO: Replace recursion with flat iteration)
#  If recursion fails, restore probability field and try next pattern.
#  If all patterns fail, return None and revert to previous recursion level.


class Solver:
    
    def __init__(self,
                 patterns:          np.ndarray, # Type?
                 rules:             np.ndarray, # Type?
                 
                 weight_map:        Optional[np.ndarray[np.float_]] = None,
                 
                 cell_heuristic:    Optional[Callable[[PF_TYPE]]] = None,
                 pattern_heuristic: Optional[Callable[[PF_TYPE]]] = None,
                 
                 is_broken:         Optional[Callable[[PF_TYPE]]] = None,
                 is_solved:         Optional[Callable[[PF_TYPE]]] = None,
                 
                 prob_field:        Optional[PF_TYPE] = None,
                 default_shape:     Tuple[int, int] = (24, 24),
                 
                #  oob_value:         int = -1,
                #  neighbor_offsets:  Tuple[Tuple[int, int], ...] = ((-1, 0), (0, -1), (1, 0), (0, 1)),
                #  pattern_size:      Tuple[int, int] = (2, 2), # (3, 3)? # Derivative of patterns?
                 ) -> None:
        
        # Patterns is the set of all possible patterns.
        #  This determines the size of the 3rd dimension of the probability field.
        self.patterns = patterns
        
        # Rules dictates which patterns are allowed to be adjacent to each other.
        self.rules = rules
        
        # Prob_field is the probability field being solved.
        #  It is a 3D boolean array of shape (w, h, t), (t being the number of patterns).
        # Shape is the dimensions of the 2d grid being solved (w, h).
        if prob_field is None:
            self.shape = default_shape
            prob_field = np.ones((default_shape[0], default_shape[1], patterns.shape[0],), dtype=np.bool_)
        else:
            assert prob_field.shape[2] == patterns.shape[0], "Error: 3rd dimension of probability field must match number of patterns."
            self.shape = prob_field.shape[:2]
        self.prob_field = prob_field
        # TODO: Apply rules to initialized probability field. (Remove invalid patterns from each cell.)
        
        # Cell heuristic is used to select the next cell to collapse.
        if cell_heuristic is None:
            
            # Weights are applied to cells during cell selection.
            if weight_map is None:
                weight_map = make_random_weight_map(self.shape)
                
            cell_heuristic = make_entropy_cell_heuristic(weight_map)
        self.cell_heuristic = cell_heuristic
        
        # Pattern heuristic is used to select the next pattern to collapse a cell into.
        if pattern_heuristic is None:
            pattern_heuristic = lambda pf: None
        self.pattern_heuristic = pattern_heuristic
        
        # is_broken is used to determine if the probability field is unsolvable.
        if is_broken is None:
            is_broken = default_is_broken
        self.is_broken = is_broken
        
        # is_solved is used to determine if the probability field is solved.
        if is_solved is None:
            is_solved = default_is_solved
        self.is_solved = is_solved
        
        # Stores past states of the probability field.
        #  Used to revert to previous states without recursion.
        self.history = []
        
        # Metrics
        self.solve_count = 0
        self.propagation_count = 0 # Optional
        self.collapse_count = 0
        
    # TODO: Alternative constructor that takes a starting image and generates the rules and patterns?
    
    
    def solve(self, verbose: bool = False):
        while not self.solve_next_cell():
            if verbose:
                print(self.summarized_prob_field)
                
        return self.prob_field
    
    def solve_next_cell(self) -> bool:
        if self.is_broken is not None and self.is_broken(self.prob_field):
            # TODO: Use history to revert to previous state.
            return False
        
        if self.is_solved is not None and self.is_solved(self.prob_field):
            return True

        # Propagate until a choice must be made.
        
        #  Identify next cell to collapse.
        x, y = self.cell_heuristic(self.prob_field)
        
        #  Identify next pattern to collapse cell to.
        p = self.pattern_heuristic(self.prob_field)
        
        #  Store copy of the probability field.
        self.history.append(self.prob_field.copy())
        # TODO: Remove selected cell/pattern from stored probability field to prevent repetition.
        
        
        
        #  Collapse cell to pattern and recurse.  (TODO: Replace recursion with flat iteration)
        #  If recursion fails, restore probability field and try next pattern.
        #  If all patterns fail, return None and revert to previous recursion level.

        
        
    
    @property
    def summarized_prob_field(self):
        return view_prob_field(self.prob_field)