# Nathaniel Alden Homans Youngren
# October 2, 2023

import numpy as np
import bisect

# Alias for the type of the probability field.
# TODO: May want a types files.
PF_TYPE = np.ndarray[np.bool_]


# Randomly selects an index from a list of weights.
def weighted_choice(weights):
    cs = np.cumsum(weights)
    return bisect.bisect(cs, np.random.random() * cs[-1])


# If any cell has no remaining options, the probability field is broken.
def default_is_broken(prob_field: np.ndarray) -> bool:
    return not np.count_nonzero(prob_field, axis=2).all()


# If all cells have only one remaining option, the probability field is solved.
# NOTE: This is contingent upon the probability field not being broken.
#       In the future I should remove this reliance despite it typically being a redundant check.
def default_is_solved(prob_field: np.ndarray) -> bool:
    return prob_field.sum() == prob_field.shape[0] * prob_field.shape[1]
