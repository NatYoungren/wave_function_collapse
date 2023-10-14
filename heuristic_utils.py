# Nathaniel Alden Homans Youngren
# October 2, 2023

import numpy as np
from typing import Optional, Tuple
from solver_utils import PF_TYPE
# TODO: Consider optimizing methods with numba.

# Reusing baked weights idea (and a few others) from: https://github.com/ikarth/wfc_2019f
def make_random_weight_map(shape: Tuple[int, int]):
    return np.random.random_sample(shape) * 0.1


# Sums are over the weights of each remaining
# allowed tile type for the square whose
# entropy we are calculating.
# Shannon entropy is calculated as:
#        shannon_entropy_for_square =
#        log(sum(weight)) -
#        (sum(weight * log(weight)) / sum(weight))

# def make_shannon_entropy_heuristic(weight_map: np.ndarray):

def make_entropy_cell_heuristic(cell_weights: np.ndarray[np.float_]):
    # Define a method to select the cell with the lowest entropy, factoring in a map of pregenerated weights.
    
    def get_next_cell(prob_field: PF_TYPE) -> Tuple[int, int]:
        # NOTE: Axis == 2 because we put the tile types in the 3rd dimension.
        
        unresolved_cell_mask = np.count_nonzero(prob_field, axis=2) > 1
        weights = np.where(
            unresolved_cell_mask,
            cell_weights + np.count_nonzero(prob_field, axis=2),
            np.inf,
        )
        row, col = np.unravel_index(np.argmin(weights), weights.shape)
        return row.item(), col.item()
    
    return get_next_cell


def make_pattern_heuristic(pattern_weights: np.ndarray[np.float_]):
    # Define a method to select the pattern with the lowest entropy, factoring in a map of pregenerated weights.
    num_of_patterns = len(pattern_weights)

    def get_next_pattern(wave: PF_TYPE) -> int:
        weighted_wave: np.ndarray[np.float_] = pattern_weights * wave
        weighted_wave /= weighted_wave.sum() # TODO: Is this necessary
        result = np.random.choice(num_of_patterns, p=weighted_wave)
        return result
    
    return get_next_pattern


# def makeWeightedPatternHeuristic(weights: NDArray[np.floating[Any]]):
#     num_of_patterns = len(weights)

#     def weightedPatternHeuristic(wave: NDArray[np.bool_], _: NDArray[np.bool_]) -> int:
#         # TODO: there's maybe a faster, more controlled way to do this sampling...
#         weighted_wave: NDArray[np.floating[Any]] = weights * wave
#         weighted_wave /= weighted_wave.sum()
#         result = numpy.random.choice(num_of_patterns, p=weighted_wave)
#         return result

#     return weightedPatternHeuristic

# From: https://github.com/ikarth/wfc_2019f
# def makeEntropyLocationHeuristic(preferences: NDArray[np.floating[Any]]) -> Callable[[NDArray[np.bool_]], Tuple[int, int]]:
#     def entropyLocationHeuristic(wave: NDArray[np.bool_]) -> Tuple[int, int]:
#         unresolved_cell_mask = numpy.count_nonzero(wave, axis=0) > 1
#         cell_weights = numpy.where(
#             unresolved_cell_mask,
#             preferences + numpy.count_nonzero(wave, axis=0),
#             numpy.inf,
#         )
#         row, col = numpy.unravel_index(numpy.argmin(cell_weights), cell_weights.shape)
#         return row.item(), col.item()

#     return entropyLocationHeuristic
