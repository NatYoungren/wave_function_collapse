# Nathaniel Alden Homans Youngren
# October 10, 2023

import numpy as np
from numba import njit
import imageio
from typing import Optional, Callable, Tuple, Dict #, Union, Any, Iterable, Sequence, List,  Set, FrozenSet, Deque, Iterator, TypeVar, Generic, NamedTuple, cast

from setup_utils import rgb_to_hex, hex_to_rgb #, view_prob_field
from solver_utils import PF_TYPE #, weighted_choice, default_is_broken, default_is_solved


# He is a legislator because he makes all the rules!
class Legislator:
    
    def __init__(self, 
                 neighbor_offsets:  Tuple[Tuple[int, int], ...] = ((-1, 0), (0, -1), (1, 0), (0, 1)),
                 pattern_size:      Tuple[int, int] = (2, 2), # (3, 3)? # Derivative of patterns?
                 
                 rotate_patterns:   bool = True, # TODO: Change to a number of 90 degree rotations?
                 flip_patterns:     bool = True, # TODO: Change to list of axes to flip along?
                 
                 wrap_edges:        bool = True,
                 oob_values:        bool = False,
                 oob_id:            int = -1,
                 # overlap amount?
                 # input_grid: np.ndarray, # type? # TODO: Use a method to read grids?
                 ) -> None:
        
        self.neighbor_offsets = neighbor_offsets
        self.pattern_size = pattern_size
        self.rotate_patterns = rotate_patterns
        self.flip_patterns = flip_patterns
        self.wrap_edges = wrap_edges
        self.oob_values = oob_values
        self.oob_id = oob_id

        # self.input_grid = input_grid
    
    
    # TODO: Add a tile size parameter. Allow groups of pixels to be lumped together into a single tile (2x2, 3x3, ...).
    def convert_img_to_tiles(self, input_img: np.ndarray) -> Tuple[np.ndarray, dict[str, int], dict[int, int]]:
        """ Converts a 3-channel image into a grid of tile ids.
        

        Args:
            input_img (np.ndarray): A 3-channel (presumably rgb) image, as a numpy array.

        Returns:
            Tuple[np.ndarray,       : The image with each unique pixel color replaced by a unique tile id.
                  dict[str, int],   : A dictionary mapping each unique hex-color to a unique tile id.
                  dict[int, int]]   : A dictionary mapping each tile id to the number of times it appears in the image.
        """
        assert input_img.ndim == 3
        assert input_img.shape[2] == 3
        
        # NOTE: Tile grid is a 2d array.
        tile_grid = np.empty(input_img.shape[:2], dtype=int)
        # NOTE: All colors and ids are unique, so the dictionary can be inverted to use as a bi-directional map.
        tile_ids = {}
        tile_counts = {}
        
        for x in range(input_img.shape[0]):
            for y in range(input_img.shape[1]):
                px = input_img[x, y]
                hex_val = rgb_to_hex(*px)
                
                # Identify the tile id for the given pixel color, adding one when necessary.
                if hex_val not in tile_ids:
                    tile_ids[hex_val] = len(tile_ids)
                tile_id = tile_ids[hex_val]
                
                # Track the number of times each tile is found.
                tile_counts[tile_id] = tile_counts.get(tile_id, 0) + 1
                
                # Store the tile id in the output grid.
                tile_grid[x, y] = tile_id
                
        return tile_grid, tile_ids, tile_counts


    def convert_patterns_to_tiles(self, pattern_grid: np.ndarray, patterns: np.ndarray) -> np.ndarray:
        tile_grid = np.empty(pattern_grid.shape, dtype=int)
        for x in range(pattern_grid.shape[0]):
            for y in range(pattern_grid.shape[1]):
                pattern = pattern_grid[x, y]
                # print(pattern)
                tile = patterns[pattern][0][0]
                # print(tile)
                tile_grid[x, y] = tile
                
        return tile_grid
    
    
    def convert_grid_to_img(self, input_grid: np.ndarray, tile_dict: dict) -> np.ndarray:
        assert input_grid.ndim == 2
        NUM_COLOR_CHANNELS = 3
        
        img = np.empty((input_grid.shape[0], input_grid.shape[1], NUM_COLOR_CHANNELS), dtype=np.uint8)
        tile_map = {v: k for k, v in tile_dict.items()}
        
        for x in range(input_grid.shape[0]):
            for y in range(input_grid.shape[1]):
                tile_id = input_grid[x, y]
                tile_color = hex_to_rgb(tile_map[tile_id])
                
                img[x, y] = tile_color
                
        return img
    
    
    def get_rules(self, input_grid: np.ndarray) -> Tuple[np.ndarray[int], list]:
        # Extract patterns from the input grid.
        # Transform patterns based on the legislator's settings.
        # Remove duplicate patterns along the way.
        # Identify which patterns are allowed to be adjacent to each other.
        
        patterns = self.extract_patterns(input_grid)
        print(patterns.shape)
        
        
        # unique_patterns = self.remove_pattern_dupes(patterns) # TODO: incorporate into extract/transform functions?
        # print(unique_patterns.shape)
        
        
        patterns = self.transform_patterns(patterns)        
        print(patterns.shape)
        
        
        unique_patterns, pattern_counts = self.remove_pattern_dupes(patterns)
        print(unique_patterns.shape)
        
        #  # Get frequency of patterns in the input grid.
        # pattern_counts = self.get_pattern_counts(patterns, unique_patterns)
        print(pattern_counts)
        
        
        adjacency_rules = self.determine_pattern_adjacency(unique_patterns)
        print(len(adjacency_rules))
        
        return unique_patterns, pattern_counts, adjacency_rules
    
    
    def validate_grid_adjacencies(self, pattern_grid: np.ndarray, adjacencies: list) -> bool:
        
        viable_grid = np.zeros((*pattern_grid.shape, 4), dtype=bool)
        
        for x in range(pattern_grid.shape[0]):
            for y in range(pattern_grid.shape[1]):
                p1 = pattern_grid[x, y]
                for i, (offset, rules) in enumerate(adjacencies):
                    
                    _x, _y = x + offset[0], y + offset[1]
                    if _x < 0 or _x >= pattern_grid.shape[0] or _y < 0 or _y >= pattern_grid.shape[1]:
                        # Considered viable if neighbor is OOB
                        viable_grid[x, y, i] = True
                        continue
                    
                    # # DEBUG: Testing bool mask rules
                    # if rules[p1, pattern_grid[_x, _y]]:
                    #     viable_grid[x, y, i] = True
                    #     continue
                    if pattern_grid[_x, _y] in rules[p1]:
                        # Consider viable if neighbor is in allowed neighbors
                        viable_grid[x, y, i] = True
                        # print('Viable:', x, y, i)
                        continue
                    
                    print('Not viable:', x, y, i, viable_grid[x, y], pattern_grid[_x, _y])#, rules[p1])
                    
                    
        return viable_grid
    
    def extract_patterns(self, input_grid: np.ndarray) -> np.ndarray:
        assert input_grid.ndim == 2
        grid = input_grid.copy()
        
        # Wrapped edges are handled by padding the grid with mirrored values.
        if self.wrap_edges:
            x_range = grid.shape[0]
            y_range = grid.shape[1]
            grid = np.pad(grid,
                          ((0, self.pattern_size[0] - 1),
                           (0, self.pattern_size[1] - 1)),
                          mode='wrap')
        
        # If not wrapping, the grid may be padded with out-of-bounds values.
        elif self.oob_values:
            # TODO: Make sure this concept gets fully implemented down the line.
            x_range = grid.shape[0] + self.pattern_size[0] - 1
            y_range = grid.shape[1] + self.pattern_size[1] - 1
            grid = np.pad(grid,
                          ((self.pattern_size[0] - 1, self.pattern_size[0] - 1),
                           (self.pattern_size[1] - 1, self.pattern_size[1] - 1)), 
                          constant_values=self.oob_id, 
                          mode='constant')
        
        # If no wrap or oob padding is applied, only the existing grid is used.
        else:
            x_range = grid.shape[0] - self.pattern_size[0] + 1
            y_range = grid.shape[1] - self.pattern_size[1] + 1
            
        # Iterate over the grid and store all cutouts of the specified pattern size.
        stored_patterns = np.empty((x_range * y_range, self.pattern_size[0], self.pattern_size[1]), dtype=int)
        i = 0
        for x in range(x_range):
            for y in range(y_range):
                
                pattern = grid[x:x+self.pattern_size[0], y:y+self.pattern_size[1]]
                # patterns_dict[len(patterns_dict)] = pattern
                stored_patterns[i] = pattern
                i+= 1
                # stored_patterns.append(pattern)
        return stored_patterns
    
    
    def transform_patterns(self, patterns: np.ndarray) -> np.ndarray:
        transformed_patterns = patterns.copy()
        
        if self.rotate_patterns:
            transformed_patterns = self.apply_rotation(transformed_patterns)
        if self.flip_patterns:
            transformed_patterns = self.apply_flip(transformed_patterns)
            
        return transformed_patterns
        
    
    def determine_pattern_adjacency(self, patterns: np.ndarray):
        return self._adjacency_helper(patterns, self.neighbor_offsets)
        # return self._bool_adjacency_helper(patterns, self.neighbor_offsets)
    
    
    @staticmethod
    @njit
    def _adjacency_helper(patterns: np.ndarray, offsets: tuple) -> np.ndarray:
        adjacencies = []
        for offset in offsets:
            adj = []
            for i1, pattern1 in enumerate(patterns):
                allowed_neighbors = []
                for i2, pattern2 in enumerate(patterns):
                    # for offset in offsets:
                    a1 = pattern1[max(0, offset[0]):min(pattern1.shape[0], pattern2.shape[0]+offset[0]), max(0, offset[1]):min(pattern1.shape[1], pattern2.shape[1]+offset[1])]
                    a2 = pattern2[max(0, -offset[0]):min(pattern2.shape[0], pattern1.shape[0]-offset[0]), max(0, -offset[1]):min(pattern2.shape[1], pattern1.shape[1]-offset[1])]
                    if np.array_equal(a1, a2):
                        allowed_neighbors.append(i2)
                        # adj.append((i1, i2))
                adj.append(allowed_neighbors)
            adjacencies.append((offset, adj))

        # Format is:
        #   (offset, [[neighbors of pattern0], [neighbors of pattern1], ...])
        #   The index of the list of neighbors indicates whose neighbors are being referenced.
        return adjacencies
    
    
    # @staticmethod
    # @njit
    # def _bool_adjacency_helper(patterns: np.ndarray, offsets: tuple) -> np.ndarray:
    #     adjacencies = np.zeros((len(offsets), len(patterns), len(patterns)), dtype=np.bool_)
    #     for i, offset in enumerate(offsets):
    #         for i1, pattern1 in enumerate(patterns):
    #             for i2, pattern2 in enumerate(patterns):
    #                 # for offset in offsets:
    #                 a1 = pattern1[max(0, offset[0]):min(pattern1.shape[0], pattern2.shape[0]+offset[0]), max(0, offset[1]):min(pattern1.shape[1], pattern2.shape[1]+offset[1])]
    #                 a2 = pattern2[max(0, -offset[0]):min(pattern2.shape[0], pattern1.shape[0]-offset[0]), max(0, -offset[1]):min(pattern2.shape[1], pattern1.shape[1]-offset[1])]
    #                 if np.array_equal(a1, a2):
    #                     adjacencies[i, i1, i2] = True
    #                     # allowed_neighbors.append(i2)
    #                     # adj.append((i1, i2))
    #         # adjacencies.append((offset, adj))

    #     # Format is:
    #     #   (offset, [[neighbors of pattern0], [neighbors of pattern1], ...])
    #     #   The index of the list of neighbors indicates whose neighbors are being referenced.
    #     return adjacencies
    
    @staticmethod
    def apply_rotation(patterns: np.ndarray) -> np.ndarray:
        assert patterns.ndim == 3
        NUM_ROTATIONS = 4
        
        rotated_patterns = np.empty((patterns.shape[0]*NUM_ROTATIONS, patterns.shape[1], patterns.shape[2]), dtype=int)
        
        for i in range(patterns.shape[0]):
            for ii in range(NUM_ROTATIONS):
                rotated_patterns[i*NUM_ROTATIONS+ii] = np.rot90(patterns[i], ii)
            
        return rotated_patterns
    
    
    @staticmethod
    def apply_flip(patterns: np.ndarray) -> np.ndarray:
        # NOTE: Flips along one axis only. Combined with rotation this should cover all possible flips.
        # NOTE: Without rotation you may still want the ability to flip vertically/horizontally, add that later.
        assert patterns.ndim == 3
        
        flipped_patterns = np.empty((patterns.shape[0]*2, patterns.shape[1], patterns.shape[2]), dtype=int)
        
        for i in range(patterns.shape[0]):
            flipped_patterns[i*2] = patterns[i]
            flipped_patterns[i*2+1] = np.flip(patterns[i], axis=1)
        
        return flipped_patterns
    
    
    # TODO: Clean up this pattern counts logic to make more sense?
    #       Should it be here? Should it be scaled between 0 and one?
    @staticmethod
    def remove_pattern_dupes(patterns: np.ndarray) -> np.ndarray:
        unique_patterns = np.unique(patterns, axis=0)
        pattern_counts = np.zeros(len(unique_patterns), dtype=np.float64)
        for i, p in enumerate(unique_patterns):
            match = np.all(patterns == p, axis=(1, 2))
            pattern_counts[i] = np.count_nonzero(match)
            # print(match, pattern_counts)
        
        return unique_patterns, pattern_counts
        
        
if __name__ == '__main__':
    imgname = 'images/samples/Flowers.png'

    rotate_patterns =   False
    flip_patterns =     False
    wrap_edges =        False
    oob_values =        False
    pattern_size =      (3, 3)
    
    input_img = imageio.v2.imread(imgname)[:, :, :3]

    # Initialize legislator
    leg = Legislator(rotate_patterns=rotate_patterns,
                     flip_patterns=flip_patterns,
                     wrap_edges=wrap_edges,
                     oob_values=oob_values,
                     pattern_size=pattern_size)
    
    
    tile_grid, tile_dict, tile_counts = leg.convert_img_to_tiles(input_img)

    # patterns = leg.extract_patterns(tile_grid)
    # print(patterns.shape)

    # patterns = leg.remove_pattern_dupes(patterns)
    # print(patterns.shape)

    # t_patterns = leg.transform_patterns(patterns)
    # print(t_patterns.shape)
        
    # t_patterns = leg.remove_pattern_dupes(t_patterns)
    # print(t_patterns.shape)

    # adjacencies = leg.determine_pattern_adjacency(t_patterns)
    # print(len(adjacencies))
    
    patterns, pattern_counts, adjacencies = leg.get_rules(tile_grid)
    print(adjacencies)
    print(patterns)
    print(pattern_counts)
    print(len(patterns))

    # # Check image
    # import cv2
    # output_img = leg.convert_grid_to_img(tile_grid, tile_dict)
    # cv2.imshow("out_img", cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() # destroy all windows
    
    
