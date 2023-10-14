# Nathaniel Alden Homans Youngren
# October 11, 2023

import numpy as np
import imageio

from wfc_legislator import Legislator
from wfc_solver import Solver

        
if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    
    img_name =          'images/samples/Flowers.png'

    rotate_patterns =   False
    flip_patterns =     False
    wrap_edges =        False
    oob_values =        False
    pattern_size =      (3, 3)
    
    # Read image
    input_img = imageio.v2.imread(img_name)[:, :, :3]
    
    # Initialize legislator
    legislator = Legislator(rotate_patterns=rotate_patterns,
                            flip_patterns=flip_patterns,
                            wrap_edges=wrap_edges,
                            oob_values=oob_values,
                            pattern_size=pattern_size)
    
    # Convert image to tiles
    tile_grid, tile_dict, tile_counts = legislator.convert_img_to_tiles(input_img)
    
    # Generate patterns and adjacency rules
    patterns, pattern_counts, adjacencies = legislator.get_rules(tile_grid)

    solver = Solver(patterns,
                    adjacencies,
                    pattern_weights=None,#pattern_counts,
                    wrap_edges=wrap_edges,
                    default_shape=(24, 24))
    
    
    out_grid = solver.solve(verbose=True)
    viable_grid = legislator.validate_grid_adjacencies(out_grid, adjacencies)
    viable_neighbors_grid = viable_grid.sum(axis=2)
    
    tile_grid = legislator.convert_patterns_to_tiles(out_grid, patterns)
    output_img = legislator.convert_grid_to_img(tile_grid, tile_dict)

    # # Print stats
    # print(out_grid)
    # print(viable_neighbors_grid)

    # for i, pattern in enumerate(patterns):
    #     print(pattern, pattern_counts[i], np.count_nonzero(out_grid.flatten()==i))
        
    # print('Viable Neighbors:',np.count_nonzero(viable_grid), np.count_nonzero(viable_grid==False))
    # print('Viable Tiles',np.count_nonzero(viable_neighbors_grid==4), np.count_nonzero(viable_neighbors_grid!=4))


    # # Check image with CV2
    # import cv2
    # cv2.imshow("out_img", cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() # destroy all windows
    
    # Check image with matplotlib
    import matplotlib.pyplot as plt
    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(output_img)
    axarr[0,0].set_title('Output Image')

    axarr[0,1].imshow(tile_grid)
    axarr[0,1].set_title('Tile IDs')
    
    axarr[1,0].imshow(out_grid)
    axarr[1,0].set_title('Pattern IDs')
    
    axarr[1,1].imshow(viable_neighbors_grid)
    axarr[1,1].set_title('Viable Neighbors')
    plt.show()
    