# Nathaniel Alden Homans Youngren
# September 16, 2023

import numpy as np

COLOR_DICT = {0: (0,   0,   0),
              1: (255, 255, 255),
              2: (255, 0,   0),
              3: (0,   255, 0),
              4: (0,   0,   255),
              5: (255, 100, 100),
              6: (100, 255, 100),
              7: (100, 100, 255),
              8: (100, 255, 255),
              9: (255, 100, 255),
             10: (255, 255, 100)}

# Used if a tile type is not in the color dictionary
DEFAULT_COLOR = (130, 130, 130)