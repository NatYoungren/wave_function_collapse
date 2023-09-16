# Nathaniel Alden Homans Youngren
# September 16, 2023

# Display vars
SCREEN_W, SCREEN_H = 800, 600
BORDER_PX = 2 # Width of border in pixels

# Color of bordering/background
BG_COLOR = (0, 0, 0)

# Used if a tile type is not in the color dictionary
DEFAULT_COLOR = (130, 130, 130)

COLOR_DICT = {0: (30,  30,  30),
              1: (255, 255, 255),
              2: (255, 30,  30),
              3: (30,  255, 30),
              4: (30,  30,  255),
              5: (255, 100, 100),
              6: (100, 255, 100),
              7: (100, 100, 255),
              8: (100, 255, 255),
              9: (255, 100, 255),
             10: (255, 255, 100)} # 10 is unused 

# Text vars
TEXT_FONT = 'freesansbold.ttf'
TEXT_COLOR = (100, 100, 200) # Blue
TEXT_ALPHA = 230
TEXT_OFFSET = (40, -20)
TEXT_SIZE = 24