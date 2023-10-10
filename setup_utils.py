# Nathaniel Alden Homans Youngren
# October 2, 2023

import numpy as np

# Could also use hashing to store colors as integers
def rgb_to_hex(r, g, b): # https://www.educative.io/answers/how-to-convert-hex-to-rgb-and-rgb-to-hex-in-python
    return '{:02x}{:02x}{:02x}'.format(r, g, b)

def hex_to_rgb(hex):
    return tuple([int(hex[i:i+2], 16) for i in (0, 2, 4)])

def view_prob_field(prob_field: np.ndarray[np.bool_], default_val: int = -1) -> np.ndarray[int]:#, patterns: np.ndarray) -> np.ndarray[int]:
    out_arr = np.full(prob_field.shape[:2], fill_value=default_val, dtype=int)
    for x in range(prob_field.shape[0]):
        for y in range(prob_field.shape[1]):
            if prob_field[x, y].sum() == 1:
                out_arr[x, y] = np.argmax(prob_field[x, y]) # patterns[np.where(prob_field[x, y])[0][0]]
    return out_arr


# TODO: Represent unresolved/unresolvable cells with distinct values?
def prob_field_to_array(prob_field: np.ndarray[np.bool_], input_types: np.ndarray) -> np.ndarray[int]:
    out_arr = np.zeros(prob_field.shape[:2], dtype=int)
    for x in range(prob_field.shape[0]):
        for y in range(prob_field.shape[1]):
            if prob_field[x, y].sum() == 1:
                out_arr[x, y] = input_types[np.where(prob_field[x, y])[0][0]]
    return out_arr