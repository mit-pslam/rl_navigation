"""Module containg random math functions."""
import numpy as np


def unit_vector(vector):
    """
    Return the unit vector of the vector.

    source: https://stackoverflow.com/a/13849249
    """
    return vector / np.linalg.norm(vector)


def angle_between_vectors(v1, v2):
    """
    Return the angle in radians between vectors 'v1' and 'v2'.

    Examples::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
        source: https://stackoverflow.com/a/13849249
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
