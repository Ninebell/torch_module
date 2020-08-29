import numpy as np
import  math


def create_2d_heat_map(point, size, sigma=3):
    base = np.zeros(size)

    x = math.ceil(point[0])
    y = math.ceil(point[1])

    for r in range(size[0]):
        for c in range(size[1]):
            base[r, c] = np.exp(-((r - y) ** 2 + (c - x) ** 2) / sigma)

    return base

