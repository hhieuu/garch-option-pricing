import numpy as np
from numba import jit


@jit(nopython=True)
def normal_pdf(x, mean=0, sd=1):
    """
    Calculate probability of x given X ~ N(mean, sd ** 2)
    """
    return 1 / (sd * np.sqrt(2 * np.pi)) \
        * np.exp(- 1 / 2 * ((x - mean) / sd) ** 2)