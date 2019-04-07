'''Habitat creatin using cython. (incomplete)'''
import numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

# use below code to import this from other modules:
# compile cython in-place at import time
#import pyximport; pyximport.install()
#from create_habitat import create_habitat

DTYPE = np.int
ctypedef np.int_t DTYPE_t

cdef create_habitat(double cover, double frag, int size):
    cdef int i, j, n
    cdef int M[size][size]

    while n < cover:
        i = np.random.choose(range(size))
        j = np.random.choose(range(size))
        if (i, j) in points:
            continue
        if (i+1, j) in points or (i-1, j) in points or (i, j+1) in points or \
                (i, j-1) in points or np.random.random() < frag:
            points.append((i, j))
            n += 1

    return points
