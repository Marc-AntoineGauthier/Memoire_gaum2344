import numpy as np
from itertools import product as itp
from logging import raiseExceptions

def k_gen(start, end, n):
    if n > 1:
        inter = (end - start)/(n-1)
        for i in range(n):
            yield start + (i*inter)
    elif n==1 and start == end:
        yield start
    elif n==1 and start < end:
        yield start
    else:
        raiseExceptions

def k_gen_cut(start, end, n):
    if n > 1:
        inter = (end - start)/(n+1)
        for i in range(1, n+2):
            yield start + (i*inter)
    else:
        raiseExceptions

def kw_gen_squ(
    start_x, end_x, start_y, end_y, taille_x, taille_y
):
    kx = k_gen(start_x, end_x, taille_x)
    ky = k_gen(start_y, end_y, taille_y)
    kz = 0.
    for x, y in itp(kx, ky):
        yield [x, y, kz]

def kw_gen_tri(start, end, taille_x, taille_y):
    kx = k_gen(start, end, taille_x)
    ky = k_gen(0., 0.5, int(taille_y/2)+1)
    kz = 0.
    for x, y in itp(kx, ky):
        if y < x or np.isclose(y, x):
            if (
                not(np.isclose(x, y)) and not(np.isclose(x, 0.5)) and
                not(np.isclose(y, 0.))
            ):
                yield [x, y, kz, 8]
            elif np.isclose(x, 0.) and np.isclose(y, 0.):
                yield [x, y, kz, 1]
            elif np.isclose(x, 0.5) and np.isclose(y, 0.5):
                yield [x, y, kz, 1]
            elif np.isclose(x, 0.5) and np.isclose(y, 0.):
                yield [x, y, kz, 2]
            elif np.isclose(x, 0.5):
                yield [x, y, kz, 4]
            elif np.isclose(y, 0.):
                yield [x, y, kz, 4]
            elif np.isclose(x, y):
                yield [x, y, kz, 4]
            else:
                print("PANIC! >:O")