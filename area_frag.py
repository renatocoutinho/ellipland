'''Analyze fragmentation by solving random landscapes.'''
from numpy import *
from multiprocessing import Pool
from pickle import load, dump
from itertools import product as iproduct
#from functools import partial
from landscape import *

def analyze_random_landscape(args):
    par, dx, cover, frag, size, radius, norm = args
    landscape = random_landscape(cover, frag, size, radius, norm)
    sol = solve_landscape(landscape, par, dx)
    count = popcount_patches(landscape, sol)
    return [landscape, count]

def run_landscapes(par, dx, cover, frag, size, runs=1, radius=1,
        norm='taxicab', processes=7, filename=None):
    if processes is None or processes > 0:
        pool = Pool(processes=processes)
        mymap = pool.map
    else:
        mymap = map

    newpars = []
    for obj in (par, dx, cover, frag, size):
        if not type(obj) is list:
            newpars.append([obj])
        else:
            newpars.append(obj)

    newpars += [[radius], [norm]]

    works = list(iproduct(*newpars))
    # print(works)
    solutions = mymap(analyze_random_landscape, works)

    if filename:
        with open(filename, 'w') as f:
            dump([ parameters, solutions ], f)

    return solutions

