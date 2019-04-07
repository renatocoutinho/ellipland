'''Single species, one type of patch and matrix - basic module.'''
from numpy import *
from collections import OrderedDict
from itertools import product as iproduct
from scipy.ndimage import label

try:
    from matplotlib.pyplot import *
except ImportError:
    pass

par = OrderedDict([ 
    ('r', 0.1),
    ('K', 1.),
    ('mu', 0.03),
    ('Dp', 1e-4),
    ('Dm', 1e-3),
    # interface condition (=Dm/Dp)
    ('g', 1.),
    # boundary conditions
    ('left', [1., 0., 0.]),
    ('right', [1., 0., 0.]),
    ('top', [1., 0., 0.]),
    ('bottom', [1., 0., 0.])
    ])

# in km
#dx = 0.0286
# 35 pixels -> 1km = 1000m
# 1 pixel -> 0.0286km
# D: 10^-2 km^2/day
# r: 10-50/year ~ 0.1 day^-1
# mu: 1/30 days ~ 0.03 day^-1

## dados novos

# in km
dx = 0.02
# 1 pixel -> 0.02km
# Dm ~ 900 m^2/day ~ 10^-3 km^2/day
# Dp ~ 5 m^2/day ~ 10^-5 km^2/day
# r: 10-50/year ~ 0.1 day^-1
# mu: 1/30 days ~ 0.03 day^-1


def solve_landscape(landscape, par, dx, f_tol=None, verbose=True):
    '''Find the stationary solution for a given landscape and set of parameters.

    Uses a Newton-Krylov solver with LGMRES sparse inverse method to find a
    stationary solution (or the solution to the elliptical problem) to the
    system of equations in 2 dimensions (x is a 2-d vector):

    .. math::
        u_t(x) &= D_p \\nabla^2 u(x) + ru(1-u(x)/K) = 0 \\text{ in a patch} \\\\
        v_t(x) &= D_m \\nabla^2 v(x) - \mu v(x) = 0 \\text{ in the matrix}

    Parameters
    ----------

    landscape : a 2-d array (of ints) describing the landscape, with 1 on
        patches and 0 on matrix
    par : a ordered dict containing parameters in the following order: 
        r: reproductive rate on patches
        K: carrying capacity on patches
        mu: mortality rate in the matrix
        Dp: diffusivity on patches
        Dm: diffusivity in the matrix
        g: habitat preference parameter \gamma, usually less than one. See
            interface conditions below
        left: (a, b, c): external boundary conditions at left border
        right: (a, b, c): external boundary conditions at right border
        top: (a, b, c): external boundary conditions at top border
        bottom: (a, b, c): external boundary conditions at bottom border
    dx : lenght of each edge
    f_tol : float, tolerance for the residue, passed on to the solver routine.
        Default is 6e-6
    verbose : print residue of the solution and its maximum and minimum values

    Returns
    -------

    solution : 2-d array of the same shape of the landscape input containing the solution

    Boundary and interface conditions
    ---------------------------------

    External boundaries are of the form

    .. math::
        a \\nabla u \cdot \hat{n} + b u + c = 0

    and may be different for left, right, top, bottom.  The derivative of u is
    taken along the normal to the boundary.

    The interfaces between patches and matrix are given by
        
    .. math::
        u(x) &= \gamma v(x) \\\\
        D_p \\nabla u(x) \cdot \hat{n} &= D_m \\nabla v(x) \cdot \hat{n}

    where u is a patch and v is the solution in the matrix. These conditions
    are handled using an assymetric finite difference scheme for the 2nd
    derivative:

    .. math::
        u_xx(x) = \\frac{4}{3h^2} (u(x-h) - 3 u(x) + 2 u(x+h/2))

    with the approximations at the interface:

    .. math::
        u(x+h/2) = \\frac{D_m v(x+h)+D_p u(x)}{(D_p+D_m g}

    if u(x) is in a patch and v(x+h) is in the matrix, or

    .. math::
        v(x+h/2) = \\frac{g(D_m v(x)+D_p u(x+h))}{D_p+D_m g}

    if v(x) is in the matrix and u(x+h) is in a patch.

    '''
    from scipy.optimize import newton_krylov

    (r, K, mu, Dp, Dm, g, (al, bl, cl), (ar, br, cr), (at, bt, ct), (ab, bb,
        cb)) = par.values()

    lin_term = r * landscape - mu * (1-landscape)
    sec_term = - landscape * r / K
    D = landscape * Dp + (1-landscape) * Dm

    Bxpm, Bxmp, Bypm, Bymp = find_interfaces(landscape)
    factor_pp = -1. + 2. * Dp/(Dp+Dm*g)
    factor_pm = -1. + 2. * Dm/(Dp+Dm*g)
    factor_mp = -1. + 2. * g * Dp/(Dp+Dm*g)
    factor_mm = -1. + 2. * g * Dm/(Dp+Dm*g)

    def residual(P):
        d2x = zeros_like(P)
        d2y = zeros_like(P)

        d2x[1:-1,:] = P[2:,:]   - 2*P[1:-1,:] + P[:-2,:]
        # external boundaries
        d2x[0,:] = P[1,:] - 2*P[0,:] + (-cl - al/dx * P[0,:])/(bl - al/dx)
        d2x[-1,:] = P[-2,:] - 2*P[-1,:] + (-cr + ar/dx * P[-1,:])/(br + ar/dx)
        # interface conditions
        d2x[:-1,:] += Bxpm * (P[:-1,:] * factor_pp + P[1:,:] * factor_pm) + \
                Bxmp * (P[:-1,:] * factor_mm + P[1:,:] * factor_mp)
        d2x[1:,:] += Bxpm * (P[:-1,:] * factor_mp + P[1:,:] * factor_mm) + \
                Bxmp * (P[:-1,:] * factor_pm + P[1:,:] * factor_pp)
        d2x[:-1,:] *= (Bxpm+Bxmp)*1./3. + Bxpm*Bxmp/3. + ones(Bxpm.shape)

        d2y[:,1:-1] = P[:,2:] - 2*P[:,1:-1] + P[:,:-2]
        # external boundaries
        d2y[:,0] = P[:,1] - 2*P[:,0] + (-cb - ab/dx * P[:,0])/(bb - ab/dx)
        d2y[:,-1] = P[:,-2] - 2*P[:,-1] + (-ct + at/dx * P[:,-1])/(bt + at/dx)
        # interface conditions
        d2y[:,:-1] += Bypm * (P[:,:-1] * factor_pp + P[:,1:] * factor_pm) + \
                Bymp * (P[:,:-1] * factor_mm + P[:,1:] * factor_mp)
        d2y[:,1:] += Bypm * (P[:,:-1] * factor_mp + P[:,1:] * factor_mm) + \
                Bymp * (P[:,:-1] * factor_pm + P[:,1:] * factor_pp)
        d2y[:,:-1] *= (Bypm+Bymp)*1./3. + Bypm*Bymp/3. + ones(Bypm.shape)

        return D*(d2x + d2y)/dx/dx + lin_term*P + sec_term*P**2

    # solve
    guess = K * ones_like(landscape)
    sol = newton_krylov(residual, guess, method='lgmres', f_tol=f_tol)
    if verbose:
        print('Residual: %e' % abs(residual(sol)).max())
        print('max. pop.: %f' % sol.max())
        print('min. pop.: %f' % sol.min())

    return sol

def find_interfaces(landscape):
    '''Helper function that marks where are the internal boundaries.'''
    B = 2*landscape - 1
    Bx = B[:-1,:]*(- B[1:,:] * B[:-1,:] + 1)//2
    Bxpm = (Bx + 1)//2
    Bxmp = (-Bx + 1)//2

    By = B[:,:-1]*(- B[:,1:] * B[:,:-1] + 1)//2
    Bypm = (By + 1)//2
    Bymp = (-By + 1)//2
    return Bxpm, Bxmp, Bypm, Bymp

def solve_multiple_parameters(variables, values, landscape, par, dx, f_tol=None, verbose=True,
        multiprocess=True):
    '''Solve given landscape for several combinations of parameters.

    Solves a given landscape with a set of common parameters and for a range of
    values for some variables, optionally using multiprocessing to speed things
    up.

    Parameters
    ----------
    variables : list of strings with the name of the varied parameters
    values : list of lists (or tuples), where each item contains a list of
        values corresponding to the parameters given in the `variables` list
    landscape : 2-d array of zeroes and ones describing the landscape
    par : common values for all the problem parameters. See documentation for
        `solve_landscape()`
    dx : lenght of each edge
    f_tol : float, tolerance for the residue, passed on to the solver routine.
        Default is 6e-6
    verbose : print residue of the solution and its maximum and minimum values.
        Notice that the order of appearance of each output is not the same as
        the input if multiprocess is True.
    multiprocess : determines whether to use multiprocessing to use multiple
        cores. True by default, in which case the total number of CPUs minus
        one are used

    Returns
    -------
    solutions : list containing the solutions to each set of parameters, in the
        same ordering of the input values

    Example
    -------
    >>> from landscape import *
    >>> lA = image_to_landscape('landA.tif')
    >>> D = arange(0.01, 0.05, 0.01)
    >>> r = arange(0.1, 0.5, 0.1)
    >>> values = [ (x, y, y) for x, y in iproduct(r, D) ]
    >>> sols = solve_multiple_parameters(['r', 'Dp', 'Dm'], values, lA, par, dx)

    '''
    from functools import partial
    # this is not compatible with python 2.6 when using multiprocessing, due to
    # bug http://bugs.python.org/issue5228
    solve_landscape_wrapper = partial(solve_landscape, landscape, dx=dx,
            f_tol=f_tol, verbose=verbose)
    works = [ OrderedDict(par, **dict(zip(variables, p))) for p in values ]

    if multiprocess:
        from multiprocessing import Pool, cpu_count
        cpus = cpu_count()
        pool = Pool(cpus if cpus == 1 else cpus - 1)
        solutions = pool.map(solve_landscape_wrapper, works)
        pool.close()
        pool.join()
    else:
        solutions = map(solve_landscape_wrapper, works)

    return solutions

def solve_multiple_landscapes(landscapes, par, dx, f_tol=None, verbose=True,
        multiprocess=True):
    '''Solve several landscapes.

    Solves a set of landscape with a given set of parameters, optionally using
    multiprocessing to speed things up.

    Parameters
    ----------
    landscape : list of 2-d array of zeroes and ones describing the landscape
    par : values for all the problem parameters. See documentation for
        `solve_landscape()`
    dx : lenght of each edge
    f_tol : float, tolerance for the residue, passed on to the solver routine.
        Default is 6e-6
    verbose : print residue of the solution and its maximum and minimum values.
        Notice that the order of appearance of each output is not the same as
        the input if multiprocess is True.
    multiprocess : determines whether to use multiprocessing to use multiple
        cores. True by default, in which case the total number of CPUs minus
        one are used

    Returns
    -------
    solutions : list containing the solutions to each landscape, in the same
        ordering of the input values

    Example
    -------
    >>> from landscape import *
    >>> lA = image_to_landscape('landA.tif')
    >>> lB = image_to_landscape('landB.tif')
    >>> lC = image_to_landscape('landC.tif')
    >>> ll = [lA, B, lC]
    >>> sols = solve_multiple_parameters(ll, par, dx)

    '''
    from functools import partial
    # this is not compatible with python 2.6 when using multiprocessing, due to
    # bug http://bugs.python.org/issue5228
    solve_landscape_wrapper = partial(solve_landscape, par=par, dx=dx,
            f_tol=f_tol, verbose=verbose)

    if multiprocess:
        from multiprocessing import Pool, cpu_count
        cpus = cpu_count()
        pool = Pool(cpus if cpus == 1 else cpus - 1)
        solutions = pool.map(solve_landscape_wrapper, ll)
        pool.close()
        pool.join()
    else:
        solutions = map(solve_landscape_wrapper, works)

    return solutions

def refine_grid(landscape, n=2):
    '''Increase the resolution of a landscape grid by a factor of n.'''
    return repeat(repeat(landscape, n, axis=1), n, axis=0)

def image_to_landscape(image):
    '''Converts an image (in any RGB format) to a 2d array of ones and zeroes.

    Most image formats (including TIFF) requires PIL.

    Parameters
    ----------
    image : an image filename

    Returns
    -------
    landscape : a 2-d array of ones and zeroes, where ones correspond to darker
        shades in the original image, which should correspond to patch, while
        the lighter shades represent the matrix

    '''
    from scipy.ndimage import imread

    # maybe use flipud/fliplr here?
    M = imread(image).sum(axis=2)[::-1,:]
    M = around(M/M.max())
    return 1 - M.astype(int16)

def random_landscape(cover, frag, size, radius=1, norm='taxicab'):
    '''Generates random square landscapes with a given cover and fragmentation.

    Parameters
    ----------
    cover : integer smaller than size**2, total number of patch elements
    frag : float between 0 and 1 (both exclusive), gives the level of
        fragmentation, with 0 being a single clump and 1 a totally random
        (scattered) landscape
    size : integer, lenght of the landscape (total area is size squared)
    radius : integer, maximum distance at which an element is considered to be
        a neighbour of another
    norm : one of 'maximum' or 'taxicab', defines the norm used for measuring
        distances between elements of the landscape

    Returns
    -------
    landscape : a square 2-d array of zeroes and ones

    Reference
    ---------
    Lenore Fahrig, Relative Effects of Habitat Loss and Fragmentation on
    Population Extinction, The Journal of Wildlife Management, Vol. 61, No. 3
    (Jul., 1997), pp. 603-610

    '''
    from numpy.random import rand, randint

    if cover > size**2:
        raise ValueError('Cover (=%i) must be smaller than total area (=%i).' %
                (cover, size**2)) 
    M = zeros((size+2*radius, size+2*radius), dtype=int)
    n = 0

    if norm.lower() == 'maximum':
        while n < cover:
            i, j = randint(radius, size+radius, 2)
            if M[i, j]:
                continue
            if radius == 0 or any(M[i-radius:i+radius+1, j-radius:j+radius+1]):
                M[i, j] = 1
                n += 1
            elif rand() < frag:
                M[i, j] = 1
                n += 1

    elif norm.lower() == 'taxicab':
        indexes = []
        for x in range(-radius, radius+1):
            indexes += [(x, y) for y in range(-radius+abs(x), radius+1-abs(x))]
        indexes.remove((0, 0))
        indexes = array(indexes)
        while n < cover:
            i, j = randint(radius, size+radius, 2)
            if M[i, j]:
                continue
            if radius == 0 or any(M[i+indexes[:,0], j+indexes[:,1]]):
                M[i, j] = 1
                n += 1
            elif rand() < frag:
                M[i, j] = 1
                n += 1
    else:
        raise ValueError('Expected either "taxicab" or "maximum" for keyword '
                'arg "norm", got %s.' % norm)

    if radius == 0:
        return M
    return M[radius:-radius,radius:-radius]

def random_landscape_inv(cover, frag, size, radius=1, norm='taxicab'):
    land0 = 1 - random_landscape(size**2-cover, frag, size, radius, norm)
    count, labeled = statistics_landscape(land0, labels=True)

    npatches = len(count) - 1
    while npatches > 1./frag:
        p = count.argmin()

def popcount_patches(landscape, solution):
    '''Labels patches and sums total area and population in each one.

    Parameters
    ----------
    landscape : 2-d array of ones and zeroes
    solution : 2-d array containing the solution to the system

    Returns
    -------
    array : 2-d array where the first line is the area and the second is the
        total population. The first column is the total area and population in
        the matrix.

    '''
    count, labeled = statistics_landscape(landscape, labels=True)

    pops = []
    for i in range(0, len(count)):
        marked = where(labeled == i)
        pops.append(solution[marked].sum())
    return array([count, pops])

def statistics_landscape(landscape, labels=False):
    '''Labels patches and sums total area in each one.

    Parameters
    ----------
    landscape : 2-d array of ones and zeroes
    labels : if True, also return the labeled 2-d array landscape

    Returns
    -------
    array : 2-d array where the first line is the area and the second is the
        total population. The first column is the total area and population in
        the matrix.
    labeled : if parameter 'labels' is True, return a tuple (array, labeled),
        where the second element is the labeled 2-d array landscape

    '''
    labeled, npatches = label(landscape)
    count = []
    for i in range(0, npatches+1):
        marked = where(labeled == i)
        area = len(marked[0])
        count.append(area)

    if labels:
        return array(count), labeled
    return array(count)

def fit_exponential(count):
    from scipy.stats import kstest

    m = 1./mean(count)
    exp_cdf = lambda x: m * exp(-m*x)
    return kstest(count, exp_cdf)

def plot_landscape(landscape, solution, dx=dx, K=1.):
    #from matplotlib.pyplot import imshow, colorbar, ylabel, xlabel, yticks, xticks, subplot, cm
    extent = (0, solution.shape[1]*dx, 0, solution.shape[0]*dx)

    subplot(211)
    imshow(landscape, cmap=cm.binary, origin='lower', interpolation='none', extent=extent)
    ylabel('y (km)')
    locs, labels = xticks()
    xticks(locs, ['']*len(locs))

    subplot(212)
    imshow(solution, cmap=cm.jet, origin='lower', interpolation='none', extent=extent, vmin=0, vmax = K)
    xlabel('x (km)')
    ylabel('y (km)')
    
    colorbar()

def plot_density_area(pops, labels, xlog=True, ylog=False):
    for y, label in zip(pops, labels):
        plot(y[0][1:], y[1][1:]/y[0][1:], '.', label=label)

    xlabel('area')
    ylabel('avg. density')
    if xlog and ylog:
        loglog()
    elif xlog:
        xscale('log')
    elif ylog:
        yscale('log')
    legend(loc='best', frameon=False)

