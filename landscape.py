'''Single species, one type of patch and matrix - basic module.'''
import numpy as np
from collections import OrderedDict
from itertools import product as iproduct

try:
    from matplotlib import pyplot as plt
except ImportError:
    pass

par = OrderedDict([ 
    ('r', 0.1),
    ('K', 1.),
    ('mu', 0.03),
    ('Dp', 1e-4),
    ('Dm', 1e-3),
    # interface condition (=Dm/Dp)
    #('g', 1.), # no longer required
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


def solve_landscape(landscape, par, dx, f_tol=None, force_positive=False, verbose=True):
    r"""Find the stationary solution for a given landscape and set of parameters.

    Uses a Newton-Krylov solver with LGMRES sparse inverse method to find a
    stationary solution (or the solution to the elliptical problem) to the
    system of equations in 2 dimensions (x is a 2-d vector):

    .. math::
        u_t(x) &= D_p \nabla^2 u(x) + ru(1-u(x)/K) = 0 \text{ in a patch} \\
        v_t(x) &= D_m \nabla^2 v(x) - \mu v(x) = 0 \text{ in the matrix}

    Parameters
    ----------
    landscape : 2-d array of ints
        describe the landscape, with 1 on patches and 0 on matrix
    par : dict
        parameters (dict keys): 

        - r : reproductive rate on patches
        - K : carrying capacity on patches
        - mu : mortality rate in the matrix
        - Dp : diffusivity on patches
        - Dm : diffusivity in the matrix
        - g : habitat discontinuity parameter \gamma, usually less than one. See
          interface conditions below (optional)
        - alpha : habitat preference, only taken into account if g is not
          present. In that case, g is calculated as g = Dm * alpha /
          (Dp*(1-alpha))
        - left : (a, b, c): external boundary conditions at left border
        - right : (a, b, c): external boundary conditions at right border
        - top : (a, b, c): external boundary conditions at top border
        - bottom : (a, b, c): external boundary conditions at bottom border
    dx : float
        lenght of each edge
    f_tol : float
        tolerance for the residue, passed on to the solver routine.  Default is
        6e-6
    force_positive : bool
        make sure the solution is always non-negative - in a hacky way. Default
        False
    verbose : bool
        print residue of the solution and its maximum and minimum values

    Returns
    -------
    solution : 2-d array of the same shape of the landscape input 
        the solution


    .. rubric:: Boundary and interface conditions

    External boundaries are of the form

    .. math:: a \nabla u \cdot \hat{n} + b u + c = 0

    and may be different for left, right, top, bottom.  The derivative of u is
    taken along the normal to the boundary.

    The interfaces between patches and matrix are given by

    .. math::
        u(x) &= \gamma v(x) \\
        D_p \nabla u(x) \cdot \hat{n} &= D_m \nabla v(x) \cdot \hat{n}

    where u is in a patch and v is the solution in the matrix. Usually the
    discontinuity $\gamma$ is a result of different diffusivities and
    preference at the border (see Ovaskainen and Cornell 2003). In that case,
    given a preference $\alpha$ (between 0 and 1, exclusive) towards the patch,
    this parameter should be:

    .. math:: \gamma = \frac{D_m}{D_p} \frac{\alpha}{1-\alpha}

    This last condition is used in case $\gamma$ is not set. If $\alpha$ isn't
    set either, it's assumed $\alpha = 1/2$. These conditions are handled using
    an asymetric finite difference scheme for the 2nd derivative:

    .. math:: u_{xx}(x) = \frac{4}{3h^2} (u(x-h) - 3 u(x) + 2 u(x+h/2))

    At the interface, $u(x+h/2)$ and $v(x+h/2)$ must obey:

    .. math::
        u(x+h/2) &= \gamma v(x+h/2) \\
        D_p (u(x+h/2) - u(x))  &= D_m (v(x+h) - v(x+h/2))

    Solving this system, we arrive at the approximation at the interface:

    .. math:: u(x+h/2) = \frac{D_m v(x+h)+D_p u(x)}{D_p+D_m / \gamma}

    if u(x) is in a patch and v(x+h) is in the matrix, or

    .. math:: v(x+h/2) = \frac{D_m v(x)+D_p u(x+h)}{D_p \gamma +D_m}

    if v(x) is in the matrix and u(x+h) is in a patch.

    References
    ----------
    Ovaskainen, Otso, and Stephen J. Cornell. "Biased movement at a boundary
    and conditional occupancy times for diffusion processes." Journal of
    Applied Probability 40.3 (2003): 557-580.

    """
    from scipy.optimize import newton_krylov

    if 'g' not in par.keys():
        if 'alpha' not in par.keys():
            par['alpha'] = 0.5
        par['g'] = par['Dm']/par['Dp'] * par['alpha'] / (1-par['alpha'])

    # not a good idea
    #(r, K, mu, Dp, Dm, g, (al, bl, cl), (ar, br, cr), (at, bt, ct), (ab, bb,
    #    cb)) = par.values()
    (al, bl, cl) = par['left']
    (ar, br, cr) = par['right']
    (at, bt, ct) = par['top']
    (ab, bb, cb) = par['bottom']

    lin_term = par['r'] * landscape - par['mu'] * (1-landscape)
    sec_term = - landscape * par['r'] / par['K']
    D = landscape * par['Dp'] + (1-landscape) * par['Dm']

    Bxpm, Bxmp, Bypm, Bymp = find_interfaces(landscape)
    factor_pp = -1. + 2. * par['Dp']/(par['Dp']+par['Dm']/par['g'])
    factor_pm = -1. + 2. * par['Dm']/(par['Dp']+par['Dm']/par['g'])
    factor_mp = -1. + 2. * par['Dp']/(par['Dp']*par['g']+par['Dm'])
    factor_mm = -1. + 2. * par['Dm']/(par['Dp']*par['g']+par['Dm'])

    def residual(P):
        if force_positive:
            P = np.abs(P)
        d2x = np.zeros_like(P)
        d2y = np.zeros_like(P)

        d2x[1:-1,:] = P[2:,:] - 2*P[1:-1,:] + P[:-2,:]
        # external boundaries
        d2x[0,:] = P[1,:] - 2*P[0,:] + (-cl - al/dx * P[0,:])/(bl - al/dx)
        d2x[-1,:] = P[-2,:] - 2*P[-1,:] + (-cr + ar/dx * P[-1,:])/(br + ar/dx)
        # interface conditions
        d2x[:-1,:] += Bxpm * (P[:-1,:] * factor_pp + P[1:,:] * factor_pm) + \
                Bxmp * (P[:-1,:] * factor_mm + P[1:,:] * factor_mp)
        d2x[1:,:] += Bxpm * (P[:-1,:] * factor_mp + P[1:,:] * factor_mm) + \
                Bxmp * (P[:-1,:] * factor_pm + P[1:,:] * factor_pp)
        d2x[:-1,:] *= (Bxpm+Bxmp)*1./3. + Bxpm*Bxmp/3. + np.ones(Bxpm.shape)
        # can Bxpm*Bxmp be non-zero??

        d2y[:,1:-1] = P[:,2:] - 2*P[:,1:-1] + P[:,:-2]
        # external boundaries
        d2y[:,0] = P[:,1] - 2*P[:,0] + (-cb - ab/dx * P[:,0])/(bb - ab/dx)
        d2y[:,-1] = P[:,-2] - 2*P[:,-1] + (-ct + at/dx * P[:,-1])/(bt + at/dx)
        # interface conditions
        d2y[:,:-1] += Bypm * (P[:,:-1] * factor_pp + P[:,1:] * factor_pm) + \
                Bymp * (P[:,:-1] * factor_mm + P[:,1:] * factor_mp)
        d2y[:,1:] += Bypm * (P[:,:-1] * factor_mp + P[:,1:] * factor_mm) + \
                Bymp * (P[:,:-1] * factor_pm + P[:,1:] * factor_pp)
        d2y[:,:-1] *= (Bypm+Bymp)*1./3. + Bypm*Bymp/3. + np.ones(Bypm.shape)

        return D*(d2x + d2y)/dx/dx + lin_term*P + sec_term*P**2

    # solve
    guess = par['K'] * np.ones_like(landscape)
    sol = newton_krylov(residual, guess, method='lgmres', f_tol=f_tol)
    if force_positive:
        sol = np.abs(sol)
    if verbose:
        print('Residual: %e' % abs(residual(sol)).max())
        print('max. pop.: %f' % sol.max())
        print('min. pop.: %f' % sol.min())

    return sol


def solve_landscape_ntypes(landscape, par, dx, f_tol=None,
        force_positive=False, verbose=True):
    r"""Find the stationary solution for a landscape with many types of habitat.

    Uses a Newton-Krylov solver with LGMRES sparse inverse method to find a
    stationary solution (or the solution to the elliptical problem) to the
    system of equations in 2 dimensions (x is a 2-d vector):

    .. math:: 
        \frac{\partial u_i}{\partial t} = D_i \nabla^2 u_i + 
        r_i u_i\left(1-\frac{u}{K_i}\right) = 0

    Parameters
    ----------
    landscape : 2-d array of ints
        describe the landscape, with any number of habitat types
    par : dict
        parameters (dict keys):

        - r : growth rates (can be negative)
        - K : carrying capacities (cn be np.Inf)
        - mu : mortality rate in the matrix
        - D : diffusivities
        - g : dict of habitat discontinuities $\gamma_{ij}$ - see interface
          conditions below. The keys are tuples (i,j) of the habitat
          types indices (optional)
        - alpha : dict of habitat preferences, only taken into account if g is
          not present. In that case, $\gamma_{ij}$ is calculated as
          $\gamma_{ij} = D_j \alpha_{ij} / (D_i*(1-\alpha_{ij}))$ (optional)
        - left : (a, b, c): external boundary conditions at left border
        - right : (a, b, c): external boundary conditions at right border
        - top : (a, b, c): external boundary conditions at top border
        - bottom : (a, b, c): external boundary conditions at bottom border
    dx : float
        lenght of each edge
    f_tol : float
        tolerance for the residue, passed on to the solver routine.  Default is
        6e-6
    force_positive : bool
        make sure the solution is always non-negative - in a hacky way. Default
        False
    verbose : bool
        print residue of the solution and its maximum and minimum values

    Returns
    -------
    solution : 2-d array of the same shape of the landscape input 
        the solution


    .. rubric:: Boundary and interface conditions

    External boundaries are of the form

    .. math:: a \nabla u \cdot \hat{n} + b u + c = 0

    and may be different for left, right, top, bottom.  The derivative of u is
    taken along the normal to the boundary.

    The interfaces between patches and matrix are given by

    .. math::
        u_i(x) &= \gamma_{ij} u_j(x) \\
        D_i \nabla u_i(x) \cdot \hat{n} &= D_j \nabla u_j(x) \cdot \hat{n}

    Usually the discontinuity $\gamma_{ij}$ is a result of different
    diffusivities and preference at the border (see Ovaskainen and Cornell
    2003). In that case, given a preference $\alpha_{ij}$ (between 0 and 1,
    exclusive) towards $i$, this parameter should be:

    .. math:: \gamma_{ij} = \frac{D_j}{D_i} \frac{\alpha_{ij}}{1-\alpha_{ij}}

    This last condition is used in case $\gamma$ is not set. If $\alpha$ isn't
    set either, it's assumed $\alpha = 1/2$. Notice that $\alpha_{ij} +
    \alpha_{ji} = 1$, and so $\gamma_{ij} = \gamma_{ji}^{-1}$. To ensure this
    condition, the key (i,j) is always taken with $i>j$.

    These conditions are handled using an asymetric finite difference scheme
    for the 2nd derivative:

    .. math:: u_{xx}(x) = \frac{4}{3h^2} (u(x-h) - 3 u(x) + 2 u(x+h/2))

    At the interface, $u(x+h/2)$ and $v(x+h/2)$ must obey:

    .. math::
        u(x+h/2) &= \gamma v(x+h/2) \\
        D_p (u(x+h/2) - u(x))  &= D_m (v(x+h) - v(x+h/2))

    Solving this system, we arrive at the approximation at the interface:

    .. math:: u(x+h/2) = \frac{D_m v(x+h)+D_p u(x)}{D_p+D_m / \gamma}

    if u(x) is in a patch and v(x+h) is in the matrix, or

    .. math:: v(x+h/2) = \frac{D_m v(x)+D_p u(x+h)}{D_p \gamma +D_m}

    if v(x) is in the matrix and u(x+h) is in a patch.

    Example
    -------
    >>> # simple patch/matrix
    >>> from landscape import *
    >>> parn = OrderedDict([
        ('r', [-0.03, 0.1]),
        ('K', [np.Inf, 1.0]),
        ('D', [0.001, 0.0001]),
        ('left', [1.0, 0.0, 0.0]),
        ('right', [1.0, 0.0, 0.0]),
        ('top', [1.0, 0.0, 0.0]),
        ('bottom', [1.0, 0.0, 0.0])
        ])
    >>> l = np.zeros((100,100), dtype=int)
    >>> l[40:60, 40:60] = 1
    >>> sol = solve_landscape_ntypes(l, parn, dx)

    """
    from scipy.optimize import newton_krylov

    n = np.unique(landscape).astype(int)

    p = par.copy()
    if 'g' not in p.keys():
        p['g'] = {}
        if 'alpha' not in p.keys():
            p['alpha'] = {}
            for i, j in iproduct(n, repeat=2):
                if i > j:
                    p['alpha'][(i,j)] = 0.5
        for i, j in iproduct(n, repeat=2):
            if i > j:
                p['g'][(i,j)] = p['D'][j]/p['D'][i] * \
                                  p['alpha'][(i,j)] / (1-p['alpha'][(i,j)])

    # this ensures the consistency of the interface discontinuities
    # it ignores the values of g_ij with i < j, replacing it by 1/g_ji
    for i, j in iproduct(n, repeat=2):
        if i < j:
            p['g'][(i,j)] = 1/p['g'][(j,i)]

    (al, bl, cl) = p['left']
    (ar, br, cr) = p['right']
    (at, bt, ct) = p['top']
    (ab, bb, cb) = p['bottom']

    D = np.zeros_like(landscape)
    r = np.zeros_like(landscape)
    c = np.zeros_like(landscape)
    for i in n:
        li = np.where(landscape == i)
        D[li] = p['D'][i]
        r[li] = p['r'][i]
        c[li] = - p['r'][i] / p['K'][i]

    Bx, By = find_interfaces_ntypes(landscape)
    factor = {}
    for i, j in iproduct(n, repeat=2):
        if i > j:
            factor[(i,j)] = (
                -1. + 2. * p['D'][i]/(p['D'][i]+p['D'][j]/p['g'][(i,j)]),
                -1. + 2. * p['D'][j]/(p['D'][i]+p['D'][j]/p['g'][(i,j)]),
                -1. + 2. * p['D'][i]/(p['D'][i]*p['g'][(i,j)]+p['D'][j]),
                -1. + 2. * p['D'][j]/(p['D'][i]*p['g'][(i,j)]+p['D'][j])
                )

    def residual(P):
        if force_positive:
            P = np.abs(P)
        d2x = np.zeros_like(P)
        d2y = np.zeros_like(P)

        d2x[1:-1,:] = P[2:,:] - 2*P[1:-1,:] + P[:-2,:]
        # external boundaries
        d2x[0,:] = P[1,:] - 2*P[0,:] + (-cl - al/dx * P[0,:])/(bl - al/dx)
        d2x[-1,:] = P[-2,:] - 2*P[-1,:] + (-cr + ar/dx * P[-1,:])/(br + ar/dx)
        # interface conditions
        # TODO: probably something wrong here
        for (i,j), fac in factor.items():
            d2x[:-1,:][Bx[(i,j)]] += (P[:-1,:] * factor[(i,j)][0] + \
                    P[1:,:] * factor[(i,j)][1])[Bx[(i,j)]]
            d2x[:-1,:][Bx[(j,i)]] += (P[:-1,:] * factor[(i,j)][3] + \
                    P[1:,:] * factor[(i,j)][2])[Bx[(j,i)]]
            d2x[1:,:][Bx[(i,j)]] += (P[:-1,:] * factor[(i,j)][2] + \
                    P[1:,:] * factor[(i,j)][3])[Bx[(i,j)]]
            d2x[1:,:][Bx[(j,i)]] += (P[:-1,:] * factor[(i,j)][1] + \
                    P[1:,:] * factor[(i,j)][0])[Bx[(j,i)]]
            d2x[:-1,:][Bx[(i,j)]] *= 4/3 
            d2x[:-1,:][Bx[(j,i)]] *= 4/3 
            # assuming Bxpm*Bxmp is always zero (how could it not be??)

        d2y[:,1:-1] = P[:,2:] - 2*P[:,1:-1] + P[:,:-2]
        # external boundaries
        d2y[:,0] = P[:,1] - 2*P[:,0] + (-cb - ab/dx * P[:,0])/(bb - ab/dx)
        d2y[:,-1] = P[:,-2] - 2*P[:,-1] + (-ct + at/dx * P[:,-1])/(bt + at/dx)
        # interface conditions
        for (i,j), fac in factor.items():
            d2y[:-1,:][By[(i,j)]] += (P[:-1,:] * factor[(i,j)][0] + \
                    P[1:,:] * factor[(i,j)][1])[By[(i,j)]]
            d2y[:-1,:][By[(j,i)]] += (P[:-1,:] * factor[(i,j)][3] + \
                    P[1:,:] * factor[(i,j)][2])[By[(j,i)]]
            d2y[1:,:][By[(i,j)]] += (P[:-1,:] * factor[(i,j)][2] + \
                    P[1:,:] * factor[(i,j)][3])[By[(i,j)]]
            d2y[1:,:][By[(j,i)]] += (P[:-1,:] * factor[(i,j)][1] + \
                    P[1:,:] * factor[(i,j)][0])[By[(j,i)]]
            d2y[:-1,:][By[(i,j)]] *= 4/3 
            d2y[:-1,:][By[(j,i)]] *= 4/3 

        return D*(d2x + d2y)/dx/dx + r*P + c*P**2

    # solve
    guess = r.copy()
    guess[guess<=0] = 0
    guess[guess>0] = 1/((-c/r)[guess>0])
    sol = newton_krylov(residual, guess, method='lgmres', f_tol=f_tol)
    if force_positive:
        sol = np.abs(sol)
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

def find_interfaces_ntypes(landscape):
    '''Determines internal boundaries for landscapes with many habitat types.

    Parameters
    ----------
    landscape : 2-d array

    Returns
    -------
    Bx, By : tuple of two dicts
        each key is a tuple (i,j) corresponding to the numbers of the habitat
        types, and the value corresponds to the indices where a boundary
        between them appears, along either the x or y direction

    '''
    n = np.unique(landscape).astype(int)
    A = 2**(landscape.astype(int))
    Ax = A[1:,:] - A[:-1,:]
    Ay = A[:,1:] - A[:,:-1]
    Bx = {}
    By = {}
    for i, j in iproduct(n, repeat=2):
        if i != j:
            Bx[(i,j)] = np.where(Ax == 2**i - 2**j)
            By[(i,j)] = np.where(Ay == 2**i - 2**j)

    return Bx, By

def solve_multiple_parameters(variables, values, landscape, par, dx,
        f_tol=None, force_positive=False, verbose=True, multiprocess=True):
    '''Solve given landscape for several combinations of parameters.

    Solves a given landscape with a set of common parameters and for a range of
    values for some variables, optionally using multiprocessing to speed it up.

    Parameters
    ----------
    variables : list of strings
        names of the varied parameters
    values : list of lists (or tuples)
        each item contains a list of values corresponding to the parameters
        given in the `variables` list
    landscape : 2-d array
        zeroes and ones describing the landscape
    par : ordered dict
        common values for all the problem parameters. See documentation for
        `solve_landscape()`
    dx: float
        length of each edge
    f_tol : float
        tolerance for the residue, passed on to the solver routine. Default is
        6e-6
    verbose : bool
        print residue of the solution and its maximum and minimum values.
        Notice that the order of appearance of each output is not the same as
        the input if multiprocess is True.
    multiprocess : bool
        determines whether to use multiprocessing to use multiple cores. True
        by default, in which case the total number of CPUs minus one are used

    Returns
    -------
    solutions: list of 2-d arrays
        solutions of each set of parameters, in the same ordering of the input
        values

    Example
    -------
    >>> from landscape import *
    >>> lA = image_to_landscape('landA.tif')
    >>> D = np.arange(0.01, 0.05, 0.01)
    >>> r = np.arange(0.1, 0.5, 0.1)
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
        force_positive=True, multiprocess=True):
    '''Solve several landscapes.

    Solves a set of landscape with a given set of parameters, optionally using
    multiprocessing to speed things up.

    Parameters
    ----------
    landscape : list of 2-d arrays
        zeroes and ones describing the landscape
    par : ordered dict
        values for all the problem parameters. See documentation for
        `solve_landscape()`
    dx : float
        lenght of each edge
    f_tol : float
        tolerance for the residue, passed on to the solver routine. Default is
        6e-6
    verbose : bool
        print residue of the solution and its maximum and minimum values.
        Notice that the order of appearance of each output is not the same as
        the input if multiprocess is True.
    multiprocess : bool
        whether to use multiprocessing to use multiple cores. True by default,
        in which case the total number of CPUs minus one are used

    Returns
    -------
    solutions : list os 2-d arrays
        solutions to each landscape, in the same ordering of the input values

    Example
    -------
    >>> from landscape import *
    >>> l = [ random_landscape(i*100, 0.8, 100) for i in range(30,70,10) ]
    >>> sols = solve_multiple_landscapes(l, par, dx)

    '''
    from functools import partial
    # this is not compatible with python 2.6 when using multiprocessing, due to
    # bug http://bugs.python.org/issue5228
    solve_landscape_wrapper = partial(solve_landscape, par=par, dx=dx,
            f_tol=f_tol, force_positive=force_positive, verbose=verbose)

    if multiprocess:
        from multiprocessing import Pool, cpu_count
        cpus = cpu_count()
        pool = Pool(cpus if cpus == 1 else cpus - 1)
        solutions = pool.map(solve_landscape_wrapper, landscapes)
        pool.close()
        pool.join()
    else:
        solutions = map(solve_landscape_wrapper, landscapes)

    return solutions

def refine_grid(landscape, n=2):
    '''Increase the resolution of a landscape grid by a factor of n.'''
    return np.repeat(np.repeat(landscape, n, axis=1), n, axis=0)

def image_to_landscape(image):
    '''Converts an image (in any RGB format) to a 2d array of ones and zeroes.

    Most image formats (including TIFF) requires PIL.

    Parameters
    ----------
    image : string
        image filename

    Returns
    -------
    landscape : 2-d array 
        ones and zeroes, where ones correspond to darker shades in the original
        image, which should correspond to patch, while the lighter shades
        represent the matrix

    '''
    from scipy.ndimage import imread

    # maybe use flipud/fliplr here?
    M = imread(image).sum(axis=2)[::-1,:]
    M = np.around(M/M.max())
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

    References
    ----------
    Lenore Fahrig, Relative Effects of Habitat Loss and Fragmentation on
    Population Extinction, The Journal of Wildlife Management, Vol. 61, No. 3
    (Jul., 1997), pp. 603-610

    '''
    from numpy.random import rand, randint

    if cover > size**2:
        raise ValueError('Cover (=%i) must be smaller than total area (=%i).' %
                (cover, size**2)) 
    M = np.zeros((size+2*radius, size+2*radius), dtype=int)
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
        indexes = np.array(indexes)
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
    landscape : 2-d array 
        ones and zeroes representing patches and matrix
    solution : 2-d array
        solution to the system

    Returns
    -------
    array : 2-d array 
        the first line is the area and the second is the total population. The
        first column is the total area and population in the matrix.

    '''
    count, labeled = statistics_landscape(landscape, labels=True)

    pops = []
    for i in range(0, len(count)):
        marked = where(labeled == i)
        pops.append(solution[marked].sum())
    return np.array([count, pops])

def statistics_landscape(landscape, labels=False):
    '''Labels patches and sums total area in each one.

    Parameters
    ----------
    landscape : 2-d array
        ones and zeroes representing patches and matrix
    labels : bool
        if True, return the labeled 2-d array landscape

    Returns
    -------
    array : 2-d array 
        the first line is the area and the second is the total population. The
        first column is the total area and population in the matrix.
    labeled : tuple
        if `labels` is True, returns a tuple (array, labeled), where the second
        element is the labeled 2-d array landscape

    '''
    from scipy.ndimage import label
    labeled, npatches = label(landscape)
    count = []
    for i in range(0, npatches+1):
        marked = np.where(labeled == i)
        area = len(marked[0])
        count.append(area)

    if labels:
        return np.array(count), labeled
    return np.array(count)

def fit_exponential(count):
    from scipy.stats import kstest

    m = 1./np.mean(count)
    exp_cdf = lambda x: m * exp(-m*x)
    return kstest(count, exp_cdf)

def plot_landscape(landscape, solution, dx=dx, K=1.):
    #from matplotlib.pyplot import imshow, colorbar, ylabel, xlabel, yticks, xticks, subplot, cm
    extent = (0, solution.shape[1]*dx, 0, solution.shape[0]*dx)

    plt.subplot(211)
    plt.imshow(landscape, cmap=plt.cm.binary, origin='lower',
               interpolation='none', extent=extent)
    plt.ylabel('y (km)')
    locs, labels = plt.xticks()
    plt.xticks(locs, ['']*len(locs))

    plt.subplot(212)
    plt.imshow(solution, cmap=plt.cm.jet, origin='lower', interpolation='none',
               extent=extent, vmin=0, vmax = K)
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')

    plt.colorbar()

def plot_density_area(pops, labels, xlog=True, ylog=False):
    for y, label in zip(pops, labels):
        plt.plot(y[0][1:], y[1][1:]/y[0][1:], '.', label=label)

    plt.xlabel('area')
    plt.ylabel('avg. density')
    if xlog and ylog:
        plt.loglog()
    elif xlog:
        plt.xscale('log')
    elif ylog:
        plt.yscale('log')
    plt.legend(loc='best', frameon=False)

