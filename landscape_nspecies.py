'''Multi-species, one type of patch and matrix.'''
from landscape import *

parn = OrderedDict([
    ('n', 2),
    ('r', [0.1, 0.1]),
    ('alpha', [[-1., -0.1], [-0.2, -0.5]]),
    ('mu', [0.05, 0.03]),
    ('Dp', [1e-4, 1e-5]),
    ('Dm', [1e-3, 1e-3]),
    # interface condition (=Dm/Dp ??)
    ('g', [1., 1.]),
    # boundary conditions
    ('left', [1., 0., 0.]),
    ('right', [1., 0., 0.]),
    ('top', [1., 0., 0.]),
    ('bottom', [1., 0., 0.])
    ])

def solve_landscape_nspecies(landscape, par, dx, f_tol=None, verbose=True):
    '''Find the stationary solution for a given landscape and set of parameters.

    Uses a Newton-Krylov solver with LGMRES sparse inverse method to find a
    stationary solution (or the solution to the elliptical problem) to the
    system of 2n equations in 2 dimensions (x is a 2-d vector):

    .. math::
        \\frac{\partial u_i}{\partial t} &= D_p \\nabla^2 u_i + r_i u_i (1-\sum_{j=1}^n \\alpha_j u_j) = 0 \\text{ in a patch} \\\\
        \\frac{\partial v_i}{\partial t} &= D_m \\nabla^2 v_i - \mu_i v_i = 0 \\text{ in the matrix}

    Parameters
    ----------
    landscape : a 2-d array (of ints) describing the landscape, with 1 on
        patches and 0 on matrix
    par : a ordered dict containing parameters in the following order:
        n: number of species
        r: list of reproductive rates on patches
        alpha: matrix of interaction parameters on patches (diagonals are minus the inverse of carrying capacity) 
        mu: list of mortality rates in the matrix
        Dp: list of diffusivities on patches
        Dm: list of diffusivities in the matrix
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
    solution : 2-d array of the same shape of the landscape input containing
        the solution

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
        u_xx(x) = (4/3/h**2) (u(x-h) - 3 u(x) + 2 u(x+h/2))

    with the approximations at the interface:

    .. math::
        u(x+h/2) = (Dm*v(x+h)+Dp*u(x))/(Dp+Dm*g)

    if u(x) is in a patch and v(x+h) is in the matrix, or

    .. math::
        v(x+h/2) = g*(Dm*v(x)+Dp*u(x+h))/(Dp+Dm*g)

    if v(x) is in the matrix and u(x+h) is in a patch.

    '''
    from scipy.optimize import newton_krylov

    # change this ugly hack for another
    (n, r, alpha, mu, Dp, Dm, g, (al, bl, cl), (ar, br, cr), (at, bt, ct), (ab, bb,
        cb)) = par.values()
    r = array(r)
    alpha = array(alpha)
    mu = array(mu)
    Dp = array(Dp)
    Dm = array(Dm)
    g = array(g)

    #lin_term = array([ r[i] * landscape - mu[i] * (1-landscape) for i in range(n) ])
    #sec_term = array([ landscape * r[i] * alpha[i,:] for i in range(n) ])
    #D = array([ landscape * Dp[i] + (1-landscape) * Dm[i] for i in range(n) ])
    lin_term = r[:,None,None] * landscape - mu[:,None,None] * (1-landscape)
    sec_term = landscape * r[:,None,None] * alpha[:,:,None,None]
    D = landscape * Dp[:,None,None] + (1-landscape) * Dm[:,None,None]

    Bxpm, Bxmp, Bypm, Bymp = find_interfaces(landscape)
    factor_pp = -1. + 2. * Dp/(Dp+Dm*g)
    factor_pm = -1. + 2. * Dm/(Dp+Dm*g)
    factor_mp = -1. + 2. * g * Dp/(Dp+Dm*g)
    factor_mm = -1. + 2. * g * Dm/(Dp+Dm*g)

    def residual(N):
        res = []
        # loops are for lazy people
        for i, P in enumerate(N):
            d2x = zeros_like(P)
            d2y = zeros_like(P)

            d2x[1:-1,:] = P[2:,:] - 2*P[1:-1,:] + P[:-2,:]
            # external boundaries
            d2x[0,:] = P[1,:] - 2*P[0,:] + (-cl - al/dx * P[0,:])/(bl - al/dx)
            d2x[-1,:] = P[-2,:] - 2*P[-1,:] + (-cr + ar/dx * P[-1,:])/(br + ar/dx)
            # interface conditions
            d2x[:-1,:] += Bxpm * (P[:-1,:] * factor_pp[i] + P[1:,:] * factor_pm[i]) + \
                    Bxmp * (P[:-1,:] * factor_mm[i] + P[1:,:] * factor_mp[i])
            d2x[1:,:] += Bxpm * (P[:-1,:] * factor_mp[i] + P[1:,:] * factor_mm[i]) + \
                    Bxmp * (P[:-1,:] * factor_pm[i] + P[1:,:] * factor_pp[i])
            d2x[:-1,:] *= (Bxpm+Bxmp)*1./3. + Bxpm*Bxmp/3. + ones(Bxpm.shape)

            d2y[:,1:-1] = P[:,2:] - 2*P[:,1:-1] + P[:,:-2]
            # external boundaries
            d2y[:,0] = P[:,1] - 2*P[:,0] + (-cb - ab/dx * P[:,0])/(bb - ab/dx)
            d2y[:,-1] = P[:,-2] - 2*P[:,-1] + (-ct + at/dx * P[:,-1])/(bt + at/dx)
            # interface conditions
            d2y[:,:-1] += Bypm * (P[:,:-1] * factor_pp[i] + P[:,1:] * factor_pm[i]) + \
                    Bymp * (P[:,:-1] * factor_mm[i] + P[:,1:] * factor_mp[i])
            d2y[:,1:] += Bypm * (P[:,:-1] * factor_mp[i] + P[:,1:] * factor_mm[i]) + \
                    Bymp * (P[:,:-1] * factor_pm[i] + P[:,1:] * factor_pp[i])
            d2y[:,:-1] *= (Bypm+Bymp)*1./3. + Bypm*Bymp/3. + ones(Bypm.shape)

            res.append(D[i]*(d2x + d2y)/dx/dx + lin_term[i]*P + P * (sec_term[i] * N).sum(axis=0))

        return array(res)

    # solve
    guess = array([ -1./alpha[i][i] * ones_like(landscape) for i in range(n) ])
    sol = newton_krylov(residual, guess, method='lgmres', f_tol=f_tol)
    if verbose:
        print('Residual: %e' % abs(residual(sol)).max())
        print('max. pop.: %f' % sol.max())
        print('min. pop.: %f' % sol.min())

    return sol

def plotall(landscape, solutions):
    n = len(solutions)
    K = ceil(max([ s.max() for s in solutions ]))
    extent = (0, landscape.shape[1]*dx, 0, landscape.shape[0]*dx)

    subplot(1, n+1, 1)
    imshow(landscape, cmap=cm.binary, origin='lower', interpolation='none', extent=extent)

    for i, s in enumerate(solutions):
        subplot(1, n+1, i+2)
        imshow(s, cmap=cm.jet, origin='lower', interpolation='none', extent=extent, vmin=0, vmax = K)

