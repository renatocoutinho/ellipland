landscape_nspecies_gen = function (land, p, dx, f_tol, verbose){
#    Parameters
#    ----------
#    landscape : filename of a 2-d array (of ints) describing the landscape,
#       with 1 on patches and 0 on matrix
#    par : a ordered dict containing parameters in the following order:
#        rp: list of reproductive rates on patches
#        rm: list of reproductive rates in the matrix
#        alphap: matrix of interaction parameters on patches (diagonals are
#           minus the inverse of carrying capacity) 
#        alpham: matrix of interaction parameters in the matrix (diagonals are
#           minus the inverse of carrying capacity). Notice that the
#           density-dependence is zero if the linear term is negative.
#        Dp: list of diffusivities on patches
#        Dm: list of diffusivities in the matrix
#        g: habitat preference parameter \gamma, usually less than one. See
#            interface conditions below
#        left: (a, b, c): external boundary conditions at left border
#        right: (a, b, c): external boundary conditions at right border
#        top: (a, b, c): external boundary conditions at top border
#        bottom: (a, b, c): external boundary conditions at bottom border
#    dx : lenght of each edge
#    f_tol : float, tolerance for the residue, passed on to the solver routine.
#        Default is 6e-6
#    verbose : print residue of the solution and its maximum and minimum values
#
#    Returns
#    -------
#    solution : 3-d array with the number of species times the same shape of
#    the landscape input containing the solution
#

    library(reticulate)
    # loads python library
    source_python('landscape_nspecies_gen.py')
    # loads landscape file
    l <- loadtxt(land)
    # converts parameters to OrderedDict
    pars  <- OrderedDict(p)
    if (missing(f_tol))
        f_tol <- NULL
    if (missing(verbose))
        verbose <- TRUE
    # actually runs the solver
    s <- solve_landscape_nspecies(l, pars, dx, f_tol)
}


# (working) example
p = list(
         list('rp', c(0.1, 0.0)),
         list('rm', c(-0.01, 0.1)),
         list('alphap', matrix(c(1.0, 1.0, 2.0, 2.0), nrow=2)),
         list('alpham', matrix(c(1.0, 1.0, 2.0, 2.0), nrow=2)),
         list('Dp', c(0.0005, 0.005)),
         list('Dm', c(0.005, 0.005)),
         list('g', c(0.1, 1.0)),
         list('left', c(1.0, 0.0, 0.0)),
         list('right', c(1.0, 0.0, 0.0)),
         list('top', c(1.0, 0.0, 0.0)),
         list('bottom', c(1.0, 0.0, 0.0))
         )

# sol <- landscape_nspecies_gen('landA.txt', p, 0.1)

