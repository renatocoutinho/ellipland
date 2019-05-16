landscape <- function (land, p, dx, f_tol=NULL, force_positive=FALSE, verbose=TRUE){
    # single-species, habitat/matrix only
    #
    # example:
    # p <- list(
    #         list('r', 0.1),
    #         list('K', 1.0),
    #         list('mu', 0.03),
    #         list('Dp', 0.0001),
    #         list('Dm', 0.001),
    #         list('left', c(1.0, 0.0, 0.0)),
    #         list('right', c(1.0, 0.0, 0.0)),
    #         list('top', c(1.0, 0.0, 0.0)),
    #         list('bottom', c(1.0, 0.0, 0.0))
    #         )
    # sol <- landscape('landA.txt', p, 0.1)

    library(reticulate)
    # loads python library
    source_python('landscape.py')
    # loads landscape file
    l <- np.loadtxt(land)
    # converts parameters to OrderedDict
    pars  <- OrderedDict(p)
    # actually runs the solver
    s <- solve_landscape(l, pars, dx, f_tol, force_positive, verbose)
}

landscape_ntypes <- function (land, p, dx, f_tol=NULL, force_positive=FALSE, verbose=TRUE){
    # single-species, n types of habitat
    #
    # example with two types of habitat (matrix and patch):
    # p <- list(
    #         list('r', c(-0.03, 0.1)),
    #         list('K', c(np.Inf, 1.0)),
    #         list('D', c(0.001, 0.0001)),
    #         list('left', c(1.0, 0.0, 0.0)),
    #         list('right', c(1.0, 0.0, 0.0)),
    #         list('top', c(1.0, 0.0, 0.0)),
    #         list('bottom', c(1.0, 0.0, 0.0))
    #         )
    # sol <- landscape_ntypes('landA.txt', p, 0.1)

    library(reticulate)
    # loads python library
    source_python('landscape.py')
    # loads landscape file
    l <- np.loadtxt(land)
    # converts parameters to OrderedDict
    pars  <- OrderedDict(p)
    # actually runs the solver
    s <- solve_landscape_ntypes(l, pars, dx, f_tol, force_positive, verbose)
}

